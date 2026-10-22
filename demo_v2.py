#!/usr/bin/env python3
"""DEFOM-Stereo ROS1 node (full teacher model) for the BeeX side-by-side stereo stream.

    python3 demo_v2.py --restore_ckpt checkpoints/defomstereo_vitl_middlebury.pth

Subscribes to a side-by-side CompressedImage (left | right), runs DEFOM-Stereo
and publishes: colourised depth (CompressedImage), raw disparity and metric
depth (32FC1), CameraInfo, and a PointCloud2 built from the metric depth.
Only the newest frame is processed, so the node stays live when inference is
slower than the camera. demo_v3.py runs a distilled student through this same
node.

vitl on an RTX 4070 at 800x600: ~0.8 s/frame at 32 iterations, most of it the
encoder, so fewer iterations save little.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

import argparse
import threading
import time

import cv2
import numpy as np
import rospy
import torch
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
import tf2_ros
from tf.transformations import quaternion_matrix

from core.defom_stereo import DEFOMStereo
from core.utils.utils import InputPadder


def repo_path(path):
    """Relative paths resolve against the cwd if they exist there, else the repo root."""
    path = Path(path)
    return path if path.is_absolute() or path.exists() else ROOT / path


def load_teacher_weights(model, ckpt_path):
    """Load a DEFOM checkpoint on the CPU (the model is moved to the GPU afterwards)."""
    state = torch.load(str(repo_path(ckpt_path)), map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model


def to_tensor(bgr, device):
    """BGR uint8 HxWx3 -> 1x3xHxW float RGB (DEFOM expects RGB)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device)


COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def colorise(values, valid, lo, hi, colormap, invert):
    """Colour-map `values` over [lo, hi] (default: 2-98th percentile); invalid pixels black."""
    out = np.zeros(values.shape + (3,), dtype=np.uint8)
    if not valid.any():
        return out
    if lo is None or hi is None:
        low, high = (float(v) for v in np.percentile(values[valid], [2.0, 98.0]))
    else:
        low, high = float(lo), float(hi)
    norm = np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    coloured = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    out[valid] = coloured[valid]
    return out


def float_image(header, array):
    msg = Image()
    msg.header = header
    msg.height, msg.width = array.shape
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = 4 * array.shape[1]
    msg.data = array.astype(np.float32).tobytes()
    return msg


def build_pointcloud(header, depth, intrinsics, color_bgr, rotation, translation, stride, with_color):
    """Reproject metric depth (pinhole: u right, v down, z forward) into a
    PointCloud2, transformed by (rotation, translation) into header.frame_id."""
    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    grid_u, grid_v = np.meshgrid(np.arange(0, width, stride), np.arange(0, height, stride))
    z = depth[grid_v, grid_u]
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        return None
    x = (grid_u.astype(np.float32) - cx) * z / fx
    y = (grid_v.astype(np.float32) - cy) * z / fy
    points = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)
    points = points @ rotation.T.astype(np.float32) + translation.astype(np.float32)

    fields = [PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1),
              PointField("z", 8, PointField.FLOAT32, 1)]
    dtype = [("x", np.float32), ("y", np.float32), ("z", np.float32)]
    if with_color and color_bgr is not None:
        fields.append(PointField("rgb", 12, PointField.FLOAT32, 1))
        dtype.append(("rgb", np.float32))
    record = np.zeros(points.shape[0], dtype=dtype)
    record["x"], record["y"], record["z"] = points[:, 0], points[:, 1], points[:, 2]
    if len(dtype) == 4:
        bgr = color_bgr[grid_v, grid_u][valid]
        # PCL packing: 0x00RRGGBB as uint32, reinterpreted as float32.
        packed = (bgr[:, 2].astype(np.uint32) << 16) | (bgr[:, 1].astype(np.uint32) << 8) | bgr[:, 0].astype(np.uint32)
        record["rgb"] = packed.view(np.float32)

    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, points.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 4 * len(dtype)
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True
    msg.data = record.tobytes()
    return msg


class DefomStereoNode(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.lock = threading.Lock()
        self.latest = None          # newest unprocessed message
        self.dropped = 0
        self.processed = 0
        self.last_report = time.time()
        self.calib_done = False
        self.colormap = COLORMAPS[args.colormap]
        self.intrinsics = None      # (fx, fy, cx, cy) at processing resolution
        self.fx_baseline = None     # metres * pixels: depth = fx_baseline / disparity

        self.model = self._build_model()
        rospy.loginfo("Model ready, %.2f GB on GPU",
                      torch.cuda.memory_allocated() / 1e9 if self.device.type == "cuda" else 0.0)

        self.pub_vis = rospy.Publisher(args.output_topic, CompressedImage, queue_size=1)
        self.pub_disp = rospy.Publisher(args.disparity_topic, Image, queue_size=1)
        self.pub_depth = rospy.Publisher(args.depth_topic, Image, queue_size=1)
        self.pub_debug = (rospy.Publisher(args.debug_topic, CompressedImage, queue_size=1)
                          if args.publish_debug else None)
        self.pub_info = rospy.Publisher(args.camera_info_topic, CameraInfo, queue_size=1, latch=True)
        self.pub_points = (rospy.Publisher(args.points_topic, PointCloud2, queue_size=1)
                           if args.publish_points else None)
        self.tf_buffer = tf2_ros.Buffer() if args.publish_points else None
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) if args.publish_points else None
        self._extrinsic_cache = {}   # source_frame -> (rotation, translation, fetched_at)

        self.sub = rospy.Subscriber(args.input_topic, CompressedImage, self.on_image,
                                    queue_size=1, buff_size=2 ** 24, tcp_nodelay=True)
        rospy.loginfo("Subscribed to %s", args.input_topic)
        if args.warmup:
            self.warmup()

    def _build_model(self):
        """Overridden by demo_v3.py to load a distilled student instead."""
        rospy.loginfo("Loading %s (%s) ...", self.args.restore_ckpt, self.args.dinov2_encoder)
        model = load_teacher_weights(DEFOMStereo(self.args), self.args.restore_ckpt)
        return model.to(self.device).eval()

    # ---------------------------------------------------- calibration

    def resolve_calibration(self, width, height):
        """Scale the intrinsics from --calib_width/--calib_height to the processed
        resolution (they scale linearly with image size) and derive fx*B."""
        args = self.args
        if args.fx <= 0.0:
            self.fx_baseline = args.fx_baseline if args.fx_baseline > 0.0 else None
            if self.fx_baseline:
                rospy.loginfo("Metric depth enabled: fx*B = %.4f m.px", self.fx_baseline)
            else:
                rospy.logwarn("No calibration: publishing DISPARITY only. Pass --fx/--baseline_m or --fx_baseline.")
            return

        calib_w = args.calib_width if args.calib_width > 0 else width
        calib_h = args.calib_height if args.calib_height > 0 else height
        sx, sy = float(width) / calib_w, float(height) / calib_h
        fx, fy = args.fx * sx, (args.fy if args.fy > 0 else args.fx) * sy
        cx = (args.cx if args.cx >= 0 else 0.5 * calib_w) * sx
        cy = (args.cy if args.cy >= 0 else 0.5 * calib_h) * sy
        self.intrinsics = (fx, fy, cx, cy)
        if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
            rospy.loginfo("Scaled intrinsics %dx%d -> %dx%d: fx %.3f cx %.3f cy %.3f",
                          calib_w, calib_h, width, height, fx, cx, cy)
        if not (0 <= cx < width and 0 <= cy < height):
            rospy.logerr("Principal point (%.1f, %.1f) is outside the %dx%d image: --calib_width/--calib_height "
                         "do not match the intrinsics, so depth will be wrong.", cx, cy, width, height)

        if args.fx_baseline > 0.0:
            self.fx_baseline = args.fx_baseline
        elif args.baseline_m > 0.0:
            self.fx_baseline = fx * args.baseline_m
        else:
            rospy.logwarn("Intrinsics but no baseline: publishing DISPARITY only. Add --baseline_m.")
            return
        rospy.loginfo("Metric depth enabled: fx*B = %.4f m.px", self.fx_baseline)

    def get_extrinsic(self, source_frame):
        """Rotation/translation from source_frame into --points_frame_id via live TF,
        cached for --points_tf_refresh seconds (the mount is rigid)."""
        target = self.args.points_frame_id
        cached = self._extrinsic_cache.get(source_frame)
        now = rospy.get_time()
        if cached is not None and now - cached[2] < self.args.points_tf_refresh:
            return cached[0], cached[1]
        try:
            transform = self.tf_buffer.lookup_transform(target, source_frame, rospy.Time(0),
                                                        rospy.Duration(self.args.points_tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(10.0, "No TF from '%s' to '%s' (%s): no point cloud this frame",
                                   source_frame, target, exc)
            return cached[:2] if cached is not None else (None, None)
        q, t = transform.transform.rotation, transform.transform.translation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
        translation = np.array([t.x, t.y, t.z], dtype=np.float32)
        self._extrinsic_cache[source_frame] = (rotation, translation, now)
        return rotation, translation

    def publish_camera_info(self, header, width, height):
        if self.intrinsics is None:
            return
        fx, fy, cx, cy = self.intrinsics
        info = CameraInfo()
        info.header = header
        info.width, info.height = width, height
        info.distortion_model = "plumb_bob"
        info.D = [0.0] * 5
        info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]   # left camera: Tx = 0
        self.pub_info.publish(info)

    # ---------------------------------------------------------------- input

    def on_image(self, msg):
        with self.lock:
            if self.latest is not None:
                self.dropped += 1
            self.latest = msg

    def take_latest(self):
        with self.lock:
            msg, self.latest = self.latest, None
        return msg

    def split_stereo(self, msg):
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn_throttle(5.0, "Failed to decode CompressedImage (format=%s)", msg.format)
            return None, None
        height, width = image.shape[:2]
        if width % 2:
            rospy.logwarn_throttle(5.0, "Odd width %d, cannot split evenly", width)
            return None, None
        if self.args.expect_width > 0 and width != self.args.expect_width:
            rospy.logwarn_throttle(10.0, "Frame is %dx%d, expected width %d", width, height, self.args.expect_width)
        left, right = image[:, :width // 2], image[:, width // 2:]
        return (right, left) if self.args.swap_lr else (left, right)

    # ------------------------------------------------------------ inference

    def warmup(self):
        dummy = np.zeros((self.args.warmup_height, self.args.warmup_width, 3), np.uint8)
        started = time.time()
        self.infer(dummy, dummy)
        rospy.loginfo("Warmup done in %.2f s", time.time() - started)

    def infer(self, left, right):
        full_h, full_w = left.shape[:2]
        scale = self.args.input_scale
        if scale != 1.0:
            left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image1, image2 = to_tensor(left, self.device), to_tensor(right, self.device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        with torch.no_grad():
            disp = self.model(image1, image2, iters=self.args.valid_iters,
                              scale_iters=self.args.scale_iters, test_mode=True)
        disp = padder.unpad(disp).cpu().squeeze().numpy().astype(np.float32)
        if scale != 1.0:
            # Disparity is in pixels: rescale the values as well as the raster.
            disp = cv2.resize(disp, (full_w, full_h), interpolation=cv2.INTER_LINEAR) / scale
        return disp

    # --------------------------------------------------------------- output

    def publish(self, header, left, disp):
        valid = np.isfinite(disp) & (disp > self.args.min_disparity)
        self.pub_disp.publish(float_image(header, disp))
        self.publish_camera_info(header, disp.shape[1], disp.shape[0])

        if self.fx_baseline is not None:
            depth = np.zeros_like(disp)
            np.divide(self.fx_baseline, disp, out=depth, where=valid)
            depth[~valid] = 0.0
            if self.args.max_depth_m > 0:
                depth[depth > self.args.max_depth_m] = 0.0
            self.pub_depth.publish(float_image(header, depth))
            field, field_valid, invert = depth, valid & (depth > 0), True   # near = warm
            if self.pub_points is not None and self.intrinsics is not None:
                rotation, translation = self.get_extrinsic(header.frame_id)
                if rotation is not None:
                    cloud_header = Header(stamp=header.stamp, frame_id=self.args.points_frame_id)
                    cloud = build_pointcloud(cloud_header, depth, self.intrinsics, left, rotation, translation,
                                             self.args.points_stride, self.args.points_color)
                    if cloud is not None:
                        self.pub_points.publish(cloud)
        else:
            if self.pub_points is not None:
                rospy.logwarn_throttle(30.0, "Point cloud needs metric depth (--fx/--baseline_m or --fx_baseline)")
            field, field_valid, invert = disp, valid, False

        vis = colorise(field, field_valid, self.args.vis_min, self.args.vis_max, self.colormap, invert)
        self.pub_vis.publish(self.to_compressed(header, vis))
        if self.pub_debug is not None:
            side = cv2.resize(vis, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.pub_debug.publish(self.to_compressed(header, np.hstack([left, side])))

    def to_compressed(self, header, bgr):
        msg = CompressedImage()
        msg.header = header
        if self.args.vis_format == "png":
            msg.format = "png"
            ok, buf = cv2.imencode(".png", bgr)
        else:
            msg.format = "jpeg"
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality])
        if ok:
            msg.data = buf.tobytes()
        else:
            rospy.logwarn_throttle(5.0, "Failed to encode output image")
        return msg

    # ----------------------------------------------------------------- loop

    def spin(self):
        rate = rospy.Rate(self.args.poll_hz)
        while not rospy.is_shutdown():
            msg = self.take_latest()
            if msg is None:
                rate.sleep()
                continue
            started = time.time()
            left, right = self.split_stereo(msg)
            if left is None:
                continue
            if not self.calib_done:   # on the first frame, against the real stream size
                self.resolve_calibration(left.shape[1], left.shape[0])
                self.calib_done = True
            try:
                disp = self.infer(left, right)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                rospy.logerr_throttle(10.0, "CUDA OOM -- try --input_scale 0.5: %s", exc)
                continue
            self.publish(msg.header, left, disp)
            self.processed += 1

            now = time.time()
            if now - self.last_report >= self.args.report_period:
                rospy.loginfo("%.2f Hz | last %.0f ms | processed %d | dropped %d | disp %.1f-%.1f px",
                              self.processed / max(1e-6, now - self.last_report), 1000.0 * (now - started),
                              self.processed, self.dropped, float(np.nanmin(disp)), float(np.nanmax(disp)))
                self.processed = 0
                self.last_report = now


# --------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------


def add_node_args(parser, node_name):
    """Topics, camera geometry, point cloud, runtime and visualisation flags,
    shared by demo_v2.py and demo_v3.py."""
    ros = parser.add_argument_group("ROS")
    ros.add_argument('--input_topic', default='/ikan/explore3d/stereo/compressed', help='side-by-side stereo')
    ros.add_argument('--output_topic', default='/ikan/explore3d/defom/depth/compressed', help='colourised result')
    ros.add_argument('--depth_topic', default='/ikan/explore3d/defom/depth', help='metric depth, 32FC1')
    ros.add_argument('--disparity_topic', default='/ikan/explore3d/defom/disparity', help='disparity px, 32FC1')
    ros.add_argument('--debug_topic', default='/ikan/explore3d/defom/debug/compressed')
    ros.add_argument('--camera_info_topic', default='/ikan/explore3d/defom/camera_info')
    ros.add_argument('--points_topic', default='/ikan/explore3d/defom/points')
    ros.add_argument('--publish_debug', action='store_true', help='also publish left | result')
    ros.add_argument('--node_name', default=node_name)
    ros.add_argument('--poll_hz', type=float, default=100.0)
    ros.add_argument('--report_period', type=float, default=5.0, help='seconds between throughput logs')

    geom = parser.add_argument_group("stereo geometry (defaults: DWE explore3D)")
    geom.add_argument('--expect_width', type=int, default=1600, help='warn on another side-by-side width; 0 = off')
    geom.add_argument('--swap_lr', action='store_true', help='the right eye is in the left half')
    geom.add_argument('--fx', type=float, default=958.532, help='rectified fx in px; <= 0 disables metric depth')
    geom.add_argument('--fy', type=float, default=957.62)
    geom.add_argument('--cx', type=float, default=795.965)
    geom.add_argument('--cy', type=float, default=645.681)
    geom.add_argument('--calib_width', type=int, default=1600, help='resolution the intrinsics belong to')
    geom.add_argument('--calib_height', type=int, default=1200)
    geom.add_argument('--baseline_m', type=float, default=0.1, help='stereo baseline in metres')
    geom.add_argument('--fx_baseline', type=float, default=0, help='fx*B in m.px, overrides --baseline_m')
    geom.add_argument('--min_disparity', type=float, default=0.5, help='smaller disparities are invalid')
    geom.add_argument('--max_depth_m', type=float, default=0.0, help='drop depth beyond this; 0 = off')

    pts = parser.add_argument_group("point cloud")
    pts.add_argument('--publish_points', action='store_true', default=True)
    pts.add_argument('--no_points', dest='publish_points', action='store_false')
    pts.add_argument('--points_frame_id', default='explore3d', help='cloud frame; reached via live TF')
    pts.add_argument('--points_stride', type=int, default=2, help='every Nth pixel in both axes')
    pts.add_argument('--points_color', action='store_true', default=True)
    pts.add_argument('--no_points_color', dest='points_color', action='store_false')
    pts.add_argument('--points_tf_timeout', type=float, default=1.0)
    pts.add_argument('--points_tf_refresh', type=float, default=5.0)

    run = parser.add_argument_group("runtime")
    run.add_argument('--device', default='cuda')
    run.add_argument('--input_scale', type=float, default=1.0, help='downscale before inference, e.g. 0.5')
    run.add_argument('--warmup', action='store_true', default=True)
    run.add_argument('--no_warmup', dest='warmup', action='store_false')
    run.add_argument('--warmup_width', type=int, default=800)
    run.add_argument('--warmup_height', type=int, default=600)

    vis = parser.add_argument_group("visualisation")
    vis.add_argument('--colormap', default='jet', choices=sorted(COLORMAPS))
    vis.add_argument('--vis_min', type=float, default=None,
                     help='fixed colour range (with --vis_max); default: 2-98th percentile per frame')
    vis.add_argument('--vis_max', type=float, default=None)
    vis.add_argument('--vis_format', default='jpeg', choices=['jpeg', 'png'])
    vis.add_argument('--jpeg_quality', type=int, default=90)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_node_args(parser, 'defom_stereo_node')
    model = parser.add_argument_group("model (must match the checkpoint)")
    model.add_argument('--restore_ckpt', default='checkpoints/defomstereo_vitl_middlebury.pth')
    model.add_argument('--mixed_precision', action='store_true')
    model.add_argument('--valid_iters', type=int, default=32, help='disparity refinement iterations')
    model.add_argument('--scale_iters', type=int, default=8, help='scaling updates')
    model.add_argument('--dinov2_encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    model.add_argument('--idepth_scale', type=float, default=0.5)
    model.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    model.add_argument('--corr_implementation', default='reg', choices=["reg", "alt", "reg_cuda", "alt_cuda"])
    model.add_argument('--shared_backbone', action='store_true')
    model.add_argument('--corr_levels', type=int, default=2)
    model.add_argument('--corr_radius', type=int, default=4)
    model.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    model.add_argument('--scale_corr_radius', type=int, default=2)
    model.add_argument('--n_downsample', type=int, default=2, choices=[2, 3])
    model.add_argument('--context_norm', default='batch', choices=['group', 'batch', 'instance', 'none'])
    model.add_argument('--n_gru_layers', type=int, default=3)
    return parser.parse_args(argv)


def main():
    # rospy.myargv strips the __name:= / __log:= arguments roslaunch adds.
    args = parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node(args.node_name, anonymous=False)
    node = DefomStereoNode(args)
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
