#!/usr/bin/env python3
"""DEFOM-Stereo ROS1 node for the BeeX explore3d stereo stream.

Subscribes to a side-by-side stereo CompressedImage (1600x600 = two 800x600
frames concatenated horizontally, left | right), runs DEFOM-Stereo, and
publishes the result as a colorised CompressedImage, raw float rasters, and a
PointCloud2 reprojected from the metric depth.

The point cloud is built from the raw 32FC1 `~/depth` raster held in memory
during processing -- NOT from the `/compressed` colour topic. That topic is a
JPEG-compressed 8-bit JET colormap: lossy and not meaningfully invertible back
to depth. The uncompressed metric values are already available at the point
they would be needed, so there is no reason to round-trip through a picture
of them.

    rosrun-style:
        python3 demo_v2.py --restore_ckpt checkpoints/defomstereo_vitl_middlebury.pth

Measured on this machine (RTX 4070, 800x600 input, vitl):

    iters=32   0.83 s/frame   1.21 Hz      iters=8    0.58 s/frame   1.72 Hz
    iters=16   0.66 s/frame   1.51 Hz      iters=4    0.54 s/frame   1.87 Hz

The DINOv2-L encoder is ~0.5 s of that and runs once regardless of `iters`,
so lowering iterations buys little and costs accuracy -- prefer the default 32
and drop frames instead. Peak GPU is 4.4 GB (3.1 GB of it weights).

Frames are processed newest-first: the subscriber keeps only the most recent
message, so when inference is slower than the publisher the node stays live on
current data instead of falling behind a queue.
"""

import sys
sys.path.append('core')

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


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def build_model(args, device):
    model = DEFOMStereo(args)
    checkpoint = torch.load(args.restore_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    return model


def to_tensor(bgr, device):
    """BGR uint8 HxWx3 -> 1x3xHxW float RGB.

    The upstream demo loads with PIL, which is RGB, so the cv2-decoded BGR
    frame must be converted or the DINOv2 encoder sees swapped channels.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    return tensor[None].to(device)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def colorise(values, valid, lo, hi, colormap, invert):
    """Percentile- or fixed-range colour mapping, invalid pixels in black."""
    out = np.zeros(values.shape + (3,), dtype=np.uint8)
    if not valid.any():
        return out
    if lo is None or hi is None:
        finite = values[valid]
        low = float(np.percentile(finite, 2.0))
        high = float(np.percentile(finite, 98.0))
    else:
        low, high = float(lo), float(hi)
    if high - low < 1e-6:
        high = low + 1e-6
    norm = np.clip((values - low) / (high - low), 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    coloured = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    out[valid] = coloured[valid]
    return out


COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def build_pointcloud(header, depth, intrinsics, color_bgr, rotation, translation, stride, with_color):
    """Reproject metric depth into a PointCloud2, already expressed in `header.frame_id`.

    Depth was computed in the pinhole convention implied by the image itself
    (u right, v down, z forward -- the only convention that makes sense
    applied directly to pixel indices). `rotation`/`translation` are the live
    TF extrinsic from that image frame into the frame actually stamped on the
    output (looked up once per call site, cached by the caller), so the
    points this function returns are correct however that mount is oriented
    -- nothing about the camera's physical tilt is assumed here.
    """
    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    vs = np.arange(0, height, stride)
    us = np.arange(0, width, stride)
    grid_u, grid_v = np.meshgrid(us, vs)

    z = depth[grid_v, grid_u]
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        return None

    x_opt = (grid_u.astype(np.float32) - cx) * z / fx
    y_opt = (grid_v.astype(np.float32) - cy) * z / fy
    points_opt = np.stack([x_opt[valid], y_opt[valid], z[valid]], axis=1).astype(np.float32)

    points = points_opt @ rotation.T.astype(np.float32) + translation.astype(np.float32)

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    count = points.shape[0]

    if with_color and color_bgr is not None:
        bgr = color_bgr[grid_v, grid_u][valid]
        # Standard PCL packing: 0x00RRGGBB as a uint32, reinterpreted bit-for-bit as float32.
        packed = (bgr[:, 2].astype(np.uint32) << 16) | (bgr[:, 1].astype(np.uint32) << 8) | bgr[:, 0].astype(np.uint32)
        rgb = packed.view(np.float32)
        fields.append(PointField("rgb", 12, PointField.FLOAT32, 1))
        point_step = 16
        record = np.zeros(count, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)])
        record["rgb"] = rgb
    else:
        point_step = 12
        record = np.zeros(count, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])

    record["x"], record["y"], record["z"] = points[:, 0], points[:, 1], points[:, 2]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = count
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * count
    msg.is_dense = True
    msg.data = record.tobytes()
    return msg


# --------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------


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
        self.fx_baseline = None     # metres * pixels; the only quantity depth needs
        self.calib_note = ""

        rospy.loginfo("Loading %s (%s) ...", args.restore_ckpt, args.dinov2_encoder)
        self.model = build_model(args, self.device)
        rospy.loginfo("Model ready, %.2f GB on GPU", torch.cuda.memory_allocated() / 1e9
                      if self.device.type == "cuda" else 0.0)

        self.pub_vis = rospy.Publisher(args.output_topic, CompressedImage, queue_size=1)
        self.pub_disp = rospy.Publisher(args.disparity_topic, Image, queue_size=1)
        # Created unconditionally: whether depth is actually published depends
        # on calibration, which is resolved on the first frame.
        self.pub_depth = rospy.Publisher(args.depth_topic, Image, queue_size=1)
        self.pub_debug = (rospy.Publisher(args.debug_topic, CompressedImage, queue_size=1)
                          if args.publish_debug else None)
        self.pub_info = rospy.Publisher(args.camera_info_topic, CameraInfo, queue_size=1, latch=True)
        self.pub_points = (rospy.Publisher(args.points_topic, PointCloud2, queue_size=1)
                           if args.publish_points else None)
        self.tf_buffer = tf2_ros.Buffer() if args.publish_points else None
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) if args.publish_points else None
        self._extrinsic_cache = {}   # source_frame -> (rotation, translation, fetched_at)

        # queue_size=1 plus a big buffer: ROS drops stale frames at the
        # transport layer rather than letting them pile up in the socket.
        self.sub = rospy.Subscriber(args.input_topic, CompressedImage, self.on_image,
                                    queue_size=1, buff_size=2 ** 24, tcp_nodelay=True)
        rospy.loginfo("Subscribed to %s", args.input_topic)

        if args.warmup:
            self.warmup()

    # ---------------------------------------------------- calibration

    def resolve_calibration(self, width, height):
        """Scale the supplied intrinsics to the resolution actually processed.

        Intrinsics are resolution-dependent: fx, fy, cx and cy all scale
        linearly with image size. A calibration quoted for a 1600x1200 sensor
        is wrong by a factor of two when applied to an 800x600 raster, and the
        resulting depth is wrong by the same factor -- silently, since nothing
        about the output looks malformed.

        So the calibration resolution is declared explicitly via
        --calib_width/--calib_height and scaled here, rather than assumed to
        match the stream.
        """
        args = self.args
        if args.fx <= 0.0:
            self.fx_baseline = args.fx_baseline if args.fx_baseline > 0.0 else None
            if self.fx_baseline:
                self.calib_note = "fx*B given directly as %.4f m.px" % self.fx_baseline
                rospy.loginfo("Metric depth enabled: %s", self.calib_note)
            else:
                rospy.logwarn("No calibration given: publishing DISPARITY in pixels, not metric depth. "
                              "Pass --fx/--baseline_m (with --calib_width) or --fx_baseline. "
                              "%s carries the raw disparity raster.", args.disparity_topic)
            return

        calib_w = args.calib_width if args.calib_width > 0 else width
        calib_h = args.calib_height if args.calib_height > 0 else height
        sx = float(width) / float(calib_w)
        sy = float(height) / float(calib_h)

        fx, fy = args.fx * sx, (args.fy if args.fy > 0 else args.fx) * sy
        cx = (args.cx if args.cx >= 0 else 0.5 * calib_w) * sx
        cy = (args.cy if args.cy >= 0 else 0.5 * calib_h) * sy
        self.intrinsics = (fx, fy, cx, cy)

        if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
            rospy.loginfo("Scaled intrinsics %dx%d -> %dx%d (x%.4f): fx %.3f->%.3f  cx %.3f->%.3f  cy %.3f->%.3f",
                          calib_w, calib_h, width, height, sx, args.fx, fx, args.cx, cx, args.cy, cy)

        # A principal point outside the frame means the calibration and the
        # raster disagree about resolution. Depth would be silently wrong.
        if not (0 <= cx < width and 0 <= cy < height):
            rospy.logerr("Principal point (%.1f, %.1f) falls OUTSIDE the %dx%d image. The intrinsics "
                         "almost certainly belong to a different resolution -- set --calib_width/"
                         "--calib_height to the resolution they were calibrated at. Depth will be "
                         "wrong by that scale factor until this is fixed.", cx, cy, width, height)
        elif abs(cx - 0.5 * width) > 0.25 * width or abs(cy - 0.5 * height) > 0.25 * height:
            rospy.logwarn("Principal point (%.1f, %.1f) is far from the image centre (%.1f, %.1f) -- "
                          "worth double-checking --calib_width/--calib_height.",
                          cx, cy, 0.5 * width, 0.5 * height)

        if args.fx_baseline > 0.0:
            self.fx_baseline = args.fx_baseline
            self.calib_note = "fx*B overridden to %.4f m.px (implies baseline %.2f mm at fx=%.3f)" % (
                self.fx_baseline, 1000.0 * self.fx_baseline / fx, fx)
        elif args.baseline_m > 0.0:
            self.fx_baseline = fx * args.baseline_m
            self.calib_note = "fx=%.3f px, baseline=%.2f mm -> fx*B=%.4f m.px" % (
                fx, 1000.0 * args.baseline_m, self.fx_baseline)
        else:
            rospy.logwarn("Intrinsics given but no baseline: publishing DISPARITY only. "
                          "Add --baseline_m, or --fx_baseline to skip the decomposition.")
            return
        rospy.loginfo("Metric depth enabled: %s", self.calib_note)

    def get_extrinsic(self, source_frame):
        """Rotation/translation from `source_frame` into --points_frame_id, via live TF.

        Cached for `--points_tf_refresh` seconds: the mount is rigid, so the
        transform does not change frame to frame, and re-querying it on every
        cloud would be pure overhead. Still re-fetched periodically rather
        than once, so a recalibration or TF restart is picked up without
        restarting this node.
        """
        target = self.args.points_frame_id
        cached = self._extrinsic_cache.get(source_frame)
        now = rospy.get_time()
        if cached is not None and (now - cached[2]) < self.args.points_tf_refresh:
            return cached[0], cached[1]
        try:
            transform = self.tf_buffer.lookup_transform(
                target, source_frame, rospy.Time(0), rospy.Duration(self.args.points_tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(10.0, "No TF from '%s' to '%s' (%s) -- skipping point cloud this frame. "
                                  "Is the static transform being published?", source_frame, target, exc)
            return cached[:2] if cached is not None else (None, None)
        q = transform.transform.rotation
        t = transform.transform.translation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
        translation = np.array([t.x, t.y, t.z], dtype=np.float32)
        self._extrinsic_cache[source_frame] = (rotation, translation, now)
        return rotation, translation

    def publish_camera_info(self, header, width, height):
        if self.intrinsics is None or self.pub_info is None:
            return
        fx, fy, cx, cy = self.intrinsics
        info = CameraInfo()
        info.header = header
        info.width, info.height = width, height
        info.distortion_model = "plumb_bob"
        info.D = [0.0] * 5
        info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # Tx = -fx * B for the right camera of a rectified pair; this node
        # publishes the LEFT frame, so Tx is 0 here.
        info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
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
        """Decode and split the side-by-side frame into (left, right)."""
        buf = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn_throttle(5.0, "Failed to decode CompressedImage (format=%s)", msg.format)
            return None, None
        height, width = image.shape[:2]
        if width % 2 != 0:
            rospy.logwarn_throttle(5.0, "Odd width %d, cannot split evenly", width)
            return None, None
        if self.args.expect_width > 0 and width != self.args.expect_width:
            rospy.logwarn_throttle(10.0, "Frame is %dx%d, expected width %d -- splitting at the midpoint anyway",
                                   width, height, self.args.expect_width)
        half = width // 2
        left, right = image[:, :half], image[:, half:]
        if self.args.swap_lr:
            left, right = right, left
        return left, right

    # ------------------------------------------------------------ inference

    def warmup(self):
        """Run one synthetic pass so the first real frame is not the slow one."""
        height = self.args.warmup_height
        width = self.args.warmup_width
        dummy = np.zeros((height, width, 3), np.uint8)
        rospy.loginfo("Warming up at %dx%d ...", width, height)
        started = time.time()
        self.infer(dummy, dummy)
        rospy.loginfo("Warmup done in %.2f s", time.time() - started)

    def infer(self, left, right):
        full_h, full_w = left.shape[:2]
        scale = self.args.input_scale
        if scale != 1.0:
            left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        image1 = to_tensor(left, self.device)
        image2 = to_tensor(right, self.device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            disp = self.model(image1, image2,
                              iters=self.args.valid_iters,
                              scale_iters=self.args.scale_iters,
                              test_mode=True)
        disp = padder.unpad(disp).cpu().squeeze().numpy().astype(np.float32)

        if scale != 1.0:
            # Disparity is a length in pixels, so going back to full resolution
            # has to rescale the *values* as well as the raster: a 0.5-scale
            # run yields half the disparity a full-scale run would.
            disp = cv2.resize(disp, (full_w, full_h), interpolation=cv2.INTER_LINEAR) / scale
        return disp

    # --------------------------------------------------------------- output

    def publish(self, header, left, disp):
        valid = np.isfinite(disp) & (disp > self.args.min_disparity)

        disp_msg = Image()
        disp_msg.header = header
        disp_msg.height, disp_msg.width = disp.shape
        disp_msg.encoding = "32FC1"
        disp_msg.is_bigendian = 0
        disp_msg.step = 4 * disp.shape[1]
        disp_msg.data = disp.tobytes()
        self.pub_disp.publish(disp_msg)
        self.publish_camera_info(header, disp.shape[1], disp.shape[0])

        if self.fx_baseline is not None:
            depth = np.zeros_like(disp)
            np.divide(self.fx_baseline, disp, out=depth, where=valid)
            depth[~valid] = 0.0
            if self.args.max_depth_m > 0:
                depth[depth > self.args.max_depth_m] = 0.0
            depth_msg = Image()
            depth_msg.header = header
            depth_msg.height, depth_msg.width = depth.shape
            depth_msg.encoding = "32FC1"
            depth_msg.is_bigendian = 0
            depth_msg.step = 4 * depth.shape[1]
            depth_msg.data = depth.tobytes()
            self.pub_depth.publish(depth_msg)
            # Near = warm, so invert: small depth should read as "close".
            field, lo, hi, invert = depth, self.args.vis_min, self.args.vis_max, True
            field_valid = valid & (depth > 0)

            if self.pub_points is not None and self.intrinsics is not None:
                rotation, translation = self.get_extrinsic(header.frame_id)
                if rotation is not None:
                    cloud_header = Header()
                    cloud_header.stamp = header.stamp
                    cloud_header.frame_id = self.args.points_frame_id
                    cloud = build_pointcloud(cloud_header, depth, self.intrinsics, left, rotation, translation,
                                             self.args.points_stride, self.args.points_color)
                    if cloud is not None:
                        self.pub_points.publish(cloud)
        elif self.pub_points is not None:
            rospy.logwarn_throttle(30.0, "--publish_points is on but no metric depth is available "
                                   "(need --fx/--baseline_m or --fx_baseline); no cloud published.")
        else:
            field, lo, hi, invert = disp, self.args.vis_min, self.args.vis_max, False
            field_valid = valid

        vis = colorise(field, field_valid, lo, hi, self.colormap, invert)
        self.pub_vis.publish(self.to_compressed(header, vis))

        if self.pub_debug is not None:
            if vis.shape[:2] != left.shape[:2]:
                vis_side = cv2.resize(vis, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                vis_side = vis
            self.pub_debug.publish(self.to_compressed(header, np.hstack([left, vis_side])))

    def to_compressed(self, header, bgr):
        msg = CompressedImage()
        msg.header = header
        if self.args.vis_format == "png":
            msg.format = "png"
            ok, buf = cv2.imencode(".png", bgr)
        else:
            msg.format = "jpeg"
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality])
        if not ok:
            rospy.logwarn_throttle(5.0, "Failed to encode output image")
            return msg
        msg.data = buf.tobytes()
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
            if not self.calib_done:
                # Deferred until the first frame, so intrinsics are scaled
                # against the real stream size rather than an assumed one.
                self.resolve_calibration(left.shape[1], left.shape[0])
                self.calib_done = True
            try:
                disp = self.infer(left, right)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    rospy.logerr_throttle(10.0, "CUDA OOM -- consider --input_scale 0.5: %s", exc)
                    continue
                raise

            self.publish(msg.header, left, disp)
            self.processed += 1

            now = time.time()
            if now - self.last_report >= self.args.report_period:
                rospy.loginfo("%.2f Hz | last %.0f ms | processed %d | dropped %d | disp %.1f-%.1f px",
                              self.processed / max(1e-6, now - self.last_report),
                              1000.0 * (now - started), self.processed, self.dropped,
                              float(np.nanmin(disp)), float(np.nanmax(disp)))
                self.processed = 0
                self.last_report = now


# --------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- ROS ---
    ros = parser.add_argument_group("ROS")
    ros.add_argument('--input_topic', default='/ikan/explore3d/stereo/compressed',
                     help='side-by-side stereo CompressedImage')
    ros.add_argument('--output_topic', default='/ikan/explore3d/defom/depth/compressed',
                     help='colourised depth (or disparity) as CompressedImage')
    ros.add_argument('--depth_topic', default='/ikan/explore3d/defom/depth',
                     help='metric depth, 32FC1 -- only published when calibration is given')
    ros.add_argument('--disparity_topic', default='/ikan/explore3d/defom/disparity',
                     help='raw disparity in pixels, 32FC1')
    ros.add_argument('--debug_topic', default='/ikan/explore3d/defom/debug/compressed',
                     help='left image next to the colourised result')
    ros.add_argument('--camera_info_topic', default='/ikan/explore3d/defom/camera_info',
                     help='CameraInfo for the published rasters, at processing resolution')
    ros.add_argument('--points_topic', default='/ikan/explore3d/defom/points',
                     help='PointCloud2 reprojected from metric depth')
    ros.add_argument('--publish_debug', action='store_true')
    ros.add_argument('--node_name', default='defom_stereo_node')
    ros.add_argument('--poll_hz', type=float, default=100.0,
                     help='how often the main loop checks for a new frame')
    ros.add_argument('--report_period', type=float, default=5.0, help='seconds between throughput logs')

    # --- stereo geometry ---
    geom = parser.add_argument_group("stereo geometry")
    geom.add_argument('--expect_width', type=int, default=1600,
                      help='expected side-by-side width; 0 disables the check')
    geom.add_argument('--swap_lr', action='store_true',
                      help='use if the right image is in the left half')
    geom.add_argument('--fx', type=float, default=958.532, help='rectified focal length x, in pixels')
    geom.add_argument('--fy', type=float, default=957.62, help='rectified focal length y; defaults to fx')
    geom.add_argument('--cx', type=float, default=795.965, help='principal point x; defaults to image centre')
    geom.add_argument('--cy', type=float, default=645.681, help='principal point y; defaults to image centre')
    geom.add_argument('--calib_width', type=int, default=1600,
                      help='width the intrinsics were calibrated at. Intrinsics scale linearly with '
                           'resolution, so this must be set whenever it differs from the stream. '
                           'Default matches the fx/fy/cx/cy defaults above (a 1600x1200 calibration '
                           'applied to the 800x600 stream this node actually processes).')
    geom.add_argument('--calib_height', type=int, default=1200, help='height the intrinsics were calibrated at')
    geom.add_argument('--baseline_m', type=float, default=0.1,
                    help='stereo baseline in metres. Measured from the explore3D Cobalt '
                        'mechanical drawing (100mm between lens centres). Used unless '
                        '--fx_baseline overrides it.')
    geom.add_argument('--fx_baseline', type=float, default=0,
                    help='fx * baseline directly, in metre-pixels, overriding --baseline_m. '
                        'Leave 0 to use --baseline_m (recommended: it is built from the '
                        'verified fx/baseline rather than a fitted constant).')
    geom.add_argument('--min_disparity', type=float, default=0.5,
                      help='disparity at or below this is treated as invalid')
    geom.add_argument('--max_depth_m', type=float, default=0.0,
                      help='clamp depth beyond this to invalid; 0 disables')

    pts = parser.add_argument_group("point cloud")
    pts.add_argument('--publish_points', action='store_true', default=True,
                     help='publish a PointCloud2 reprojected from metric depth')
    pts.add_argument('--no_points', dest='publish_points', action='store_false')
    pts.add_argument('--points_frame_id', default='ikan/camera_link',
                     help='frame_id stamped on the cloud. Points are rotated into this frame using the '
                          'LIVE TF from the image frame (e.g. explore3d) -- looked up at runtime, never '
                          'assumed, since the mount is not a textbook optical<->body swap (measured: a '
                          '~100 deg roll about X, not the usual -90/0/-90).')
    pts.add_argument('--points_stride', type=int, default=2,
                     help='take every Nth pixel in both axes; 1 = full resolution (480k points/frame), '
                          '2 = quarter density (120k), etc.')
    pts.add_argument('--points_color', action='store_true', default=True,
                     help='attach RGB from the left image to each point')
    pts.add_argument('--no_points_color', dest='points_color', action='store_false')
    pts.add_argument('--points_tf_timeout', type=float, default=1.0,
                     help='seconds to wait for the extrinsic TF before giving up for that frame')
    pts.add_argument('--points_tf_refresh', type=float, default=5.0,
                     help='seconds between re-querying the extrinsic TF; it is cached in between '
                          'since the mount is rigid')

    # --- runtime ---
    run = parser.add_argument_group("runtime")
    run.add_argument('--device', default='cuda')
    run.add_argument('--input_scale', type=float, default=1.0,
                     help='downscale factor before inference, e.g. 0.5 for speed/VRAM')
    run.add_argument('--warmup', action='store_true', default=True)
    run.add_argument('--no_warmup', dest='warmup', action='store_false')
    run.add_argument('--warmup_width', type=int, default=800)
    run.add_argument('--warmup_height', type=int, default=600)

    # --- visualisation ---
    vis = parser.add_argument_group("visualisation")
    vis.add_argument('--colormap', default='jet', choices=sorted(COLORMAPS))
    vis.add_argument('--vis_min', type=float, default=None,
                     help='fixed colour-scale minimum; default (unset) is a 2nd-98th percentile '
                          'stretch computed fresh per frame. Setting only one of --vis_min/--vis_max '
                          'is ignored -- both are required together, or neither.')
    vis.add_argument('--vis_max', type=float, default=None, help='fixed colour-scale maximum')
    vis.add_argument('--vis_format', default='jpeg', choices=['jpeg', 'png'])
    vis.add_argument('--jpeg_quality', type=int, default=90)

    # --- model (must match the checkpoint; defaults mirror demo.py) ---
    parser.add_argument('--restore_ckpt', default='checkpoints/defomstereo_vitl_sceneflow.pth',
                        help='path to the .pth checkpoint')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='disparity refinement iterations')
    parser.add_argument('--scale_iters', type=int, default=8,
                        help='scaling updates per forward pass')
    parser.add_argument('--dinov2_encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    parser.add_argument('--corr_implementation', default='reg',
                        choices=["reg", "alt", "reg_cuda", "alt_cuda"])
    parser.add_argument('--shared_backbone', action='store_true')
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--scale_list', type=float, nargs='+',
                        default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument('--scale_corr_radius', type=int, default=2)
    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3])
    parser.add_argument('--context_norm', default='batch',
                        choices=['group', 'batch', 'instance', 'none'])
    parser.add_argument('--n_gru_layers', type=int, default=3)

    return parser.parse_args(argv)


def main():
    # rospy.myargv strips the __name:= / __log:= arguments roslaunch injects,
    # which argparse would otherwise reject.
    args = parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node(args.node_name, anonymous=False)
    node = DefomStereoNode(args)
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
