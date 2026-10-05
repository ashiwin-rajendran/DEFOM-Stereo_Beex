#!/usr/bin/env python3
"""DEFOM-Stereo multi-model comparison node.

Runs several DEFOM checkpoints on the SAME stereo frame and publishes each
one under its own namespace, so vitl and vits can be compared live on real
data instead of on one saved frame.

    python3 demo_v3.py

Default line-up (override with --models):

    middlebury   vitl   defomstereo_vitl_middlebury.pth    1530 MB
    sceneflow_l  vitl   defomstereo_vitl_sceneflow.pth     1530 MB
    sceneflow_s  vits   defomstereo_vits_sceneflow.pth      173 MB

Per model, under /ikan/explore3d/defom/<name>/ :
    depth/compressed   colourised depth          (CompressedImage)
    depth              metric depth, 32FC1       (Image)
    disparity          raw disparity px, 32FC1   (Image)
    points             reprojected cloud         (PointCloud2)
    camera_info        scaled intrinsics         (CameraInfo)

Plus one shared comparison topic:
    /ikan/explore3d/defom/compare/compressed
        left frame followed by every model's depth, ALL ON ONE COLOUR SCALE.

That shared scale matters. Each model colourised against its own 2-98th
percentile would look near-identical even where the models disagree badly,
because the stretch hides the offset. The reference model sets the scale and
everyone else is drawn against it, so a real disagreement shows up as a real
colour difference.

Agreement against the reference model (the first one listed) is logged
periodically: mean/median/p90 absolute disparity difference and the fraction
of pixels within 1, 2 and 3 px.

All inference maths -- the model builder, the tensor conversion, the colour
mapping and the point cloud construction -- is imported from demo_v2 rather
than copied, so a fix there applies here too.

GPU budget: two vitl plus one vits is ~3.2 GB of weights, plus ~2.8 GB of
transient activation for whichever model is mid-forward (they run one at a
time, so activation is not multiplied). Around 7.5 GB total. Stop any running
demo_v2 first or this will not fit on a 12 GB card.
"""

import sys
sys.path.append('core')

import argparse
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import rospy
import torch
import tf2_ros
from tf.transformations import quaternion_matrix
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2

from core.utils.utils import InputPadder

# Shared with demo_v2 on purpose -- single source of truth for the maths.
from demo_v2 import build_model, to_tensor, colorise, build_pointcloud, COLORMAPS


DEFAULT_MODELS = [
    "middlebury:vitl:checkpoints/defomstereo_vitl_middlebury.pth",
    "sceneflow_l:vitl:checkpoints/defomstereo_vitl_sceneflow.pth",
    "sceneflow_s:vits:checkpoints/defomstereo_vits_sceneflow.pth",
]


class ModelSpec(object):
    """One checkpoint plus everything published for it."""

    def __init__(self, name, encoder, checkpoint, args, device):
        self.name = name
        self.encoder = encoder
        self.checkpoint = checkpoint
        self.device = device
        self.last_seconds = 0.0

        # DEFOM reads its architecture from the args namespace, and the
        # encoder differs per checkpoint -- so each model gets its own shallow
        # copy with dinov2_encoder and restore_ckpt swapped in.
        model_args = argparse.Namespace(**vars(args))
        model_args.dinov2_encoder = encoder
        model_args.restore_ckpt = checkpoint
        self.args = model_args

        rospy.loginfo("[%s] loading %s (%s) ...", name, checkpoint, encoder)
        before = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        self.model = build_model(model_args, device)
        after = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        self.weight_gb = (after - before) / 1e9
        rospy.loginfo("[%s] ready, %.2f GB of weights", name, self.weight_gb)

        ns = "%s/%s" % (args.output_ns.rstrip("/"), name)
        self.pub_vis = rospy.Publisher(ns + "/depth/compressed", CompressedImage, queue_size=1)
        self.pub_depth = rospy.Publisher(ns + "/depth", Image, queue_size=1)
        self.pub_disp = rospy.Publisher(ns + "/disparity", Image, queue_size=1)
        self.pub_info = rospy.Publisher(ns + "/camera_info", CameraInfo, queue_size=1, latch=True)
        self.pub_points = (rospy.Publisher(ns + "/points", PointCloud2, queue_size=1)
                           if args.publish_points else None)

    def infer(self, left, right):
        scale = self.args.input_scale
        full_h, full_w = left.shape[:2]
        if scale != 1.0:
            left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        image1 = to_tensor(left, self.device)
        image2 = to_tensor(right, self.device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        started = time.time()
        with torch.no_grad():
            disp = self.model(image1, image2,
                              iters=self.args.valid_iters,
                              scale_iters=self.args.scale_iters,
                              test_mode=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_seconds = time.time() - started

        disp = padder.unpad(disp).cpu().squeeze().numpy().astype(np.float32)
        if scale != 1.0:
            disp = cv2.resize(disp, (full_w, full_h), interpolation=cv2.INTER_LINEAR) / scale
        return disp


class MultiModelNode(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if args.device != "auto"
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lock = threading.Lock()
        self.latest = None
        self.dropped = 0
        self.processed = 0
        self.last_report = time.time()
        self.calib_done = False
        self.intrinsics = None
        self.fx_baseline = None
        self.colormap = COLORMAPS[args.colormap]
        self._extrinsic_cache = {}
        self.agreement = {}

        self.models = OrderedDict()
        for spec in args.models:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError("--models entries must be name:encoder:checkpoint, got %r" % spec)
            name, encoder, checkpoint = parts
            try:
                self.models[name] = ModelSpec(name, encoder, checkpoint, args, self.device)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise SystemExit(
                        "CUDA OOM loading '%s'. Two vitl models plus activation need ~7.5 GB; "
                        "stop any running demo_v2, or pass a shorter --models list." % name)
                raise
        if not self.models:
            raise SystemExit("No models configured.")
        self.reference = args.reference or list(self.models)[0]
        if self.reference not in self.models:
            raise SystemExit("--reference %r is not one of %s" % (self.reference, list(self.models)))
        rospy.loginfo("Reference model for agreement stats: %s", self.reference)

        self.pub_compare = (rospy.Publisher(args.compare_topic, CompressedImage, queue_size=1)
                            if args.publish_compare else None)
        self.tf_buffer = tf2_ros.Buffer() if args.publish_points else None
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) if args.publish_points else None

        self.sub = rospy.Subscriber(args.input_topic, CompressedImage, self.on_image,
                                    queue_size=1, buff_size=2 ** 24, tcp_nodelay=True)
        rospy.loginfo("Subscribed to %s", args.input_topic)
        if self.device.type == "cuda":
            rospy.loginfo("Total weights resident: %.2f GB",
                          sum(m.weight_gb for m in self.models.values()))

    # ------------------------------------------------------------- input

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
        width = image.shape[1]
        if width % 2:
            rospy.logwarn_throttle(5.0, "Odd width %d, cannot split", width)
            return None, None
        half = width // 2
        left, right = image[:, :half], image[:, half:]
        if self.args.swap_lr:
            left, right = right, left
        return left, right

    # ------------------------------------------------------- calibration

    def resolve_calibration(self, width, height):
        args = self.args
        if args.fx <= 0.0:
            self.fx_baseline = args.fx_baseline if args.fx_baseline > 0 else None
            if self.fx_baseline is None:
                rospy.logwarn("No calibration: publishing disparity only, no metric depth or clouds.")
            return
        calib_w = args.calib_width if args.calib_width > 0 else width
        calib_h = args.calib_height if args.calib_height > 0 else height
        sx, sy = float(width) / calib_w, float(height) / calib_h
        fx, fy = args.fx * sx, (args.fy if args.fy > 0 else args.fx) * sy
        cx = (args.cx if args.cx >= 0 else 0.5 * calib_w) * sx
        cy = (args.cy if args.cy >= 0 else 0.5 * calib_h) * sy
        self.intrinsics = (fx, fy, cx, cy)
        if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
            rospy.loginfo("Scaled intrinsics %dx%d -> %dx%d (x%.4f): fx %.3f->%.3f cx %.3f->%.3f cy %.3f->%.3f",
                          calib_w, calib_h, width, height, sx, args.fx, fx, args.cx, cx, args.cy, cy)
        if not (0 <= cx < width and 0 <= cy < height):
            rospy.logerr("Principal point (%.1f, %.1f) is outside the %dx%d image -- intrinsics almost "
                         "certainly belong to another resolution; set --calib_width/--calib_height.",
                         cx, cy, width, height)
        if args.fx_baseline > 0:
            self.fx_baseline = args.fx_baseline
        elif args.baseline_m > 0:
            self.fx_baseline = fx * args.baseline_m
        if self.fx_baseline:
            rospy.loginfo("Metric depth enabled: fx*B = %.4f m.px (fx=%.3f)", self.fx_baseline, fx)

    def get_extrinsic(self, source_frame):
        target = self.args.points_frame_id
        cached = self._extrinsic_cache.get(source_frame)
        now = rospy.get_time()
        if cached is not None and (now - cached[2]) < self.args.points_tf_refresh:
            return cached[0], cached[1]
        try:
            tr = self.tf_buffer.lookup_transform(target, source_frame, rospy.Time(0),
                                                 rospy.Duration(self.args.points_tf_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(10.0, "No TF '%s' -> '%s' (%s); skipping clouds this frame.",
                                   source_frame, target, exc)
            return (cached[0], cached[1]) if cached else (None, None)
        q, t = tr.transform.rotation, tr.transform.translation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
        translation = np.array([t.x, t.y, t.z], dtype=np.float32)
        self._extrinsic_cache[source_frame] = (rotation, translation, now)
        return rotation, translation

    # ------------------------------------------------------------ output

    @staticmethod
    def _raster(header, array):
        msg = Image()
        msg.header = header
        msg.height, msg.width = array.shape
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = 4 * array.shape[1]
        msg.data = array.tobytes()
        return msg

    def _compressed(self, header, bgr):
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
        return msg

    def publish_one(self, spec, header, left, disp, scale_lo, scale_hi, rotation, translation):
        valid = np.isfinite(disp) & (disp > self.args.min_disparity)
        spec.pub_disp.publish(self._raster(header, disp))

        if spec.pub_info is not None and self.intrinsics is not None:
            fx, fy, cx, cy = self.intrinsics
            info = CameraInfo()
            info.header = header
            info.width, info.height = disp.shape[1], disp.shape[0]
            info.distortion_model = "plumb_bob"
            info.D = [0.0] * 5
            info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            spec.pub_info.publish(info)

        if self.fx_baseline is None:
            vis = colorise(disp, valid, scale_lo, scale_hi, self.colormap, False)
            spec.pub_vis.publish(self._compressed(header, vis))
            return vis

        depth = np.zeros_like(disp)
        np.divide(self.fx_baseline, disp, out=depth, where=valid)
        depth[~valid] = 0.0
        if self.args.max_depth_m > 0:
            depth[depth > self.args.max_depth_m] = 0.0
        spec.pub_depth.publish(self._raster(header, depth))

        vis = colorise(depth, valid & (depth > 0), scale_lo, scale_hi, self.colormap, True)
        spec.pub_vis.publish(self._compressed(header, vis))

        if spec.pub_points is not None and self.intrinsics is not None and rotation is not None:
            cloud_header = Header()
            cloud_header.stamp = header.stamp
            cloud_header.frame_id = self.args.points_frame_id
            cloud = build_pointcloud(cloud_header, depth, self.intrinsics, left,
                                     rotation, translation, self.args.points_stride,
                                     self.args.points_color)
            if cloud is not None:
                spec.pub_points.publish(cloud)
        return vis

    # -------------------------------------------------------------- loop

    def spin(self):
        rate = rospy.Rate(self.args.poll_hz)
        while not rospy.is_shutdown():
            msg = self.take_latest()
            if msg is None:
                rate.sleep()
                continue

            left, right = self.split_stereo(msg)
            if left is None:
                continue
            if not self.calib_done:
                self.resolve_calibration(left.shape[1], left.shape[0])
                self.calib_done = True

            try:
                disparities = OrderedDict(
                    (name, spec.infer(left, right)) for name, spec in self.models.items())
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    rospy.logerr_throttle(10.0, "CUDA OOM during inference: %s", exc)
                    continue
                raise

            # One colour scale for every model, derived from the reference, so
            # the comparison image shows real differences instead of each
            # model's own percentile stretch.
            ref_disp = disparities[self.reference]
            ref_valid = np.isfinite(ref_disp) & (ref_disp > self.args.min_disparity)
            if self.args.vis_min is not None and self.args.vis_max is not None:
                scale_lo, scale_hi = self.args.vis_min, self.args.vis_max
            elif self.fx_baseline is not None and ref_valid.any():
                ref_depth = self.fx_baseline / np.clip(ref_disp, 1e-6, None)
                good = ref_depth[ref_valid & np.isfinite(ref_depth)]
                scale_lo, scale_hi = float(np.percentile(good, 2)), float(np.percentile(good, 98))
            elif ref_valid.any():
                scale_lo, scale_hi = (float(np.percentile(ref_disp[ref_valid], 2)),
                                      float(np.percentile(ref_disp[ref_valid], 98)))
            else:
                scale_lo = scale_hi = None

            rotation = translation = None
            if self.args.publish_points and self.tf_buffer is not None:
                rotation, translation = self.get_extrinsic(msg.header.frame_id)

            panels = []
            for name, spec in self.models.items():
                vis = self.publish_one(spec, msg.header, left, disparities[name],
                                       scale_lo, scale_hi, rotation, translation)
                panels.append((name, vis))

            self.update_agreement(disparities)
            if self.pub_compare is not None:
                self.publish_compare(msg.header, left, panels)

            self.processed += 1
            now = time.time()
            if now - self.last_report >= self.args.report_period:
                timings = "  ".join("%s %.0fms" % (n, 1000 * s.last_seconds)
                                    for n, s in self.models.items())
                rospy.loginfo("%.2f Hz | %s | processed %d | dropped %d",
                              self.processed / max(1e-6, now - self.last_report),
                              timings, self.processed, self.dropped)
                self.log_agreement()
                self.processed = 0
                self.last_report = now

    # --------------------------------------------------------- agreement

    def update_agreement(self, disparities):
        ref = disparities[self.reference]
        ref_valid = np.isfinite(ref) & (ref > self.args.min_disparity)
        for name, disp in disparities.items():
            if name == self.reference:
                continue
            both = ref_valid & np.isfinite(disp) & (disp > self.args.min_disparity)
            if not both.any():
                continue
            diff = np.abs(disp[both] - ref[both])
            acc = self.agreement.setdefault(name, {"n": 0, "mean": 0.0, "median": 0.0,
                                                   "p90": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0})
            acc["n"] += 1
            for key, value in (("mean", diff.mean()), ("median", np.median(diff)),
                               ("p90", np.percentile(diff, 90)),
                               ("w1", (diff < 1).mean()), ("w2", (diff < 2).mean()),
                               ("w3", (diff < 3).mean())):
                acc[key] += float(value)

    def log_agreement(self):
        for name, acc in self.agreement.items():
            n = max(1, acc["n"])
            rospy.loginfo("  vs %s | %s: mean %.2f px  median %.2f  p90 %.2f  "
                          "within 1/2/3px %.0f%%/%.0f%%/%.0f%%  (%d frames)",
                          self.reference, name, acc["mean"] / n, acc["median"] / n, acc["p90"] / n,
                          100 * acc["w1"] / n, 100 * acc["w2"] / n, 100 * acc["w3"] / n, acc["n"])

    def publish_compare(self, header, left, panels):
        tiles = [left]
        for name, vis in panels:
            tile = vis.copy()
            cv2.putText(tile, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(tile, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            ms = self.models[name].last_seconds * 1000.0
            label = "%.0f ms" % ms
            cv2.putText(tile, label, (8, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(tile, label, (8, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        if self.args.compare_layout == "grid" and len(tiles) == 4:
            row1 = np.hstack(tiles[:2])
            row2 = np.hstack(tiles[2:])
            canvas = np.vstack([row1, row2])
        else:
            canvas = np.hstack(tiles)
        self.pub_compare.publish(self._compressed(header, canvas))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help='models to run, each as name:encoder:checkpoint. The encoder MUST match '
                             'the checkpoint (vitl/vits) or loading fails.')
    parser.add_argument('--reference', default=None,
                        help='model name to measure agreement against; default is the first listed')

    ros = parser.add_argument_group("ROS")
    ros.add_argument('--input_topic', default='/ikan/explore3d/stereo/compressed')
    ros.add_argument('--output_ns', default='/ikan/explore3d/defom',
                     help='namespace prefix; each model publishes under <ns>/<name>/...')
    ros.add_argument('--compare_topic', default='/ikan/explore3d/defom/compare/compressed')
    ros.add_argument('--publish_compare', action='store_true', default=True)
    ros.add_argument('--no_compare', dest='publish_compare', action='store_false')
    ros.add_argument('--compare_layout', default='grid', choices=['grid', 'row'],
                     help='grid is 2x2 for the 3-model default; row is a single strip')
    ros.add_argument('--node_name', default='defom_stereo_compare')
    ros.add_argument('--poll_hz', type=float, default=100.0)
    ros.add_argument('--report_period', type=float, default=10.0)

    geom = parser.add_argument_group("stereo geometry")
    geom.add_argument('--swap_lr', action='store_true')
    geom.add_argument('--fx', type=float, default=958.532)
    geom.add_argument('--fy', type=float, default=957.62)
    geom.add_argument('--cx', type=float, default=795.965)
    geom.add_argument('--cy', type=float, default=645.681)
    geom.add_argument('--calib_width', type=int, default=1600)
    geom.add_argument('--calib_height', type=int, default=1200)
    geom.add_argument('--baseline_m', type=float, default=0.1)
    geom.add_argument('--fx_baseline', type=float, default=0)
    geom.add_argument('--min_disparity', type=float, default=0.5)
    geom.add_argument('--max_depth_m', type=float, default=0.0)

    pts = parser.add_argument_group("point cloud")
    pts.add_argument('--publish_points', action='store_true', default=False,
                     help='off by default here: three clouds per frame is a lot of bandwidth '
                          'for a comparison run')
    pts.add_argument('--points_frame_id', default='ikan/camera_link')
    pts.add_argument('--points_stride', type=int, default=2)
    pts.add_argument('--points_color', action='store_true', default=True)
    pts.add_argument('--no_points_color', dest='points_color', action='store_false')
    pts.add_argument('--points_tf_timeout', type=float, default=1.0)
    pts.add_argument('--points_tf_refresh', type=float, default=5.0)

    run = parser.add_argument_group("runtime")
    run.add_argument('--device', default='cuda')
    run.add_argument('--input_scale', type=float, default=1.0)

    vis = parser.add_argument_group("visualisation")
    vis.add_argument('--colormap', default='jet', choices=sorted(COLORMAPS))
    vis.add_argument('--vis_min', type=float, default=None,
                     help='fixed colour-scale minimum shared by all models; default derives it from '
                          'the reference model per frame')
    vis.add_argument('--vis_max', type=float, default=None)
    vis.add_argument('--vis_format', default='jpeg', choices=['jpeg', 'png'])
    vis.add_argument('--jpeg_quality', type=int, default=90)

    arch = parser.add_argument_group("architecture (shared by all models)")
    arch.add_argument('--mixed_precision', action='store_true')
    arch.add_argument('--valid_iters', type=int, default=32)
    arch.add_argument('--scale_iters', type=int, default=8)
    arch.add_argument('--idepth_scale', type=float, default=0.5)
    arch.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    arch.add_argument('--corr_implementation', default='reg',
                      choices=["reg", "alt", "reg_cuda", "alt_cuda"])
    arch.add_argument('--shared_backbone', action='store_true')
    arch.add_argument('--corr_levels', type=int, default=2)
    arch.add_argument('--corr_radius', type=int, default=4)
    arch.add_argument('--scale_list', type=float, nargs='+',
                      default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    arch.add_argument('--scale_corr_radius', type=int, default=2)
    arch.add_argument('--n_downsample', type=int, default=2, choices=[2, 3])
    arch.add_argument('--context_norm', default='batch',
                      choices=['group', 'batch', 'instance', 'none'])
    arch.add_argument('--n_gru_layers', type=int, default=3)
    # build_model() reads these off the namespace; set per model in ModelSpec.
    arch.add_argument('--dinov2_encoder', default='vitl')
    arch.add_argument('--restore_ckpt', default=DEFAULT_MODELS[0].split(':')[2])

    return parser.parse_args(argv)


def main():
    args = parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node(args.node_name, anonymous=False)
    node = MultiModelNode(args)
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
