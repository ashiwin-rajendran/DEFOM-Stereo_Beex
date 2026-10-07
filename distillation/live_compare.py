#!/usr/bin/env python3
"""Live student-vs-teacher comparison on the stereo ROS stream.

Subscribes to the same side-by-side stereo topic demo_v2/demo_v3 use, runs
BOTH the vitl teacher and the distilled student on every frame, and publishes
a side-by-side view plus live agreement stats -- so you can watch training
progress check out against the real camera, not just the held-out dataset
frames.

    python3 distillation/live_compare.py

Defaults to the latest checkpoint under distillation/checkpoints/<run>/ --
no need to edit the epoch number after every resume. Override with
--student_ckpt to pin a specific one.

ONE model instance, not two: the student is swapped into the SAME DEFOMStereo
object's defomencoder.depth_anything for its forward pass, then swapped back
-- exactly the trick validate_disparity() uses during training. Running two
full 1.5 GB DEFOM instances would work on most GPUs but there is no reason to
pay for it.

Topics:
    /ikan/explore3d/defom/teacher/depth/compressed   colourised, shared scale
    /ikan/explore3d/defom/student/depth/compressed   colourised, SAME scale
    /ikan/explore3d/defom/teacher/depth              metric, 32FC1
    /ikan/explore3d/defom/student/depth              metric, 32FC1
    /ikan/explore3d/defom/compare/compressed         left | teacher | student

The compare image uses ONE colour scale, taken from the teacher's own 2-98th
percentile per frame. If each panel used its own stretch the two would look
similar even when they disagree badly -- see demo_v3.py for the same reasoning
applied to vitl vs vits.

Live agreement (periodic log, same definition training used for
[val vs teacher]): mean/median/p90 px difference, and the fraction of pixels
within 1/2/3 px of the teacher.
"""

import sys
from pathlib import Path

# Needs to work regardless of cwd: resolve the repo root relative to this
# file's own location, same approach distill_defom.py uses, rather than
# relying on relative strings that only happen to work when launched from
# the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import glob
import re
import threading
import time
import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from core.utils.utils import InputPadder
from demo_v2 import build_model, to_tensor, colorise, COLORMAPS
from defom_student import build_student


DEFAULT_CKPT_DIR = "distillation/checkpoints/underwater_vitl_to_student"


def resolve_student_checkpoint(path):
    """A file is used as-is. A directory resolves to its highest-epoch
    student_epoch####.pth -- so this keeps working across resumes without
    editing a flag every time."""
    p = Path(path)
    if p.is_file():
        return p
    candidates = sorted(Path(path).glob("student_epoch*.pth"),
                        key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)))
    if not candidates:
        raise FileNotFoundError("No student_epoch*.pth under %s" % path)
    return candidates[-1]


class LiveCompareNode(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if args.device != "auto"
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lock = threading.Lock()
        self.latest = None
        self.dropped = 0
        self.processed = 0
        self.last_report = time.time()
        self.intrinsics = None
        self.fx_baseline = None
        self.calib_done = False
        self.colormap = COLORMAPS[args.colormap]
        self.agreement = {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0,
                          "w1": 0.0, "w2": 0.0, "w3": 0.0}

        rospy.loginfo("Loading teacher: %s" % args.teacher_ckpt)
        self.teacher_args = self._arch_namespace(args, "vitl")
        self.teacher_args.restore_ckpt = args.teacher_ckpt
        self.model = build_model(self.teacher_args, self.device)
        self.teacher_encoder = self.model.defomencoder.depth_anything
        rospy.loginfo("Teacher ready, %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

        student_path = resolve_student_checkpoint(args.student_ckpt)
        rospy.loginfo("Loading student: %s" % student_path)
        ck = torch.load(str(student_path), map_location=self.device)
        self.student = build_student(features=ck["out_dim"], **ck["student_cfg"]).to(self.device)
        self.student.load_state_dict(ck["student"])
        self.student.eval()
        self.student_epoch = ck.get("epoch", "?")
        rospy.loginfo("Student ready: epoch %s, backbone %s, %.1f MB"
                      % (self.student_epoch, ck["student_cfg"]["backbone"],
                         sum(p.numel() for p in self.student.parameters()) * 4 / 1e6))

        ns = args.output_ns.rstrip("/")
        self.pub_teacher_vis = rospy.Publisher(ns + "/teacher/depth/compressed", CompressedImage, queue_size=1)
        self.pub_teacher_depth = rospy.Publisher(ns + "/teacher/depth", Image, queue_size=1)
        self.pub_student_vis = rospy.Publisher(ns + "/student/depth/compressed", CompressedImage, queue_size=1)
        self.pub_student_depth = rospy.Publisher(ns + "/student/depth", Image, queue_size=1)
        self.pub_info = rospy.Publisher(ns + "/camera_info", CameraInfo, queue_size=1, latch=True)
        self.pub_compare = rospy.Publisher(args.compare_topic, CompressedImage, queue_size=1)

        self.sub = rospy.Subscriber(args.input_topic, CompressedImage, self.on_image,
                                    queue_size=1, buff_size=2 ** 24, tcp_nodelay=True)
        rospy.loginfo("Subscribed to %s" % args.input_topic)

    @staticmethod
    def _arch_namespace(args, encoder):
        return argparse.Namespace(
            dinov2_encoder=encoder, idepth_scale=args.idepth_scale,
            hidden_dims=list(args.hidden_dims), corr_implementation=args.corr_implementation,
            corr_levels=args.corr_levels, corr_radius=args.corr_radius,
            scale_list=list(args.scale_list), scale_corr_radius=args.scale_corr_radius,
            n_downsample=args.n_downsample, context_norm=args.context_norm,
            n_gru_layers=args.n_gru_layers, shared_backbone=args.shared_backbone,
            mixed_precision=args.mixed_precision,
        )

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
            rospy.logwarn_throttle(5.0, "Failed to decode CompressedImage")
            return None, None
        width = image.shape[1]
        if width % 2:
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
                rospy.logwarn("No calibration: comparing raw DISPARITY, not metric depth.")
            return
        calib_w = args.calib_width if args.calib_width > 0 else width
        calib_h = args.calib_height if args.calib_height > 0 else height
        sx, sy = float(width) / calib_w, float(height) / calib_h
        fx, fy = args.fx * sx, (args.fy if args.fy > 0 else args.fx) * sy
        cx = (args.cx if args.cx >= 0 else 0.5 * calib_w) * sx
        cy = (args.cy if args.cy >= 0 else 0.5 * calib_h) * sy
        self.intrinsics = (fx, fy, cx, cy)
        if args.fx_baseline > 0:
            self.fx_baseline = args.fx_baseline
        elif args.baseline_m > 0:
            self.fx_baseline = fx * args.baseline_m
        if self.fx_baseline:
            rospy.loginfo("Metric depth: fx*B = %.4f m.px" % self.fx_baseline)

    # ---------------------------------------------------------- inference

    def infer(self, left, right):
        image1 = to_tensor(left, self.device)
        image2 = to_tensor(right, self.device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        with torch.no_grad():
            disp = self.model(image1, image2, iters=self.args.valid_iters,
                              scale_iters=self.args.scale_iters, test_mode=True)
        return padder.unpad(disp).cpu().squeeze().numpy().astype(np.float32)

    def run_both(self, left, right):
        started = time.time()
        self.model.defomencoder.depth_anything = self.teacher_encoder
        disp_teacher = self.infer(left, right)
        t_teacher = time.time() - started

        started = time.time()
        self.model.defomencoder.depth_anything = self.student
        disp_student = self.infer(left, right)
        t_student = time.time() - started

        self.model.defomencoder.depth_anything = self.teacher_encoder
        return disp_teacher, disp_student, t_teacher, t_student

    # ---------------------------------------------------------- output

    @staticmethod
    def _raster(header, array):
        msg = Image()
        msg.header = header
        msg.height, msg.width = array.shape
        msg.encoding = "32FC1"
        msg.step = 4 * array.shape[1]
        msg.data = array.tobytes()
        return msg

    def _compressed(self, header, bgr):
        msg = CompressedImage()
        msg.header = header
        msg.format = "jpeg"
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality])
        if ok:
            msg.data = buf.tobytes()
        return msg

    def to_field(self, disp):
        valid = np.isfinite(disp) & (disp > self.args.min_disparity)
        if self.fx_baseline is None:
            return disp, valid, False
        depth = np.zeros_like(disp)
        np.divide(self.fx_baseline, disp, out=depth, where=valid)
        depth[~valid] = 0.0
        return depth, valid & (depth > 0), True

    def publish(self, header, left, disp_teacher, disp_student, t_teacher, t_student):
        field_t, valid_t, is_depth = self.to_field(disp_teacher)
        field_s, valid_s, _ = self.to_field(disp_student)

        for disp, field, pub_raster, pub_vis, label in (
                (disp_teacher, field_t, self.pub_teacher_depth, self.pub_teacher_vis, "teacher"),
                (disp_student, field_s, self.pub_student_depth, self.pub_student_vis, "student")):
            pub_raster.publish(self._raster(header, field if is_depth else disp))

        # ONE colour scale, shared by both panels -- see module docstring for
        # why this matters. Default: 2nd-98th percentile of the teacher's own
        # frame, recomputed every frame -- always uses full colour contrast,
        # but the same colour does NOT mean the same depth across frames.
        # Pass --vis_min/--vis_max for a fixed scale instead -- same colour
        # means the same absolute depth everywhere, comparable over time and
        # against /ikan/explore3d/depth/image_color, which uses a fixed scale
        # of its own (not these values; that node's range is not exposed).
        if self.args.vis_min is not None and self.args.vis_max is not None:
            lo, hi = self.args.vis_min, self.args.vis_max
        elif valid_t.any():
            lo, hi = float(np.percentile(field_t[valid_t], 2)), float(np.percentile(field_t[valid_t], 98))
        else:
            lo, hi = None, None
        vis_t = colorise(field_t, valid_t, lo, hi, self.colormap, is_depth)
        vis_s = colorise(field_s, valid_s, lo, hi, self.colormap, is_depth)
        self.pub_teacher_vis.publish(self._compressed(header, vis_t))
        self.pub_student_vis.publish(self._compressed(header, vis_s))

        if self.pub_info is not None and self.intrinsics is not None:
            fx, fy, cx, cy = self.intrinsics
            info = CameraInfo()
            info.header = header
            info.width, info.height = disp_teacher.shape[1], disp_teacher.shape[0]
            info.distortion_model = "plumb_bob"
            info.D = [0.0] * 5
            info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            self.pub_info.publish(info)

        def tile(vis, name, ms):
            t = vis.copy()
            for text, y, scale in ((name, 26, 0.8), ("%.0f ms" % ms, t.shape[0] - 10, 0.6)):
                cv2.putText(t, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(t, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
            return t

        canvas = np.hstack([left, tile(vis_t, "teacher (vitl)", t_teacher * 1000),
                            tile(vis_s, "student ep%s" % self.student_epoch, t_student * 1000)])
        self.pub_compare.publish(self._compressed(header, canvas))

    # --------------------------------------------------------- agreement

    def update_agreement(self, disp_teacher, disp_student):
        valid = (np.isfinite(disp_teacher) & (disp_teacher > self.args.min_disparity)
                & np.isfinite(disp_student) & (disp_student > self.args.min_disparity))
        if not valid.any():
            return
        diff = np.abs(disp_student[valid] - disp_teacher[valid])
        acc = self.agreement
        acc["n"] += 1
        for key, value in (("mean", diff.mean()), ("median", np.median(diff)),
                           ("p90", np.percentile(diff, 90)), ("w1", (diff < 1).mean()),
                           ("w2", (diff < 2).mean()), ("w3", (diff < 3).mean())):
            acc[key] += float(value)

    def log_agreement(self):
        acc = self.agreement
        n = max(1, acc["n"])
        rospy.loginfo("  student vs teacher: mean %.2fpx median %.2f p90 %.2f  "
                      "within 1/2/3px %.0f%%/%.0f%%/%.0f%%  (%d frames)",
                      acc["mean"] / n, acc["median"] / n, acc["p90"] / n,
                      100 * acc["w1"] / n, 100 * acc["w2"] / n, 100 * acc["w3"] / n, acc["n"])

    # --------------------------------------------------------------- loop

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
                disp_t, disp_s, t_t, t_s = self.run_both(left, right)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    rospy.logerr_throttle(10.0, "CUDA OOM: %s" % exc)
                    continue
                raise

            self.publish(msg.header, left, disp_t, disp_s, t_t, t_s)
            self.update_agreement(disp_t, disp_s)
            self.processed += 1

            now = time.time()
            if now - self.last_report >= self.args.report_period:
                rospy.loginfo("%.2f Hz | teacher %.0fms | student %.0fms | processed %d | dropped %d",
                              self.processed / max(1e-6, now - self.last_report),
                              t_t * 1000, t_s * 1000, self.processed, self.dropped)
                self.log_agreement()
                self.processed = 0
                self.last_report = now


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--teacher_ckpt', default='checkpoints/defomstereo_vitl_sceneflow.pth')
    parser.add_argument('--student_ckpt', default=DEFAULT_CKPT_DIR,
                        help='a student_epoch*.pth file, or a directory to auto-pick the latest')

    ros = parser.add_argument_group("ROS")
    ros.add_argument('--input_topic', default='/ikan/explore3d/stereo/compressed')
    ros.add_argument('--output_ns', default='/ikan/explore3d/defom')
    ros.add_argument('--compare_topic', default='/ikan/explore3d/defom/compare/compressed')
    ros.add_argument('--node_name', default='defom_live_compare')
    ros.add_argument('--poll_hz', type=float, default=100.0)
    ros.add_argument('--report_period', type=float, default=5.0)

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

    run = parser.add_argument_group("runtime")
    run.add_argument('--device', default='cuda')

    vis = parser.add_argument_group("visualisation")
    vis.add_argument('--colormap', default='jet', choices=sorted(COLORMAPS))
    vis.add_argument('--vis_min', type=float, default=None,
                     help='fixed colour-scale minimum (metres, if calibrated; px otherwise). '
                          'Default (unset): 2nd-98th percentile of the teacher, recomputed every '
                          'frame -- always full contrast, but not comparable across frames. Set '
                          'both --vis_min/--vis_max for a fixed scale, e.g. to match '
                          '/ikan/explore3d/depth/image_color or to compare frames over time.')
    vis.add_argument('--vis_max', type=float, default=None)
    vis.add_argument('--jpeg_quality', type=int, default=90)

    arch = parser.add_argument_group("architecture (teacher, must match the checkpoint)")
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

    return parser.parse_args(argv)


def main():
    args = parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node(args.node_name, anonymous=False)
    node = LiveCompareNode(args)
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
