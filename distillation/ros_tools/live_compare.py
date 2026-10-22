#!/usr/bin/env python3
"""Live teacher-vs-student comparison on the stereo ROS stream.

    python3 distillation/ros_tools/live_compare.py [--student_ckpt distillation/checkpoints/<run>]

Runs the full teacher and a distilled student on every frame (one DEFOM
instance; the student is swapped in for its pass) and publishes:

    <output_ns>/teacher/depth[/compressed]   metric 32FC1 / colourised
    <output_ns>/student/depth[/compressed]   same, on the TEACHER's colour scale
    <compare_topic>                          left | teacher | student

and logs agreement as in training validation (mean/median/p90 px, % within
1/2/3 px). The stereo core and architecture come from the student checkpoint
(default: the multi-scale run's student_best.pth). Frames are resized to
--long_side (640, as deployed).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "distillation"))

import argparse
import threading
import time
import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from core.defom_stereo import DEFOMStereo
from core.utils.utils import InputPadder
from demo_v2 import COLORMAPS, colorise, float_image, load_teacher_weights, repo_path, to_tensor
from defom_student import load_student


DEFAULT_CKPT_DIR = "distillation/checkpoints/underwater_vitl_middlebury_to_student_multiscale"


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

        self.student, ck, student_path = load_student(args.student_ckpt, self.device)
        rospy.loginfo("Loaded student: %s" % student_path)
        if args.valid_iters is None:
            args.valid_iters = ck["config"]["arch"]["valid_iters"]
        if args.scale_iters is None:
            args.scale_iters = ck["config"]["arch"]["scale_iters"]

        teacher_ckpt = repo_path(args.teacher_ckpt or ck["config"]["paths"]["teacher_ckpt"])
        rospy.loginfo("Loading teacher (the student's own stereo core): %s" % teacher_ckpt)
        arch = argparse.Namespace(**dict(ck["config"]["arch"], mixed_precision=args.mixed_precision))
        self.model = load_teacher_weights(DEFOMStereo(arch), teacher_ckpt).to(self.device).eval()
        self.teacher_encoder = self.model.defomencoder.depth_anything
        rospy.loginfo("Teacher ready, %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

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
        long_side = self.args.long_side
        if long_side > 0 and max(left.shape[:2]) != long_side:
            # Same resampling as training and the deployed node.
            scale = float(long_side) / max(left.shape[:2])
            size = (int(round(left.shape[1] * scale)), int(round(left.shape[0] * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            left, right = cv2.resize(left, size, interpolation=interp), cv2.resize(right, size, interpolation=interp)
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

        self.pub_teacher_depth.publish(float_image(header, field_t))
        self.pub_student_depth.publish(float_image(header, field_s))

        # ONE colour scale for both (the teacher's 2-98th percentile, or
        # --vis_min/--vis_max): separate stretches would hide disagreement.
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

        if self.intrinsics is not None:
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
            except torch.cuda.OutOfMemoryError as exc:
                self.model.defomencoder.depth_anything = self.teacher_encoder
                torch.cuda.empty_cache()
                rospy.logerr_throttle(10.0, "CUDA OOM: %s" % exc)
                continue

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

    parser.add_argument('--student_ckpt', default=DEFAULT_CKPT_DIR,
                        help='a .pth file, or a run folder (its student_best.pth, else the latest epoch)')
    parser.add_argument('--teacher_ckpt', default=None,
                        help='override the stereo core; default is the one recorded in the student '
                             'checkpoint, which is the one it was distilled against')

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
    run.add_argument('--long_side', type=int, default=640,
                     help='resize each eye so its long side is this (aspect kept); 0 = native size')
    run.add_argument('--mixed_precision', action='store_true')
    run.add_argument('--valid_iters', type=int, default=None,
                     help='refinement iterations; default is the checkpoint\'s own value')
    run.add_argument('--scale_iters', type=int, default=None,
                     help='scaling updates; default is the checkpoint\'s own value')

    vis = parser.add_argument_group("visualisation")
    vis.add_argument('--colormap', default='jet', choices=sorted(COLORMAPS))
    vis.add_argument('--vis_min', type=float, default=None,
                     help='fixed colour range (with --vis_max; metres if calibrated, else px); '
                          'default: the teacher\'s 2-98th percentile per frame')
    vis.add_argument('--vis_max', type=float, default=None)
    vis.add_argument('--jpeg_quality', type=int, default=90)

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
