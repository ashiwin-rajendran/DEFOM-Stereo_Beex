#!/usr/bin/env python3
"""DEFOM-Stereo ROS1 node running a DISTILLED STUDENT instead of the full
vitl/vits encoder -- same stream, same topics, same math as demo_v2.py, just
a ~75-95 MB encoder instead of 1530 MB.

    python3 demo_v3.py                       # latest sceneflow-distilled student
    python3 demo_v3.py --run middlebury      # latest middlebury-distilled student
    python3 demo_v3.py --student_ckpt distillation/checkpoints/.../student_epoch0019.pth

Two trained lines exist, named after the DEFOM-Stereo checkpoint each one's
student was distilled to match (see distillation/README.md):

    sceneflow   distillation/checkpoints/underwater_vitl_to_student/
    middlebury  distillation/checkpoints/underwater_vitl_middlebury_to_student/

--run picks the directory and auto-resolves its highest-epoch student_epoch
####.pth -- no path to edit after every resume. --student_ckpt overrides with
an exact file when you want a specific epoch instead of the latest.

WHY THIS FILE IS SHORT: every student checkpoint records its own full
training config (architecture, and which DEFOM checkpoint its stereo core and
calibration target came from) under checkpoint["config"]. Loading reads that
back directly instead of taking ~15 architecture flags on the CLI the way
demo_v2.py's --restore_ckpt path does -- there is no "mixed_precision" or
"hidden_dims" flag here, because passing one that does not match the
checkpoint would silently build the wrong shapes. One fact, one source.

Everything else -- publishers, calibration, point-cloud TF lookup, the main
loop -- is UNCHANGED from demo_v2.py: this subclasses DefomStereoNode and
overrides only model construction (see demo_v2.py's _build_model docstring
for why that seam exists). A bug fixed in demo_v2.py's plumbing is fixed here
too, by construction, not by remembering to port it.

Student encoders are small enough that --warmup and steady-state both run
noticeably lighter than demo_v2.py: no second ~1.5 GB encoder is ever held in
memory -- the teacher encoder used transiently to build a correctly-shaped
stereo core is freed immediately after the student is swapped in (see
load_student_model below).
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import rospy
import torch

sys.path.append('core')
sys.path.append('distillation')

from core.defom_stereo import DEFOMStereo
from defom_student import build_student
from demo_v2 import DefomStereoNode, COLORMAPS


RUN_DIRS = {
    "sceneflow": "distillation/checkpoints/underwater_vitl_to_student",
    "middlebury": "distillation/checkpoints/underwater_vitl_middlebury_to_student",
}


def resolve_student_checkpoint(path):
    """A file is used as-is. A directory resolves to its highest-epoch
    student_epoch####.pth, so --run keeps working across resumes without
    anyone editing a flag."""
    p = Path(path)
    if p.is_file():
        return p
    candidates = sorted(Path(path).glob("student_epoch*.pth"),
                        key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)))
    if not candidates:
        raise FileNotFoundError("No student_epoch*.pth under %s" % path)
    return candidates[-1]


def load_student_model(student_ckpt_path, device):
    """Build a DEFOMStereo with its encoder replaced by a distilled student.

    Three things come directly from the checkpoint rather than the CLI,
    because getting any of them wrong silently builds mismatched shapes:

    1. Architecture (checkpoint["config"]["arch"]) -- the GRU/correlation
       stack dimensions the student's output must match.
    2. The teacher checkpoint path (checkpoint["config"]["paths"]["teacher_ckpt"])
       -- the stereo core (fnet/cnet/GRU) was trained alongside THAT specific
       encoder, and the student's own output statistics were calibrated
       against it (see distillation/distill_defom.py's calibrate_output_scales).
       Pairing a student with the wrong core's weights, or the right core but
       the wrong calibration target, both produce a model that runs without
       error and gives degraded disparity -- there is no shape mismatch to
       catch it. So this is never a flag; it is read from the one place that
       cannot drift out of sync with what was actually trained.
    3. student_cfg / out_dim -- backbone choice and width.

    The teacher encoder is only needed transiently, to build a DEFOMStereo
    whose fnet/cnet/GRU weights load correctly (their input dims come from
    the teacher's out_dim). It is discarded immediately after the swap --
    this deployment only ever needs ONE encoder resident.
    """
    student_path = resolve_student_checkpoint(student_ckpt_path)
    rospy.loginfo("Loading student: %s", student_path)
    ck = torch.load(str(student_path), map_location=device)

    arch_args = argparse.Namespace(**ck["config"]["arch"])
    teacher_ckpt = ck["config"]["paths"]["teacher_ckpt"]
    rospy.loginfo("Stereo core + calibration target: %s (as recorded in the checkpoint)", teacher_ckpt)

    model = DEFOMStereo(arch_args)
    teacher_state = torch.load(teacher_ckpt, map_location=device)
    model.load_state_dict(teacher_state["model"] if "model" in teacher_state else teacher_state)

    # mobilevit_init_from in student_cfg is the ORIGINAL training-time warm
    # start path -- irrelevant here (ck["student"] below overwrites every
    # weight it would set) and actively harmful: it is an absolute,
    # machine-specific path baked in at training time (confirmed to differ
    # between this machine and the DGX Spark), so loading it as recorded
    # would crash deployment on any machine but the one that trained it, for
    # a value that cannot change the outcome. Stripped before construction.
    student_cfg = dict(ck["student_cfg"], mobilevit_init_from="")
    student = build_student(features=ck["out_dim"], **student_cfg).to(device)
    student.load_state_dict(ck["student"])
    student.eval()

    del model.defomencoder.depth_anything   # frees the ~1.5 GB teacher encoder
    model.defomencoder.depth_anything = student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.to(device).eval()

    n_student = sum(p.numel() for p in student.parameters())
    rospy.loginfo("Student ready: epoch %s, backbone %s, %.1f MB (teacher encoder freed)",
                  ck.get("epoch", "?"), ck["student_cfg"]["backbone"], n_student * 4 / 1e6)
    return model


class DefomStudentNode(DefomStereoNode):
    def _build_model(self):
        return load_student_model(self.args.student_ckpt, self.device)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--run', default='sceneflow', choices=sorted(RUN_DIRS),
                        help='which distilled line to load the latest checkpoint from. '
                             'Overridden by --student_ckpt if given.')
    parser.add_argument('--student_ckpt', default=None,
                        help='an exact student_epoch####.pth, or a directory to auto-pick the '
                             'latest from. Overrides --run.')

    # --- ROS --- (identical defaults to demo_v2.py; same stream, same topics)
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
    ros.add_argument('--node_name', default='defom_student_node')
    ros.add_argument('--poll_hz', type=float, default=100.0,
                     help='how often the main loop checks for a new frame')
    ros.add_argument('--report_period', type=float, default=5.0, help='seconds between throughput logs')

    # --- stereo geometry --- (same camera as demo_v2.py, same defaults)
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
                      help='width the intrinsics were calibrated at')
    geom.add_argument('--calib_height', type=int, default=1200, help='height the intrinsics were calibrated at')
    geom.add_argument('--baseline_m', type=float, default=0.1,
                      help='stereo baseline in metres (explore3D Cobalt mechanical drawing)')
    geom.add_argument('--fx_baseline', type=float, default=0,
                      help='fx * baseline directly, overriding --baseline_m. 0 = use --baseline_m')
    geom.add_argument('--min_disparity', type=float, default=0.5,
                      help='disparity at or below this is treated as invalid')
    geom.add_argument('--max_depth_m', type=float, default=0.0,
                      help='clamp depth beyond this to invalid; 0 disables')

    # --- point cloud ---
    pts = parser.add_argument_group("point cloud")
    pts.add_argument('--publish_points', action='store_true', default=True)
    pts.add_argument('--no_points', dest='publish_points', action='store_false')
    pts.add_argument('--points_frame_id', default='explore3d',
                     help='frame_id stamped on the cloud; rotated via live TF from the image frame')
    pts.add_argument('--points_stride', type=int, default=2,
                     help='take every Nth pixel in both axes')
    pts.add_argument('--points_color', action='store_true', default=True)
    pts.add_argument('--no_points_color', dest='points_color', action='store_false')
    pts.add_argument('--points_tf_timeout', type=float, default=1.0)
    pts.add_argument('--points_tf_refresh', type=float, default=5.0)

    # --- runtime ---
    run = parser.add_argument_group("runtime")
    run.add_argument('--device', default='cuda')
    run.add_argument('--input_scale', type=float, default=1.0,
                     help='downscale factor before inference, e.g. 0.5 for speed/VRAM')
    run.add_argument('--warmup', action='store_true', default=True)
    run.add_argument('--no_warmup', dest='warmup', action='store_false')
    run.add_argument('--warmup_width', type=int, default=800)
    run.add_argument('--warmup_height', type=int, default=600)
    # Inference-time speed/quality knobs -- NOT architecture, so these stay as
    # CLI flags. Default to None: "use whatever the checkpoint's own training
    # config used" (set after parsing, from the loaded checkpoint).
    run.add_argument('--valid_iters', type=int, default=None,
                     help='disparity refinement iterations; default is the checkpoint\'s own value')
    run.add_argument('--scale_iters', type=int, default=None,
                     help='scaling updates per forward pass; default is the checkpoint\'s own value')

    # --- visualisation ---
    vis = parser.add_argument_group("visualisation")
    vis.add_argument('--colormap', default='jet', choices=sorted(COLORMAPS))
    vis.add_argument('--vis_min', type=float, default=None,
                     help='fixed colour-scale minimum; default is a 2nd-98th percentile stretch per frame')
    vis.add_argument('--vis_max', type=float, default=None)
    vis.add_argument('--vis_format', default='jpeg', choices=['jpeg', 'png'])
    vis.add_argument('--jpeg_quality', type=int, default=90)

    return parser.parse_args(argv)


def main():
    args = parse_args(rospy.myargv(argv=sys.argv)[1:])
    if args.student_ckpt is None:
        args.student_ckpt = RUN_DIRS[args.run]

    rospy.init_node(args.node_name, anonymous=False)

    # valid_iters/scale_iters: fill in from the checkpoint's own training
    # config if the user did not override them, so the CLI default is never
    # silently wrong for whichever checkpoint got loaded.
    if args.valid_iters is None or args.scale_iters is None:
        ckpt = torch.load(str(resolve_student_checkpoint(args.student_ckpt)), map_location="cpu")
        if args.valid_iters is None:
            args.valid_iters = ckpt["config"]["arch"]["valid_iters"]
        if args.scale_iters is None:
            args.scale_iters = ckpt["config"]["arch"]["scale_iters"]
        del ckpt

    node = DefomStudentNode(args)
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
