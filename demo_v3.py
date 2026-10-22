#!/usr/bin/env python3
"""DEFOM-Stereo ROS1 node running a DISTILLED STUDENT (~94 MB instead of 1530 MB).

    python3 demo_v3.py                                   # multi-scale run's student_best.pth
    python3 demo_v3.py --run middlebury                  # another trained run
    python3 demo_v3.py --student_ckpt distillation/checkpoints/<run>[/student_epoch0019.pth]

A run folder resolves to its student_best.pth, else its newest epoch. The
architecture, the stereo core (teacher checkpoint) and the iteration counts
all come from the student checkpoint, since a mismatch would run without error
but give wrong disparity. Topics, calibration, point cloud and main loop are
demo_v2.py's (this subclasses its node).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "distillation"))

import rospy

from core.defom_stereo import DEFOMStereo
from defom_student import load_student
from demo_v2 import DefomStereoNode, add_node_args, load_teacher_weights

RUN_DIRS = {
    "sceneflow": "distillation/checkpoints/underwater_vitl_to_student",
    "middlebury": "distillation/checkpoints/underwater_vitl_middlebury_to_student",
    "multiscale": "distillation/checkpoints/underwater_vitl_middlebury_to_student_multiscale",
}


def load_student_model(student_ckpt, device):
    """DEFOMStereo built from the checkpoint's own arch and teacher, with the
    student swapped in for the teacher encoder (which is then freed).
    Returns (model, checkpoint dict)."""
    student, ck, student_path = load_student(student_ckpt, device)
    rospy.loginfo("Loaded student: %s", student_path)
    teacher_ckpt = ck["config"]["paths"]["teacher_ckpt"]
    rospy.loginfo("Stereo core (recorded in the checkpoint): %s", teacher_ckpt)
    model = load_teacher_weights(DEFOMStereo(argparse.Namespace(**ck["config"]["arch"])), teacher_ckpt)
    del model.defomencoder.depth_anything   # frees the ~1.5 GB teacher encoder
    model.defomencoder.depth_anything = student
    model.to(device).eval()
    rospy.loginfo("Student ready: epoch %s, %.1f MB", ck.get("epoch", "?"),
                  sum(p.numel() for p in student.parameters()) * 4 / 1e6)
    return model, ck


class DefomStudentNode(DefomStereoNode):
    def _build_model(self):
        model, ck = load_student_model(self.args.student_ckpt, self.device)
        # Unless overridden on the command line, use the checkpoint's own iterations.
        if self.args.valid_iters is None:
            self.args.valid_iters = ck["config"]["arch"]["valid_iters"]
        if self.args.scale_iters is None:
            self.args.scale_iters = ck["config"]["arch"]["scale_iters"]
        return model


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default='multiscale', choices=sorted(RUN_DIRS),
                        help='trained run to load (ignored if --student_ckpt is given)')
    parser.add_argument('--student_ckpt', default=None, help='a .pth file or a run folder')
    parser.add_argument('--valid_iters', type=int, default=None, help="default: the checkpoint's own")
    parser.add_argument('--scale_iters', type=int, default=None, help="default: the checkpoint's own")
    add_node_args(parser, 'defom_student_node')
    args = parser.parse_args(argv)
    if args.student_ckpt is None:
        args.student_ckpt = RUN_DIRS[args.run]
    return args


def main():
    args = parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node(args.node_name, anonymous=False)
    node = DefomStudentNode(args)
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
