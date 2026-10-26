#!/usr/bin/env python3
"""DEFOM-Stereo ROS1 node running a distilled model instead of the 1530 MB teacher.

    python3 demo_v3.py                                   # multi-scale student (student_best.pth)
    python3 demo_v3.py --run slim_v9                     # slim DEFOM v9 (the Orin NX model)
    python3 demo_v3.py --student_ckpt distillation/checkpoints/<run>[/student_epoch0005.pth]

A run folder resolves to its student_best.pth, else its newest epoch. Two
kinds of checkpoint work:
  * a student (distill_defom.py): swapped into DEFOM's stereo core, which is
    read from the checkpoint's recorded teacher (~94 MB in total);
  * a slim DEFOM (train_slim.py): a complete, faster stereo model, run as is.
Architecture and iteration counts come from the checkpoint, since a mismatch
would run without error but give wrong disparity. Topics, calibration, point
cloud and main loop are demo_v2.py's (this subclasses its node).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "distillation"))

import rospy
import torch

from core.defom_stereo import DEFOMStereo
from defom_student import load_student, resolve_student_checkpoint
from demo_v2 import DefomStereoNode, add_node_args, load_teacher_weights
from slim_stereo import load_slim

RUN_DIRS = {
    "multiscale": "distillation/checkpoints/underwater_vitl_middlebury_to_student_multiscale",
    "slim_v9": "distillation/checkpoints/slim_v9",
    "slim_v9_joint": "distillation/checkpoints/slim_v9_joint",
}


def load_model(ckpt, device):
    """Returns (model, checkpoint dict, default (iters, scale_iters))."""
    path = resolve_student_checkpoint(ckpt)
    if "slim" in torch.load(str(path), map_location="cpu", weights_only=True):
        model, ck, path = load_slim(path, device)
        rospy.loginfo("Loaded slim DEFOM %s: %s (epoch %s)", ck.get("preset"), path, ck.get("epoch"))
        return model, ck, (ck["config"]["training"]["iters"], ck["config"]["training"]["scale_iters"])
    student, ck, path = load_student(path, device)
    teacher_ckpt = ck["config"]["paths"]["teacher_ckpt"]
    rospy.loginfo("Loaded student %s (epoch %s); stereo core from %s", path, ck.get("epoch"), teacher_ckpt)
    model = load_teacher_weights(DEFOMStereo(argparse.Namespace(**ck["config"]["arch"])), teacher_ckpt)
    del model.defomencoder.depth_anything   # frees the ~1.5 GB teacher encoder
    model.defomencoder.depth_anything = student
    model.to(device).eval()
    return model, ck, (ck["config"]["arch"]["valid_iters"], ck["config"]["arch"]["scale_iters"])


class DefomStudentNode(DefomStereoNode):
    def _build_model(self):
        model, ck, (iters, scale_iters) = load_model(self.args.student_ckpt, self.device)
        # Unless overridden on the command line, use the checkpoint's own iterations.
        if self.args.valid_iters is None:
            self.args.valid_iters = iters
        if self.args.scale_iters is None:
            self.args.scale_iters = scale_iters
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
