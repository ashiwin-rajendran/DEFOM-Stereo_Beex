#!/usr/bin/env python3
"""Record training frames from separate left / right camera topics (public stereo bags): pair by
stamp, rectify from each camera's CameraInfo, save side-by-side like capture_stereo.py.

    rosbag play --loop bus_outside.bag
    python3 distillation/7_ros_tools/capture_public_stereo.py --out distillation/1_datasets/public \\
        --left /slave1/image_raw/compressed --right /slave2/image_raw/compressed --session bus_outside

--left / --right: sensor_msgs/CompressedImage (or Image) per camera, default /cam_fl/image_raw/compressed
and /cam_fr/image_raw/compressed (cemetery.bag). The left camera is the one whose rectified P has
P[0,3] = 0; the right one's is negative (-fx * baseline): swapped topics are refused.

Frames are paired by header stamp (closest within --max_dt), then undistorted and rectified from
<topic>/cam_info (or --left_info / --right_info) straight to --width (default 800: 800x600 for
1600x1200 cameras, the DWE training size). --no_rectify for bags whose images are already rectified
(they are only resized to --width).

Output, de-duplication and appending are capture_stereo.py's: <out>/<session>/NNNNNN.png (left | right),
--min_mad, --max_frames, --session. Keep public data out of 1_datasets/underwater unless it is meant
to be trained on with the dives (the configs' image_dir).

python3 capture_public_stereo.py --out /mnt/BeeX-Ashwin/Vision_Tools/Online_Registration/DEFOM_Stereo/DEFOM-Stereo_Beex/distillation/1_datasets/public --left /cam_fl/image_raw/compressed --right /cam_fr/image_raw/compressed --session cemetery

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_stereo as cs  # noqa: E402  (Recorder, output arguments)
from compare_beex_bags import cv2, rospy  # noqa: E402  (system OpenCV, rospy)
from compare_public_bags import Rectifier, decode_image  # noqa: E402
import message_filters  # noqa: E402
from sensor_msgs.msg import CameraInfo, CompressedImage, Image  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="/cam_fl/image_raw/compressed")
    ap.add_argument("--right", default="/cam_fr/image_raw/compressed")
    ap.add_argument("--left_info", default="", help="CameraInfo topic (default: <left topic's parent>/cam_info)")
    ap.add_argument("--right_info", default="", help="CameraInfo topic (default: <right topic's parent>/cam_info)")
    ap.add_argument("--max_dt", type=float, default=0.02, help="largest left/right stamp difference to pair, s")
    ap.add_argument("--width", type=int, default=800, help="saved width per eye (height keeps the aspect)")
    ap.add_argument("--no_rectify", action="store_true", help="images are already rectified: only resize")
    cs.add_args(ap)
    args = ap.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("stereo_capture_public", anonymous=True)

    rect = None
    if not args.no_rectify:
        infos = []
        for topic in (args.left_info or args.left.rsplit("/", 1)[0] + "/cam_info",
                      args.right_info or args.right.rsplit("/", 1)[0] + "/cam_info"):
            rospy.loginfo("waiting for %s (rosbag playing?)", topic)
            infos.append(rospy.wait_for_message(topic, CameraInfo))
        rect = Rectifier(infos[0], infos[1], args.width)
        rospy.loginfo("rectifying %dx%d -> %dx%d per eye (fx %.1f, baseline %.4f m)",
                      *rect.src, *rect.size, rect.fx, rect.baseline)

    def resize(img):
        h = max(1, round(img.shape[0] * args.width / img.shape[1]))
        interp = cv2.INTER_AREA if args.width < img.shape[1] else cv2.INTER_LINEAR
        return cv2.resize(img, (args.width, h), interpolation=interp)

    rec = cs.Recorder(args)

    def callback(left_msg, right_msg):
        if rec.done:
            return
        try:
            left, right = decode_image(left_msg), decode_image(right_msg)
            if rect is not None:
                left, right = rect(left, 0), rect(right, 1)
            elif left.shape != right.shape:
                raise ValueError("left %s and right %s differ in size" % (left.shape, right.shape))
            else:
                left, right = resize(left), resize(right)
        except ValueError as e:
            rospy.logwarn_throttle(10, "skipping a pair: %s", e)
            return
        rec.add(left, right)

    subs = [message_filters.Subscriber(t, CompressedImage if t.endswith("compressed") else Image,
                                       queue_size=4, buff_size=2 ** 24) for t in (args.left, args.right)]
    message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=args.max_dt).registerCallback(callback)
    rec.wait("%s + %s" % (args.left, args.right))


if __name__ == "__main__":
    main()
