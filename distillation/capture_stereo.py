#!/usr/bin/env python3
"""Record side-by-side stereo frames from the live ROS topic into a training set.

Distillation needs no ground-truth disparity -- the student is trained to match
the teacher's outputs -- so any unlabelled stereo footage from the deployment
domain works, and more of it is strictly better.

    python3 capture_stereo.py --out datasets/underwater --max_frames 4000

Frames are deduplicated by mean absolute difference against the last kept
frame. Survey footage is highly redundant at 6.5 Hz: without this you get
thousands of near-identical images that inflate epoch time without adding
information, and bias the student toward whatever the vehicle lingered on.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--topic', default='/ikan/explore3d/stereo/compressed')
    parser.add_argument('--out', required=True, help='output directory')
    parser.add_argument('--max_frames', type=int, default=4000)
    parser.add_argument('--min_mad', type=float, default=2.0,
                        help='minimum mean-abs-difference vs the last kept frame; 0 keeps everything')
    parser.add_argument('--expect_width', type=int, default=1600)
    parser.add_argument('--format', default='png', choices=['png', 'jpg'],
                        help='png is lossless -- preferred, since the teacher sees these exact pixels')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rospy.init_node('stereo_capture', anonymous=True)

    state = {'kept': 0, 'seen': 0, 'last': None, 'start': time.time()}

    def callback(msg):
        if state['kept'] >= args.max_frames:
            return
        state['seen'] += 1
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        if args.expect_width > 0 and image.shape[1] != args.expect_width:
            rospy.logwarn_throttle(10.0, "Frame is %dx%d, expected width %d",
                                   image.shape[1], image.shape[0], args.expect_width)
        small = cv2.resize(image, (160, 60))
        if state['last'] is not None and args.min_mad > 0:
            if np.abs(small.astype(np.float32) - state['last']).mean() < args.min_mad:
                return
        state['last'] = small.astype(np.float32)
        name = out / ("%06d.%s" % (state['kept'], args.format))
        cv2.imwrite(str(name), image)
        state['kept'] += 1
        if state['kept'] % 50 == 0:
            rate = state['kept'] / max(1e-6, time.time() - state['start'])
            rospy.loginfo("kept %d / seen %d  (%.1f kept/s)", state['kept'], state['seen'], rate)

    rospy.Subscriber(args.topic, CompressedImage, callback, queue_size=2, buff_size=2 ** 24)
    rospy.loginfo("Recording %s -> %s (target %d frames)", args.topic, out, args.max_frames)
    while not rospy.is_shutdown() and state['kept'] < args.max_frames:
        rospy.sleep(0.1)
    rospy.loginfo("Done: kept %d of %d seen, in %s", state['kept'], state['seen'], out)


if __name__ == '__main__':
    main()
