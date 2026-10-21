#!/usr/bin/env python3
"""Record side-by-side stereo frames from the live ROS topic into a training set.

Distillation needs no ground-truth disparity -- the student is trained to match
the teacher's outputs -- so any unlabelled stereo footage from the deployment
domain works, and more of it is strictly better.

    python3 capture_stereo.py --out datasets/underwater --max_frames 4000

Each run writes its own session folder, <out>/<session>/NNNNNN.png, named
<eyeW>x<eyeH>_<timestamp> unless --session is given. Any side-by-side size
works (DWE SVC 1600x600, DWE raw 3200x1200, Voyis 5632x2816): the trainer
groups frames by session for its temporal validation split and by per-eye
size for balanced sampling, so sources can be mixed freely under one root.
Frames must already be rectified.

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
    parser.add_argument('--out', required=True, help='dataset root; a session folder is created inside it')
    parser.add_argument('--session', default='',
                        help='session folder name; default <eyeW>x<eyeH>_<timestamp> from the first frame')
    parser.add_argument('--max_frames', type=int, default=4000)
    parser.add_argument('--min_mad', type=float, default=2.0,
                        help='minimum mean-abs-difference vs the last kept frame; 0 keeps everything')
    parser.add_argument('--expect_width', type=int, default=0,
                        help='warn if the side-by-side width differs; 0 accepts any size')
    parser.add_argument('--format', default='png', choices=['png', 'jpg'],
                        help='png is lossless -- preferred, since the teacher sees these exact pixels')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node('stereo_capture', anonymous=True)

    state = {'kept': 0, 'seen': 0, 'last': None, 'start': time.time(), 'dir': None}

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
        if state['dir'] is None:
            # Session folder is named after the first frame's per-eye size.
            session = args.session or "%dx%d_%s" % (image.shape[1] // 2, image.shape[0],
                                                    time.strftime("%Y%m%d_%H%M%S"))
            state['dir'] = Path(args.out) / session
            state['dir'].mkdir(parents=True, exist_ok=True)
            rospy.loginfo("Writing to %s", state['dir'])
        # Aspect-preserving thumbnail, so the dedup threshold means the same
        # thing for every camera.
        small = cv2.resize(image, (160, max(1, round(160 * image.shape[0] / image.shape[1]))),
                           interpolation=cv2.INTER_AREA)
        if state['last'] is not None and args.min_mad > 0:
            if np.abs(small.astype(np.float32) - state['last']).mean() < args.min_mad:
                return
        state['last'] = small.astype(np.float32)
        name = state['dir'] / ("%06d.%s" % (state['kept'], args.format))
        cv2.imwrite(str(name), image)
        state['kept'] += 1
        if state['kept'] % 50 == 0:
            rate = state['kept'] / max(1e-6, time.time() - state['start'])
            rospy.loginfo("kept %d / seen %d  (%.1f kept/s)", state['kept'], state['seen'], rate)

    rospy.Subscriber(args.topic, CompressedImage, callback, queue_size=2, buff_size=2 ** 24)
    rospy.loginfo("Recording %s -> %s (target %d frames)", args.topic, args.out, args.max_frames)
    while not rospy.is_shutdown() and state['kept'] < args.max_frames:
        rospy.sleep(0.1)
    rospy.loginfo("Done: kept %d of %d seen, in %s", state['kept'], state['seen'], state['dir'])


if __name__ == '__main__':
    main()
