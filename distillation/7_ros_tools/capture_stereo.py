#!/usr/bin/env python3
"""Record side-by-side stereo frames from a ROS topic (live or a rosbag) into a training session.

    rosbag play <dive>.beex
    python3 distillation/7_ros_tools/capture_stereo.py --out distillation/1_datasets/underwater          # DWE
    python3 distillation/7_ros_tools/capture_stereo.py --out distillation/1_datasets/underwater \\
        --topic /ikan/voyis/stereo_concatenated/compressed --session voyis_dive3                          # Voyis

Any stereo message compare_beex_bags.py reads works (decoded with the definition the bag sends):
bx_msgs/DepthImage (DWE /ikan/explore3d/depth_image), side-by-side CompressedImage / Image
(Voyis stereo_concatenated), bx_msgs/StereoImage. Frames must already be rectified; for separate
left / right camera topics (public datasets) use capture_public_stereo.py.

Each run writes <out>/<session>/NNNNNN.png (left | right); the session defaults to
<eyeW>x<eyeH>_<time>, and reusing a session name appends after its existing frames.
Near-duplicates (mean abs difference below --min_mad on a thumbnail) are skipped, so record while
moving.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_beex_bags as cb  # noqa: E402  (system OpenCV, rospy, AnyTopic, decode)
from compare_beex_bags import cv2, np, rospy  # noqa: E402


class Recorder:
    """Writes stereo pairs into one session folder: <out>/<session>/NNNNNN.png (left | right).
    Shared with capture_public_stereo.py, so both write the same sessions."""

    def __init__(self, args):
        self.args = args
        self.kept = self.seen = self.first = 0
        self.last, self.dir = None, None
        self.start = time.time()
        self.lock = threading.Lock()

    @property
    def done(self):
        return self.kept >= self.args.max_frames

    def add(self, left, right):
        a = self.args
        with self.lock:
            if self.done:
                return
            self.seen += 1
            image = np.hstack([left, right])
            if self.dir is None:              # the session is named after the first frame's per-eye size
                session = a.session or "%dx%d_%s" % (left.shape[1], left.shape[0], time.strftime("%Y%m%d_%H%M%S"))
                self.dir = Path(a.out) / session
                self.dir.mkdir(parents=True, exist_ok=True)
                existing = [int(p.stem) for p in self.dir.iterdir() if p.stem.isdigit()]
                self.first = max(existing) + 1 if existing else 0      # never overwrite: continue after the last
                rospy.loginfo("Writing to %s (from frame %d)", self.dir, self.first)
            # Aspect-preserving thumbnail: the same threshold for every camera.
            small = cv2.resize(image, (160, max(1, round(160 * image.shape[0] / image.shape[1]))),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
            if self.last is not None and a.min_mad > 0 and np.abs(small - self.last).mean() < a.min_mad:
                return
            self.last = small
            cv2.imwrite(str(self.dir / ("%06d.%s" % (self.first + self.kept, a.format))), image)
            self.kept += 1
            if self.kept % 50 == 0:
                rate = self.kept / max(1e-6, time.time() - self.start)
                rospy.loginfo("kept %d / seen %d  (%.1f kept/s)", self.kept, self.seen, rate)

    def wait(self, what):
        rospy.loginfo("Recording %s -> %s (target %d frames)", what, self.args.out, self.args.max_frames)
        while not rospy.is_shutdown() and not self.done:
            rospy.sleep(0.1)
        rospy.loginfo("Done: kept %d of %d seen, in %s", self.kept, self.seen, self.dir)


def add_args(ap):
    """Output arguments shared with capture_public_stereo.py."""
    ap.add_argument("--out", required=True, help="dataset root; a session folder is created inside it")
    ap.add_argument("--session", default="", help="session folder name; default <eyeW>x<eyeH>_<timestamp>")
    ap.add_argument("--max_frames", type=int, default=10000)
    ap.add_argument("--min_mad", type=float, default=2.0,
                    help="minimum mean-abs-difference vs the last kept frame; 0 keeps everything")
    ap.add_argument("--format", default="png", choices=["png", "jpg"], help="png is lossless (preferred)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="/ikan/explore3d/depth_image")
    add_args(ap)
    args = ap.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("stereo_capture", anonymous=True)
    if "raw" in args.topic:
        rospy.logwarn("'%s' looks unrectified: training needs rectified pairs (capture_public_stereo.py "
                      "rectifies separate camera topics)", args.topic)
    rec = Recorder(args)

    def callback(msg):
        if rec.done:
            return
        try:
            left, right = cb.decode(msg, 1)[:2]
        except ValueError as e:
            rospy.logwarn_throttle(10, "skipping a message: %s", e)
            return
        rec.add(left, right)

    cb.AnyTopic(args.topic, callback)
    rec.wait(args.topic)


if __name__ == "__main__":
    main()
