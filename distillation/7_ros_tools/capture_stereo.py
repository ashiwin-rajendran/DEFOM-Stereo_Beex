#!/usr/bin/env python3
"""Record side-by-side stereo frames from a ROS topic (live or a rosbag) into a training session.

    rosbag play <dive>.beex
    python3 distillation/7_ros_tools/capture_stereo.py --out distillation/1_datasets/underwater          # DWE
    python3 distillation/7_ros_tools/capture_stereo.py --out distillation/1_datasets/underwater \\
        --topic /ikan/voyis/stereo_concatenated/compressed --session voyis_dive3                          # Voyis

Any stereo message compare_beex_bags.py reads works (decoded with the definition the bag sends):
bx_msgs/DepthImage (DWE /ikan/explore3d/depth_image), side-by-side CompressedImage / Image
(Voyis stereo_concatenated), bx_msgs/StereoImage. Frames must already be rectified.

Each run writes <out>/<session>/NNNNNN.png (left | right); the session defaults to
<eyeW>x<eyeH>_<time>, and reusing a session name appends after its existing frames.
Near-duplicates (mean abs difference below --min_mad on a thumbnail) are skipped, so record while
moving.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_beex_bags as cb  # noqa: E402  (system OpenCV, rospy, AnyTopic, decode)
from compare_beex_bags import cv2, np, rospy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="/ikan/explore3d/depth_image")
    ap.add_argument("--out", required=True, help="dataset root; a session folder is created inside it")
    ap.add_argument("--session", default="", help="session folder name; default <eyeW>x<eyeH>_<timestamp>")
    ap.add_argument("--max_frames", type=int, default=4000)
    ap.add_argument("--min_mad", type=float, default=2.0,
                    help="minimum mean-abs-difference vs the last kept frame; 0 keeps everything")
    ap.add_argument("--format", default="png", choices=["png", "jpg"], help="png is lossless (preferred)")
    args = ap.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("stereo_capture", anonymous=True)
    if "raw" in args.topic:
        rospy.logwarn("'%s' looks unrectified: training needs rectified pairs", args.topic)

    state = {"kept": 0, "seen": 0, "last": None, "start": time.time(), "dir": None, "first": 0}

    def callback(msg):
        if state["kept"] >= args.max_frames:
            return
        state["seen"] += 1
        try:
            left, right = cb.decode(msg, 1)[:2]
        except ValueError as e:
            rospy.logwarn_throttle(10, "skipping a message: %s", e)
            return
        image = np.hstack([left, right])
        if state["dir"] is None:              # the session is named after the first frame's per-eye size
            session = args.session or "%dx%d_%s" % (left.shape[1], left.shape[0], time.strftime("%Y%m%d_%H%M%S"))
            state["dir"] = Path(args.out) / session
            state["dir"].mkdir(parents=True, exist_ok=True)
            existing = [int(p.stem) for p in state["dir"].iterdir() if p.stem.isdigit()]
            state["first"] = max(existing) + 1 if existing else 0      # never overwrite: continue after the last
            rospy.loginfo("Writing to %s (from frame %d)", state["dir"], state["first"])
        # Aspect-preserving thumbnail: the same threshold for every camera.
        small = cv2.resize(image, (160, max(1, round(160 * image.shape[0] / image.shape[1]))),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        if state["last"] is not None and args.min_mad > 0 and np.abs(small - state["last"]).mean() < args.min_mad:
            return
        state["last"] = small
        cv2.imwrite(str(state["dir"] / ("%06d.%s" % (state["first"] + state["kept"], args.format))), image)
        state["kept"] += 1
        if state["kept"] % 50 == 0:
            rate = state["kept"] / max(1e-6, time.time() - state["start"])
            rospy.loginfo("kept %d / seen %d  (%.1f kept/s)", state["kept"], state["seen"], rate)

    cb.AnyTopic(args.topic, callback)
    rospy.loginfo("Recording %s -> %s (target %d frames)", args.topic, args.out, args.max_frames)
    while not rospy.is_shutdown() and state["kept"] < args.max_frames:
        rospy.sleep(0.1)
    rospy.loginfo("Done: kept %d of %d seen, in %s", state["kept"], state["seen"], state["dir"])


if __name__ == "__main__":
    main()
