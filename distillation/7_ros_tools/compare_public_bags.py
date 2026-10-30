#!/usr/bin/env python3
"""Compare DEFOM models on public stereo bags: separate left / right camera topics, raw (unrectified).

    rosbag play --loop cemetery.bag
    python3 compare_public_bags.py --show

--left / --right: sensor_msgs/CompressedImage (or Image) per camera, default /cam_fl/image_raw/compressed
and /cam_fr/image_raw/compressed. Frames are paired by header stamp (closest within --max_dt): the
cameras need not run at the same rate (cemetery.bag: left 29.6 Hz, right 15.3 Hz, ~3 ms apart).

Rectification from each camera's CameraInfo (K, D, R, P; read once at start from <topic>/cam_info,
or --left_info / --right_info), at --width (default 800, i.e. 800x600 for 1600x1200 cameras).
Depth = fx * baseline / disparity with the rectified fx and baseline = -P_right[0,3] / fx.
--no_rectify for bags whose images are already rectified (then --fx --cx --cy --baseline are needed).

Same models, keys and filters as compare_beex_bags.py (this node reuses its engine):
    keys 1-9 / n / p in the window (--show), or
    rostopic pub -1 /defom_compare_public/select std_msgs/String "vits_s2_release_512"
    rostopic pub -1 /defom_compare_public/range std_msgs/String "data: '0 3'"     # metres; 'off' clears
    rostopic pub -1 /defom_compare_public/texture std_msgs/String "data: '6'"     # 'off' clears
(write values as "data: '...'": rostopic reads a bare 6 as a number and off as false)

Publishes (node defom_compare_public, so it runs alongside compare_beex_bags.py's defom_compare):
    ~compare          left (rectified) | model depth (near = red)
    ~rectified        left | right rectified, with horizontal lines (check the rectification)
    ~depth            model depth, 32FC1 metres (NaN = invalid)
    ~depth/color      model depth, coloured
    ~points           model point cloud, XYZRGB, in the left camera's frame
    ~camera_info      rectified left camera
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_beex_bags as dc  # noqa: E402  (also sets up the system OpenCV, rospy)
from compare_beex_bags import cv2, rospy  # noqa: E402
from sensor_msgs.msg import CameraInfo, CompressedImage, Image  # noqa: E402
from std_msgs.msg import Header  # noqa: E402


def decode_image(msg):
    """CompressedImage or Image -> BGR."""
    if hasattr(msg, "format"):
        return dc.jpeg(msg.data)
    return dc.raw_image(msg)


class Rectifier:
    """Undistort + rectify both cameras from their CameraInfo, straight into the output size."""

    def __init__(self, left_info, right_info, width):
        self.src = (left_info.width, left_info.height)
        s = width / float(left_info.width)
        self.size = (int(round(left_info.width * s)), int(round(left_info.height * s)))
        self.maps = [self.make_map(i, s) for i in (left_info, right_info)]
        P = np.array(right_info.P).reshape(3, 4)
        self.fx, self.fy = P[0, 0] * s, P[1, 1] * s
        self.cx, self.cy = P[0, 2] * s, P[1, 2] * s
        self.baseline = -P[0, 3] / P[0, 0]
        if self.baseline <= 0:
            raise SystemExit("right CameraInfo P[0,3] = %g: expected negative (-fx * baseline); are --left and "
                             "--right swapped?" % P[0, 3])

    def make_map(self, info, s):
        # The images are first shrunk by s (INTER_AREA, no aliasing), so K and P are scaled to match.
        K = np.array(info.K).reshape(3, 3).copy()
        K[:2] *= s
        P = np.array(info.P).reshape(3, 4)[:, :3].copy()
        P[:2] *= s
        return cv2.initUndistortRectifyMap(K, np.array(info.D), np.array(info.R).reshape(3, 3), P, self.size,
                                           cv2.CV_32FC1)

    def __call__(self, img, side):
        if (img.shape[1], img.shape[0]) != self.src:
            raise ValueError("image is %dx%d, CameraInfo says %dx%d" % (img.shape[1], img.shape[0], *self.src))
        small = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        return cv2.remap(small, *self.maps[side], cv2.INTER_LINEAR)


class PublicStereo(dc.Compare):
    """compare_beex_bags.py's engine (models, keys, filters, outputs, loop); input from two camera topics."""

    NODE = "defom_compare_public"

    def __init__(self, args):     # not Compare.__init__: no DWE/Voyis input or reference depth
        self.setup(args)
        self.rect = None
        if args.no_rectify:
            missing = [k for k in ("fx", "cx", "cy", "baseline") if getattr(args, k) is None]
            if missing:
                raise SystemExit("--no_rectify needs --%s" % " --".join(missing))
        else:
            infos = []
            for topic in (args.left_info or args.left.rsplit("/", 1)[0] + "/cam_info",
                          args.right_info or args.right.rsplit("/", 1)[0] + "/cam_info"):
                rospy.loginfo("waiting for %s (rosbag playing?)", topic)
                infos.append(rospy.wait_for_message(topic, CameraInfo))
            self.rect = Rectifier(infos[0], infos[1], args.width)
            for k in ("fx", "fy", "cx", "cy", "baseline"):
                if getattr(args, k) is not None:
                    setattr(self.rect, k, getattr(args, k))
            rospy.loginfo("rectified %dx%d -> %dx%d: fx %.3f, cx %.2f, cy %.2f, baseline %.4f m",
                          *self.rect.src, *self.rect.size, self.rect.fx, self.rect.cx, self.rect.cy, self.rect.baseline)
        self.advertise(("rectified", Image))
        self.buf = (collections.deque(maxlen=10), collections.deque(maxlen=10))   # (stamp, msg) per side
        self.paired = None                                                          # stamp of the last right used
        for side, topic in enumerate((args.left, args.right)):
            typ = CompressedImage if topic.endswith("compressed") else Image
            rospy.Subscriber(topic, typ, self.on_image, callback_args=side, queue_size=4, buff_size=2 ** 26)
        rospy.loginfo("waiting for %s and %s", args.left, args.right)

    def on_image(self, msg, side):
        """Pair the newest right frame with the closest left one (the right camera is the slower here)."""
        with self.lock:
            self.buf[side].append((msg.header.stamp.to_sec(), msg))
            if not self.buf[0] or not self.buf[1]:
                return
            tr, right = self.buf[1][-1]
            if tr == self.paired:
                return
            tl, left = min(self.buf[0], key=lambda b: abs(b[0] - tr))
            if abs(tl - tr) > self.args.max_dt:
                if side == 0 and tl > tr:      # lefts have moved past this right: it has no partner
                    rospy.logwarn_throttle(10, "no left frame within %.3f s of the right one (closest %.3f s); "
                                           "raise --max_dt?", self.args.max_dt, abs(tl - tr))
                return
            self.paired = tr
        self.submit((left, right, tl - tr))

    def process(self, pair):
        a = self.args
        left_msg, right_msg, dt = pair
        left, right = decode_image(left_msg), decode_image(right_msg)
        if self.rect is not None:
            left, right = self.rect(left, 0), self.rect(right, 1)
            K, baseline = (self.rect.fx, self.rect.fy, self.rect.cx, self.rect.cy), self.rect.baseline
        else:
            if left.shape != right.shape:
                raise ValueError("left %s and right %s differ in size" % (left.shape, right.shape))
            K, baseline = (a.fx, a.fy or a.fx, a.cx, a.cy), a.baseline
        depth, ms, size = self.estimate(left, right, K[0], baseline)
        valid = np.isfinite(depth)
        note = "pair dt %+.0f ms, depth %s" % (1000 * dt, "%.2f..%.2f m (median %.2f)" % (
            np.nanpercentile(depth, 2), np.nanpercentile(depth, 98), np.nanmedian(depth)) if valid.any() else "none")
        self.stats.append((ms, note, size))

        header = Header(stamp=left_msg.header.stamp, frame_id=a.frame_id or left_msg.header.frame_id or "camera")
        mc = self.publish(header, left, depth, K)
        if self.wanted("rectified"):
            both = np.hstack([left, right])
            for y in range(20, both.shape[0], 40):
                cv2.line(both, (0, y), (both.shape[1], y), (0, 0, 255), 1)
            self.pub["rectified"].publish(dc.image_msg(header, both, "bgr8"))
        return self.compose(header, [(left, "left (rectified)")], mc, ms, note, size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="/cam_fl/image_raw/compressed")
    ap.add_argument("--right", default="/cam_fr/image_raw/compressed")
    ap.add_argument("--left_info", default="", help="CameraInfo topic (default: <left topic's parent>/cam_info)")
    ap.add_argument("--right_info", default="", help="CameraInfo topic (default: <right topic's parent>/cam_info)")
    ap.add_argument("--max_dt", type=float, default=0.02, help="largest left/right stamp difference to pair, s")
    ap.add_argument("--width", type=int, default=800, help="rectified image width (the model's input at most)")
    ap.add_argument("--no_rectify", action="store_true", help="images are already rectified (give --fx --cx --cy "
                    "--baseline, at the image size)")
    for k in ("fx", "fy", "cx", "cy"):
        ap.add_argument("--" + k, type=float, default=None, help="override, rectified, at --width")
    ap.add_argument("--baseline", type=float, default=None, help="override, metres")
    ap.add_argument("--frame_id", default="", help="default: the left image's")
    dc.add_common_args(ap)
    args = dc.parse_args(ap)
    rospy.init_node(PublicStereo.NODE)
    PublicStereo(args).spin()


if __name__ == "__main__":
    main()
