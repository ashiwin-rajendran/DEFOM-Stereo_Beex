#!/usr/bin/env python3
"""Receive DEFOM depth from the SVC (defom_svc) over UDP and publish it on ROS.

    python defom_receiver.py                        # ROS: /ikan/explore3d/defom/{depth,points,...}
    python defom_receiver.py  --no_ros              # no ROS: print rate and timing only
    python defom_receiver.py  --no_ros --save out   # also save the first frames (depth .npy, left .jpg)

Topics under --ns (same names as the PC's defom_stereo_pointcloud node, so
existing RViz configs work; do not run both at once):
    depth             sensor_msgs/Image 32FC1, metres, NaN = invalid, 1600x1200
    camera_info       sensor_msgs/CameraInfo, rectified intrinsics at the same size
    left/compressed   sensor_msgs/CompressedImage, rectified left JPEG, pixel-aligned with depth
    points            sensor_msgs/PointCloud2 XYZRGB (only built while someone subscribes)
    depth/compressed  JET-coloured depth JPEG for viewing (only while someone subscribes)

Message stamps: the SVC's clock is not synchronised, so each frame is stamped
with its arrival time here minus its age on the SVC (capture -> sent).

Runs on Python 2.7 (ROS Melodic: `python2 defom_receiver.py`) and Python 3
(ROS Noetic). Needs numpy (+ rospy and cv2 for ROS). Each frame is ~8 MB:
raise the receive buffer once (distillation/README.md, section 6.4) or frames will be dropped.
"""

from __future__ import division, print_function

import argparse
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

try:
    import queue
except ImportError:              # Python 2
    import Queue as queue


def say(text):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()

# Must match defom_svc.cpp (little-endian, packed).
PACKET = struct.Struct("<IHHIIIIII")                        # 32 bytes
PACKET_MAGIC = 0x504D4644                                   # "DFMP"
FRAME = struct.Struct("<IHHIIIfffffIIIIIIIIIIifffQQ16x")    # 128 bytes
FRAME_MAGIC = 0x464D4644                                    # "DFMF"
FRAME_FIELDS = ("magic", "version", "header_bytes", "frame_id", "width", "height", "fx", "fy", "cx", "cy",
                "baseline_m", "depth_format", "depth_offset", "depth_bytes", "image_format", "image_offset",
                "image_bytes", "rectified", "infer_width", "infer_height", "age_us", "lr_skew_us", "prep_ms",
                "infer_ms", "post_ms", "capture_mono_ns", "sent_wall_ns")
assert PACKET.size == 32 and FRAME.size == 128


class Reassembler:
    """Collects datagrams into frames; a frame is delivered once every chunk is in."""

    def __init__(self):
        self.partial = {}          # frame_id -> [buffer, chunks seen (bytearray), count, first arrival]
        self.complete = self.dropped = self.packets = self.bad = 0

    def add(self, view, nbytes, now):
        self.packets += 1
        if nbytes < PACKET.size:
            self.bad += 1
            return None
        magic, version, hbytes, fid, index, count, offset, total, _ = PACKET.unpack_from(view)
        if magic != PACKET_MAGIC or version != 1 or index >= count or offset + nbytes - hbytes > total:
            self.bad += 1
            return None
        entry = self.partial.get(fid)
        if entry is None or len(entry[0]) != total:
            entry = self.partial[fid] = [bytearray(total), bytearray(count), 0, now]
        if not entry[1][index]:
            entry[1][index] = 1
            entry[2] += 1
            entry[0][offset:offset + nbytes - hbytes] = view[hbytes:nbytes]
        if entry[2] < count:
            return None
        del self.partial[fid]
        self.complete += 1
        # Older unfinished frames will not complete now (frame ids restart with the SVC).
        for old in [k for k, v in self.partial.items() if 0 < (fid - k) % 2**32 < 1000 or now - v[3] > 2.0]:
            del self.partial[old]
            self.dropped += 1
        return bytes(entry[0]), entry[3]


def parse_frame(blob):
    h = dict(zip(FRAME_FIELDS, FRAME.unpack_from(blob)))
    if h["magic"] != FRAME_MAGIC or h["version"] != 1 or h["depth_format"] != 1 or h["image_format"] != 1:
        raise ValueError("unknown frame format (magic/version mismatch with defom_svc)")
    depth = np.frombuffer(blob, np.float32, h["width"] * h["height"], h["depth_offset"])
    jpeg = blob[h["image_offset"]:h["image_offset"] + h["image_bytes"]]
    return h, depth.reshape(h["height"], h["width"]), jpeg


def open_socket(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf_mb << 20)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)
    got = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF) // 2   # Linux reports double
    say("listening on %s:%d, receive buffer %.1f MB" % (args.bind, args.port, got / 2**20))
    if got < 8 << 20:
        say("WARNING: receive buffer below 8 MB -- frames will be dropped. Once, as root:\n"
            "  sudo sysctl -w net.core.rmem_max=%d   (distillation/README.md 6.4: make it permanent)" % (args.rcvbuf_mb << 20))
    return sock


def receive_loop(sock, out_queue, stats, stop):
    """Own thread: nothing else may delay reading the socket."""
    buf = bytearray(65536)
    view = memoryview(buf)
    asm = stats["asm"]
    while not stop.is_set():
        try:
            n = sock.recv_into(buf)
        except socket.timeout:
            continue
        done = asm.add(view, n, time.time())
        if done is not None:
            if out_queue.full():
                try:
                    out_queue.get_nowait()
                    stats["skipped"] += 1        # the publisher fell behind: keep the newest
                except queue.Empty:
                    pass
            out_queue.put(done)


class RosPublisher:
    def __init__(self, args):
        import cv2
        import rospy
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
        from std_msgs.msg import Header
        self.cv2, self.rospy, self.Header = cv2, rospy, Header
        self.Image, self.CameraInfo, self.CompressedImage = Image, CameraInfo, CompressedImage
        self.PointCloud2, self.PointField = PointCloud2, PointField
        rospy.init_node(args.node_name, anonymous=False, disable_signals=True)
        ns = args.ns.rstrip("/")
        self.args = args
        self.pub_depth = rospy.Publisher(ns + "/depth", Image, queue_size=1)
        self.pub_info = rospy.Publisher(ns + "/camera_info", CameraInfo, queue_size=1)
        self.pub_left = rospy.Publisher(ns + "/left/compressed", CompressedImage, queue_size=1)
        self.pub_points = rospy.Publisher(ns + "/points", PointCloud2, queue_size=1)
        self.pub_vis = rospy.Publisher(ns + "/depth/compressed", CompressedImage, queue_size=1)
        self.grid = None
        say("publishing %s/{depth,camera_info,left/compressed,points,depth/compressed} in frame '%s'"
            % (ns, args.frame_id))

    def publish(self, h, depth, jpeg, stamp):
        a = self.args
        header = self.Header(stamp=self.rospy.Time.from_sec(stamp), frame_id=a.frame_id)
        W, H = h["width"], h["height"]
        if a.min_depth > 0 or a.max_depth > 0:
            depth = depth.copy()
            if a.min_depth > 0:
                depth[depth < a.min_depth] = np.nan
            if a.max_depth > 0:
                depth[depth > a.max_depth] = np.nan

        info = self.CameraInfo(header=header, width=W, height=H, distortion_model="plumb_bob")
        info.D = [0.0] * 5
        info.K = [h["fx"], 0, h["cx"], 0, h["fy"], h["cy"], 0, 0, 1]
        info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        info.P = [h["fx"], 0, h["cx"], 0, 0, h["fy"], h["cy"], 0, 0, 0, 1, 0]
        self.pub_info.publish(info)
        self.pub_depth.publish(self.Image(header=header, height=H, width=W, encoding="32FC1", is_bigendian=0,
                                          step=W * 4, data=depth.tobytes()))
        self.pub_left.publish(self.CompressedImage(header=header, format="jpeg", data=jpeg))

        if self.pub_points.get_num_connections() > 0:
            self.pub_points.publish(self.points(h, depth, jpeg, header))
        if self.pub_vis.get_num_connections() > 0:
            lo = a.min_depth if a.min_depth > 0 else float(np.nanpercentile(depth, 2))
            hi = a.max_depth if a.max_depth > 0 else float(np.nanpercentile(depth, 98))
            valid = np.isfinite(depth)
            u8 = np.clip((np.where(valid, depth, lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            vis = self.cv2.applyColorMap(u8, self.cv2.COLORMAP_JET)
            vis[~valid] = 0
            data = self.cv2.imencode(".jpg", vis, [self.cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            self.pub_vis.publish(self.CompressedImage(header=header, format="jpeg", data=data))

    def points(self, h, depth, jpeg, header):
        s = max(1, self.args.stride)
        z = depth[::s, ::s]
        key = (h["width"], h["height"], s, h["fx"], h["fy"], h["cx"], h["cy"])
        if self.grid is None or self.grid[0] != key:
            v, u = np.mgrid[0:h["height"]:s, 0:h["width"]:s].astype(np.float32)
            self.grid = (key, (u - h["cx"]) / h["fx"], (v - h["cy"]) / h["fy"])
        _, xf, yf = self.grid
        valid = np.isfinite(z) & (z > 0)
        cloud = np.empty(int(valid.sum()), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
        cloud["x"], cloud["y"], cloud["z"] = (xf * z)[valid], (yf * z)[valid], z[valid]
        bgr = self.cv2.imdecode(np.frombuffer(jpeg, np.uint8), self.cv2.IMREAD_COLOR)[::s, ::s][valid]
        rgb = (bgr[:, 2].astype(np.uint32) << 16) | (bgr[:, 1].astype(np.uint32) << 8) | bgr[:, 0].astype(np.uint32)
        cloud["rgb"] = rgb.view(np.float32)
        F = self.PointField
        msg = self.PointCloud2(header=header, height=1, width=len(cloud), is_bigendian=False, is_dense=True,
                               point_step=16, row_step=16 * len(cloud), data=cloud.tobytes())
        msg.fields = [F(name=n, offset=4 * i, datatype=F.FLOAT32, count=1) for i, n in enumerate("x y z rgb".split())]
        return msg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5610)
    ap.add_argument("--bind", default="0.0.0.0", help="local address to listen on")
    ap.add_argument("--rcvbuf_mb", type=int, default=64, help="socket receive buffer to request")
    ap.add_argument("--no_ros", action="store_true", help="only receive and report")
    ap.add_argument("--save", default="", help="folder: save the first --save_frames frames")
    ap.add_argument("--save_frames", type=int, default=3)
    ap.add_argument("--ns", default="/ikan/explore3d/defom")
    ap.add_argument("--frame_id", default="explore3d_left_optical",
                    help="TF frame of the left camera's OPTICAL axes (x right, y down, z forward)")
    ap.add_argument("--node_name", default="defom_receiver")
    ap.add_argument("--min_depth", type=float, default=0.0, help="metres; 0 = no limit (depth and points)")
    ap.add_argument("--max_depth", type=float, default=0.0, help="metres; 0 = no limit (depth and points)")
    ap.add_argument("--stride", type=int, default=2, help="point cloud: every Nth pixel")
    ap.add_argument("--report_s", type=float, default=5.0)
    argv = args_without_ros()
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    publisher = None if args.no_ros else RosPublisher(args)
    sock = open_socket(args)
    frames = queue.Queue(maxsize=1)
    stats = {"asm": Reassembler(), "skipped": 0}
    stop = threading.Event()
    rx = threading.Thread(target=receive_loop, args=(sock, frames, stats, stop))
    rx.daemon = True
    rx.start()

    save = args.save
    if save and not os.path.isdir(save):
        os.makedirs(save)
    saved, n, t_report = 0, 0, time.time()
    sums = dict(age=0.0, net=0.0, pub=0.0, infer=0.0, skew=0.0, valid=0.0)
    last = None
    try:
        while publisher is None or not publisher.rospy.is_shutdown():
            try:
                blob, first_arrival = frames.get(timeout=0.5)
            except queue.Empty:
                blob = None
            if blob is not None:
                t0 = time.time()
                h, depth, jpeg = parse_frame(blob)
                stamp = first_arrival - h["age_us"] * 1e-6
                if publisher is not None:
                    publisher.publish(h, depth, jpeg, stamp)
                if save and saved < args.save_frames:
                    np.save(os.path.join(save, "frame%06d_depth.npy" % h["frame_id"]), depth)
                    with open(os.path.join(save, "frame%06d_left.jpg" % h["frame_id"]), "wb") as f:
                        f.write(jpeg)
                    saved += 1
                n += 1
                sums["age"] += h["age_us"] / 1000
                sums["net"] += 1000 * (t0 - first_arrival)
                sums["pub"] += 1000 * (time.time() - t0)
                sums["infer"] += h["infer_ms"]
                sums["skew"] += abs(h["lr_skew_us"]) / 1000
                sums["valid"] += float(np.isfinite(depth).mean())
                last = h
            if time.time() - t_report >= args.report_s:
                asm, dt = stats["asm"], time.time() - t_report
                if n and last is not None:
                    say("%.2f Hz | %dx%d, rectified %s, valid %.0f%% | SVC: capture->sent %.0f ms, infer %.0f ms, "
                          "L/R skew %.1f ms | network+reassembly %.0f ms, publish %.0f ms | complete %d, "
                          "dropped %d, skipped %d, bad %d"
                          % (n / dt, last["width"], last["height"], "yes" if last["rectified"] else "no",
                             100 * sums["valid"] / n, sums["age"] / n, sums["infer"] / n, sums["skew"] / n, sums["net"] / n,
                             sums["pub"] / n, asm.complete, asm.dropped, stats["skipped"], asm.bad))
                else:
                    say("no frames in the last %.0f s (%d datagrams) -- is defom_svc running and sending here?"
                        % (dt, asm.packets))
                n, t_report = 0, time.time()
                sums = dict.fromkeys(sums, 0.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if publisher is not None:
            publisher.rospy.signal_shutdown("exit")


def args_without_ros():
    """rospy.myargv() strips ROS remapping arguments when launched by roslaunch."""
    try:
        import rospy
        return rospy.myargv(argv=sys.argv)[1:]
    except ImportError:
        return None


if __name__ == "__main__":
    main()
