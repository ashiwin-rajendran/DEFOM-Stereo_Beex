#!/usr/bin/env python3
"""Is the raw 1600x1200 stereo stream rectified? Measured on the SVC.

    python3 check_rectification.py --dwecal e3d1008.dwecal            # live from /dev/video0 + /dev/video2
    python3 check_rectification.py --dwecal e3d1008.dwecal --images pairs/*.png   # saved side-by-side frames

Grabs left/right pairs, matches SIFT features between the eyes and measures
their VERTICAL offset -- zero everywhere for a rectified pair -- on the raw
frames and again after applying the .dwecal rectification maps.

Run it UNDERWATER (sea or tank): the DWE calibration is for the cameras in
water, and refraction at the port changes focal length and distortion, so
in-air frames are not rectified by it even when it is correct. Point the
cameras at a still, textured scene 1-3 m away (the cameras are not
hardware-synchronised) and stop DWE's stream first: only one program can
stream a camera. Needs only python3, numpy and cv2 (already on the SVC).

Writes to --out: raw and rectified pairs (PNG), report.json, and lines.jpg
(rectified pairs with horizontal guide lines, for a visual check).
"""

import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np


def load_dwecal(path):
    """DWECal v1.x: 'DWECAL', version (3 B), n_cameras (1 B), resolution code (4 B),
    fx fy cx cy (4 float32), then the left and right maps: H x W x (x, y) float32
    each, giving the raw pixel each rectified pixel samples (cv2.remap's format)."""
    blob = Path(path).read_bytes()
    if blob[:6] != b"DWECAL":
        raise SystemExit("%s is not a DWECal file" % path)
    fx, fy, cx, cy = np.frombuffer(blob[14:30], "<f4")
    n = (len(blob) - 30) // 4
    for w, h in ((1600, 1200), (800, 600), (1280, 720), (640, 480)):
        if n == 2 * h * w * 2:
            maps = np.frombuffer(blob[30:], "<f4").reshape(2, h, w, 2)
            return (w, h), (float(fx), float(fy), float(cx), float(cy)), maps
    raise SystemExit("Unrecognised map size in %s (%d floats)" % (path, n))


def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    if not cap.isOpened() or not ok:
        raise SystemExit("Cannot stream %s -- is DWE's stream (or another program) using it?" % device)
    if frame.shape[:2] != (height, width):
        raise SystemExit("%s gives %dx%d, not %dx%d" % (device, frame.shape[1], frame.shape[0], width, height))
    return cap


def camera_serial(device):
    node = Path("/sys/class/video4linux") / Path(device).name / "device"
    for parent in [node] + list(node.resolve().parents)[:3]:
        serial = parent / "serial"
        if serial.is_file():
            return serial.read_text().strip()
    return "?"


def vertical_offsets(left, right, sift):
    """Matched features: (x, y) in the left image, and dx, dy to the right image."""
    gl, gr = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in (left, right))
    kl, dl = sift.detectAndCompute(gl, None)
    kr, dr = sift.detectAndCompute(gr, None)
    if dl is None or dr is None or len(kl) < 10 or len(kr) < 10:
        return np.zeros((0, 4))
    pairs = cv2.BFMatcher().knnMatch(dl, dr, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.7 * n.distance]
    pts = np.array([(*kl[m.queryIdx].pt, kl[m.queryIdx].pt[0] - kr[m.trainIdx].pt[0],
                     kl[m.queryIdx].pt[1] - kr[m.trainIdx].pt[1]) for m in good]).reshape(-1, 4)
    # Drop gross mismatches: implausible vertical offsets or negative disparity.
    return pts[(np.abs(pts[:, 3]) < 40) & (pts[:, 2] > -5)]


def summarise(pts, width, height):
    if len(pts) < 20:
        return {"matches": int(len(pts))}
    dy = pts[:, 3]
    # dy = a + b*x + c*y: a residual rotation/scale between the eyes shows up as a slope.
    A = np.c_[np.ones(len(pts)), pts[:, 0] / width, pts[:, 1] / height]
    a, b, c = np.linalg.lstsq(A, dy, rcond=None)[0]
    return {"matches": int(len(pts)), "dy_median_abs_px": float(np.median(np.abs(dy))),
            "dy_p90_abs_px": float(np.percentile(np.abs(dy), 90)), "dy_mean_px": float(dy.mean()),
            "dy_trend_across_width_px": float(b), "dy_trend_across_height_px": float(c),
            "disparity_median_px": float(np.median(pts[:, 2]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dwecal", required=True)
    ap.add_argument("--left", default="/dev/video0")
    ap.add_argument("--right", default="/dev/video2")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between captured pairs")
    ap.add_argument("--images", nargs="*", default=[], help="saved side-by-side frames instead of cameras")
    ap.add_argument("--out", default="rectification_check")
    args = ap.parse_args()

    (W, H), (fx, fy, cx, cy), maps = load_dwecal(args.dwecal)
    map_l = [np.ascontiguousarray(maps[0][..., k]) for k in (0, 1)]
    map_r = [np.ascontiguousarray(maps[1][..., k]) for k in (0, 1)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("dwecal: %dx%d, rectified fx %.3f fy %.3f cx %.3f cy %.3f" % (W, H, fx, fy, cx, cy))

    pairs = []
    if args.images:
        for f in sorted(p for g in args.images for p in glob.glob(g)):
            im = cv2.imread(f)
            if im is not None and im.shape[:2] == (H, 2 * W):
                pairs.append((Path(f).stem, im[:, :W], im[:, W:]))
    else:
        print("left %s (serial %s), right %s (serial %s)" % (args.left, camera_serial(args.left),
                                                             args.right, camera_serial(args.right)))
        caps = [open_camera(d, W, H, 15) for d in (args.left, args.right)]
        for k in range(args.pairs):
            for _ in range(3):                        # flush stale buffers
                for c in caps:
                    c.grab()
            t0 = time.monotonic()
            for c in caps:
                c.grab()
            skew_ms = 1000 * (time.monotonic() - t0)
            frames = [c.retrieve()[1] for c in caps]
            pairs.append(("pair%02d" % k, frames[0], frames[1]))
            cv2.imwrite(str(out / ("raw_%02d.png" % k)), np.hstack(frames))
            print("  captured pair %d (grab spread %.1f ms)" % (k, skew_ms))
            time.sleep(args.interval)
        for c in caps:
            c.release()
    if not pairs:
        raise SystemExit("No %dx%d pairs" % (W, H))

    sift = cv2.SIFT_create(4000)
    raw_pts, rect_pts, guide = [], [], []
    for name, left, right in pairs:
        rl = cv2.remap(left, map_l[0], map_l[1], cv2.INTER_LINEAR)
        rr = cv2.remap(right, map_r[0], map_r[1], cv2.INTER_LINEAR)
        cv2.imwrite(str(out / ("rectified_%s.png" % name)), np.hstack([rl, rr]))
        p_raw, p_rect = vertical_offsets(left, right, sift), vertical_offsets(rl, rr, sift)
        raw_pts.append(p_raw)
        rect_pts.append(p_rect)
        r, q = summarise(p_raw, W, H), summarise(p_rect, W, H)
        print("  %-8s raw: %4d matches, |dy| median %s | rectified: %4d matches, |dy| median %s"
              % (name, r["matches"], "%.2f px" % r["dy_median_abs_px"] if "dy_median_abs_px" in r else "-",
                 q["matches"], "%.2f px" % q["dy_median_abs_px"] if "dy_median_abs_px" in q else "-"))
        if len(guide) < 3:
            g = np.hstack([rl, rr]).copy()
            for y in range(0, H, 60):
                cv2.line(g, (0, y), (2 * W, y), (0, 255, 0), 1)
            guide.append(cv2.resize(g, None, fx=0.4, fy=0.4))

    raw = summarise(np.concatenate(raw_pts), W, H)
    rect = summarise(np.concatenate(rect_pts), W, H)
    report = {"dwecal": str(args.dwecal), "size": [W, H], "rectified_intrinsics": [fx, fy, cx, cy],
              "pairs": len(pairs), "raw": raw, "rectified_with_dwecal": rect}

    def verdict(s):
        if "dy_median_abs_px" not in s:
            return "not enough matches (need a textured scene)"
        flat = abs(s["dy_trend_across_width_px"]) < 1.0 and abs(s["dy_trend_across_height_px"]) < 1.0
        return "RECTIFIED" if s["dy_median_abs_px"] < 0.5 and flat else "NOT rectified"

    report["verdict_raw"], report["verdict_rectified"] = verdict(raw), verdict(rect)
    for key, label in (("raw", "raw stream"), ("rectified_with_dwecal", "after .dwecal maps")):
        s = report[key]
        if "dy_median_abs_px" in s:
            print("%-19s %5d matches | vertical offset |dy| median %.2f px, p90 %.2f px | trend across "
                  "width %+.2f px, height %+.2f px | disparity median %.1f px"
                  % (label, s["matches"], s["dy_median_abs_px"], s["dy_p90_abs_px"],
                     s["dy_trend_across_width_px"], s["dy_trend_across_height_px"], s["disparity_median_px"]))
    print("VERDICT raw stream: %s | with .dwecal: %s" % (report["verdict_raw"], report["verdict_rectified"]))
    if "disparity_median_px" in rect and rect["disparity_median_px"] < 0:
        print("NOTE: negative disparity -- the left/right devices look swapped")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    cv2.imwrite(str(out / "lines.jpg"), np.vstack(guide))
    print("wrote %s/report.json, lines.jpg and the pairs" % out)


if __name__ == "__main__":
    main()
