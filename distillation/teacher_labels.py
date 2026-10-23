#!/usr/bin/env python3
"""Precompute the full DEFOM teacher's disparity for every training frame.

    python3 distillation/teacher_labels.py --config distillation/configs/slim_stage2.yaml

The slim stereo core (train_slim.py) learns to reproduce these. Each frame is
resized, aspect kept, to --long_side (default: the config's data.label_long_side,
800 -- the deployment scale) and the teacher's left disparity is saved as a
16-bit PNG holding disparity x 64 (1/64 px resolution up to 1023 px, lossless):

    <labels_dir>/<session>/<frame stem>.png      + <labels_dir>/labels.json

Resumable: frames whose label exists are skipped. Run it once per dataset (and
again after adding sessions).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from core.utils.utils import InputPadder
from distill_defom import StereoPairDataset, build_teacher, discover_images, load_config, resolve_paths

LABEL_SCALE = 64.0          # stored value = disparity x 64
MAX_DISPARITY = 65535 / LABEL_SCALE


def label_path(labels_dir, image_dir, frame):
    # abspath, not resolve(): sessions may be symlinks into another folder.
    rel = Path(os.path.abspath(frame)).relative_to(os.path.abspath(image_dir))
    return Path(labels_dir) / rel.with_suffix(".png")


def read_label(path):
    """Disparity (float32, px at the label scale) or None; 0 = invalid."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if raw is None else raw.astype(np.float32) / LABEL_SCALE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--long_side", type=int, default=0, help="default: data.label_long_side (800)")
    ap.add_argument("--iters", type=int, default=0, help="teacher iterations; default arch.valid_iters (32)")
    ap.add_argument("--scale_iters", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (testing)")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config))
    image_dir = cfg["paths"]["image_dir"]
    labels_dir = Path(cfg["paths"]["labels_dir"])
    long_side = args.long_side or cfg["data"]["label_long_side"]
    iters = args.iters or cfg["arch"]["valid_iters"]
    scale_iters = args.scale_iters or cfg["arch"]["scale_iters"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    frames = discover_images(image_dir)
    if args.limit:
        frames = frames[:args.limit]
    todo = [f for f in frames if not label_path(labels_dir, image_dir, f).is_file()]
    print("%d frames, %d already labelled, %d to do -> %s (long side %d, %d/%d iterations)"
          % (len(frames), len(frames) - len(todo), len(todo), labels_dir, long_side, iters, scale_iters))
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta = {"teacher_ckpt": cfg["paths"]["teacher_ckpt"], "long_side": long_side, "iters": iters,
            "scale_iters": scale_iters, "encoding": "uint16 PNG, disparity x %g, 0 = invalid" % LABEL_SCALE}
    meta_path = labels_dir / "labels.json"
    if meta_path.is_file():
        old = json.loads(meta_path.read_text())
        if {k: old.get(k) for k in meta} != meta:
            raise SystemExit("%s holds labels made with different settings %s; use another labels_dir"
                             % (labels_dir, old))
    meta_path.write_text(json.dumps(meta, indent=2))
    if not todo:
        return

    teacher = build_teacher(cfg, device)
    data = StereoPairDataset(todo, cfg, long_side=long_side)
    started, done, skipped = time.time(), 0, 0
    for index in range(len(data)):
        try:
            left, right = data[index]
        except RuntimeError as exc:
            print("  skip %s (%s)" % (todo[index], exc))
            skipped += 1
            continue
        left, right = left[None].to(device), right[None].to(device)
        padder = InputPadder(left.shape, divis_by=32)
        try:
            with torch.no_grad():
                disp = padder.unpad(teacher(*padder.pad(left, right), iters=iters, scale_iters=scale_iters,
                                            test_mode=True))[0, 0].float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("  skip %s: CUDA OOM" % todo[index].name)
            skipped += 1
            continue
        valid = np.isfinite(disp) & (disp > 0) & (disp < MAX_DISPARITY)
        out = np.where(valid, np.round(disp * LABEL_SCALE), 0).astype(np.uint16)
        path = label_path(labels_dir, image_dir, todo[index])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(path) + ".tmp.png"
        cv2.imwrite(tmp, out)
        os.replace(tmp, str(path))
        done += 1
        if done % 50 == 0 or done == len(todo):
            rate = (time.time() - started) / done
            print("  %d/%d  (%.2f s/frame, %.0f min left)" % (done, len(todo), rate, rate * (len(todo) - done) / 60),
                  flush=True)
    print("done: %d labelled, %d skipped" % (done, skipped))


if __name__ == "__main__":
    main()
