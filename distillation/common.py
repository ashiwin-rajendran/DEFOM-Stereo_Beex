"""Shared by the training, labelling and evaluation scripts: config, stereo frames,
the DEFOM ViT-L teacher, checkpoint and log writing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(Path(__file__).resolve().parent), str(ROOT / "core"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import collections
import copy
import csv
import os
import random

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from core.defom_stereo import DEFOMStereo

# DEFOM-Stereo's architecture, as released (every defomstereo_*.pth; only the encoder differs).
DEFOM_ARCH = {
    "dinov2_encoder": "vitl",
    "idepth_scale": 0.5,
    "hidden_dims": [128, 128, 128],
    "corr_implementation": "reg",
    "corr_levels": 2,
    "corr_radius": 4,
    "scale_list": [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "scale_corr_radius": 2,
    "n_downsample": 2,
    "context_norm": "batch",
    "n_gru_layers": 3,
    "shared_backbone": False,
    "mixed_precision": False,
    "valid_iters": 32,
    "scale_iters": 8,
}

PATH_KEYS = ("image_dir", "probe_dir", "labels_dir", "teacher_ckpt", "release_ckpt", "student_ckpt",
             "checkpoint_dir", "init_from", "resume_from")


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def default_config():
    """Defaults every config starts from; the YAML overrides them."""
    return {
        "run_name": "",
        "description": "",
        "paths": dict({k: "" for k in PATH_KEYS},
                      image_dir="distillation/1_datasets/underwater",
                      teacher_ckpt="checkpoints/defomstereo_vitl_middlebury.pth"),
        "training": {
            "seed": 17,
            "num_workers": 4,
            "val_fraction": 0.05,
            "val_split": "temporal",          # temporal | random (random leaks near-duplicate frames)
        },
        "data": {
            "swap_lr": False,          # true if the right eye is in the left half
            "photometric": True,       # training: brightness/contrast jitter, the same on both eyes
            "brightness": 0.2,
            "contrast": 0.2,
        },
        "arch": copy.deepcopy(DEFOM_ARCH),
    }


def deep_update(base, extra):
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    cfg = default_config()
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            deep_update(cfg, yaml.safe_load(handle) or {})
    if cfg["training"]["val_split"] not in ("temporal", "random"):
        raise SystemExit("training.val_split must be 'temporal' or 'random'")
    return cfg


def resolve_paths(cfg):
    """A copy of cfg with relative paths resolved (cwd if they exist there, else
    the repo root). Checkpoints keep the config as written, so stay portable."""
    cfg = copy.deepcopy(cfg)
    for key in PATH_KEYS:
        value = cfg["paths"].get(key, "")
        if value and not Path(value).is_absolute():
            cfg["paths"][key] = str(Path(value).resolve() if Path(value).exists() else ROOT / value)
    return cfg


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

# OpenCV's own thread pool deadlocks or thrashes inside DataLoader workers.
cv2.setNumThreads(0)


class StereoPairDataset(Dataset):
    """Side-by-side stereo frames split into (left, right) RGB float tensors, 0-255, each eye
    resized (aspect kept) to long_side. Training crops and augmentation are LabelledPairs'
    (train_slim.py); this is the whole frame, as deployed."""

    EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, paths, cfg, long_side):
        self.paths = list(paths)
        self.long_side = long_side
        self.swap_lr = bool(cfg["data"]["swap_lr"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Unreadable image: %s" % self.paths[index])
        half = image.shape[1] // 2
        left, right = image[:, :half], image[:, half:]
        if self.swap_lr:
            left, right = right, left
        h, w = left.shape[:2]
        scale = self.long_side / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        interp = cv2.INTER_AREA if size[0] < w else cv2.INTER_LINEAR
        to_t = lambda a: torch.from_numpy(cv2.cvtColor(cv2.resize(a, size, interpolation=interp),
                                                       cv2.COLOR_BGR2RGB).astype(np.float32)).permute(2, 0, 1).contiguous()
        return to_t(left), to_t(right)


def discover_images(image_dir):
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError("image_dir not found: %s" % root)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in StereoPairDataset.EXTS)
    if not paths:
        raise FileNotFoundError("No images under %s" % root)
    return paths


def eye_size(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Unreadable image: %s" % path)
    return image.shape[1] // 2, image.shape[0]


def split_sessions(paths, cfg):
    """Per-session train/val split, and each frame's source (its camera's
    per-eye size). 'temporal' holds out the tail of each session: consecutive
    frames are near-duplicates, so a random split leaks into validation."""
    sessions = collections.OrderedDict()
    for path in paths:
        sessions.setdefault(path.parent, []).append(path)
    frac = cfg["training"]["val_fraction"]
    temporal = cfg["training"]["val_split"] == "temporal"
    rng = random.Random(cfg["training"]["seed"])
    train_paths, val_paths, source_of = [], [], {}
    for folder, items in sessions.items():
        source = "%dx%d" % eye_size(items[0])
        n_val = int(len(items) * frac)
        held = items[len(items) - n_val:] if temporal else rng.sample(items, n_val)
        held_set = set(held)
        train_paths += [p for p in items if p not in held_set]
        val_paths += held
        for p in items:
            source_of[p] = source
        print("  session %-40s %-10s %5d frames  (%d val)" % (folder.name, source, len(items), len(held)))
    if not val_paths:
        # Every session too short to hold anything out: use the largest one's last frame.
        items = max(sessions.values(), key=len)
        val_paths = items[-1:]
        train_paths = [p for p in train_paths if p != items[-1]]
    return train_paths, val_paths, source_of


def subsample(items, n):
    """At most n items, evenly spread."""
    return items[::max(1, len(items) // n)][:n]


# --------------------------------------------------------------------------
# teacher, checkpoints, log
# --------------------------------------------------------------------------


def build_teacher(cfg, device):
    """The frozen DEFOM teacher (paths.teacher_ckpt, default ViT-L Middlebury)."""
    model = DEFOMStereo(argparse.Namespace(**cfg["arch"]))
    checkpoint = torch.load(cfg["paths"]["teacher_ckpt"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def atomic_save(obj, path):
    """torch.save via a temporary file, so a crash never leaves a truncated checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


def append_log(checkpoint_dir, record):
    """Append one epoch to training_log.csv; if the columns changed (e.g. a
    resume that validates other cameras), rewrite it with their union."""
    path = Path(checkpoint_dir) / "training_log.csv"
    rows, fields = [], list(record.keys())
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            old = list(reader.fieldnames or [])
        if old == fields:
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(record)
            return
        fields = old + [k for k in fields if k not in old]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows + [record])
