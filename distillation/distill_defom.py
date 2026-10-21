#!/usr/bin/env python3
"""Distil DEFOM-Stereo's DepthAnythingV2 encoder into a small student.

    python3 distillation/distill_defom.py --config distillation/configs/underwater_vitl.yaml

WHAT IS AND IS NOT BEING DISTILLED
-----------------------------------
Measured parameter budget of defomstereo_vitl_*:

    DINOv2-L backbone  1217.5 MB   frozen
    depth_head          123.8 MB   frozen
    depth_feat          122.5 MB   trained by DEFOM
    fnet/cnet/GRU        66.8 MB   trained by DEFOM   <- KEPT, NOT DISTILLED
    -------------------------------------------------
    total              1530.5 MB

95.6% of the model is the encoder, so that is the only part worth replacing.
The 66.8 MB stereo core is reused verbatim from the vitl checkpoint, which is
why the student only has to reproduce the encoder's output tuple rather than
learn stereo matching from scratch. Result: 1530.5 MB -> ~75.6 MB (20.2x).

For reference, the official vits checkpoint is 173.2 MB with no training at
all, and on this footage it tracks vitl closely. So the honest target here is
roughly 2.3x beyond vits, plus the freedom to pick the student's size.

NO LABELS ARE NEEDED. The student regresses the teacher's outputs on
unlabelled stereo pairs, so any footage from the deployment domain works.

LOSS
----
Five teacher tensors are matched. Feature maps use the same reduction as the
DINOv3 distillation in Yolo_Seg/vit_training: sum over the channel dim, mean
over batch and space. 'mean' over everything would dilute a large error in a
few channels across the 255 that already agree; raw 'sum' makes the gradient
scale with resolution and forces an LR retune whenever the input size changes.

    feature loss = mean_{B,H,W} [ sum_C (s - t)^2 ]

idepth is a single channel and is matched with L1, which is less sensitive to
the sparse large outliers the monocular prior produces at depth discontinuities.
It is matched RAW: DefomEncoder normalises by its own per-sample max downstream,
so an extra normalisation here would discard the scale the teacher actually
emits and leave the student free to drift.

RESOLUTION
----------
Training is camera- and resolution-agnostic. Frames may come from any stereo
camera at any size (DWE SVC 800x600, DWE raw 1600x1200, Voyis 2816x2816 per
eye) and can be mixed under one image_dir, one folder per capture session.
Each eye is resized, aspect preserved, to a random long side in
data.long_side, then both eyes take the same random crop of data.crop. The
deployed ROS node resizes whole frames to its export resolution, so this is
the range of content scales the student actually meets. No intrinsics are
involved anywhere here: the loss is label-free and the encoder runs on each
eye independently. Calibration belongs to rectification (before capture) and
to depth computation (in the ROS node), nowhere else.

VALIDATION
----------
Feature loss is a proxy. What matters is end-to-end disparity, so validation
temporarily swaps the student into the real DEFOM model and compares its
disparity against the unmodified teacher on held-out frames -- the same
metric demo_v3.py reports live (mean/median/p90 px, % within 1/2/3 px).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import collections
import copy
import csv
import json
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import yaml
except ImportError:
    yaml = None

from core.defom_stereo import DEFOMStereo
from core.utils.utils import InputPadder, get_danv2_io_size
from defom_student import build_student, count_parameters


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def default_config():
    return {
        "run_name": "defom_distill",
        "description": "",
        "paths": {
            "image_dir": "",
            "teacher_ckpt": "checkpoints/defomstereo_vitl_sceneflow.pth",
            "checkpoint_dir": "distillation/checkpoints",
            "init_from": "",
            "resume_from": "",
        },
        "training": {
            "epochs": 20,
            "batch_size": 2,
            "lr": 3.0e-4,
            "min_lr": 1.0e-6,
            "weight_decay": 1.0e-4,
            "grad_clip": 1.0,
            "num_workers": 4,
            "seed": 17,
            "checkpoint_every": 1,
            "val_fraction": 0.05,
            # temporal holds out the tail; random shuffles. See train() for why
            # random leaks on sequential survey footage.
            "val_split": "temporal",
            "val_every": 1,
            "val_max_frames": 24,
            "amp": True,
            # Teacher batches used to measure output mean/std before training.
            "calibration_batches": 8,
            # Batches used to re-estimate the student's BatchNorm statistics
            # before they are frozen for training. 0 keeps the loaded ones.
            "bn_recalibration_batches": 50,
            # Run the teacher in bfloat16: ~1.7x faster teacher (~35% faster
            # epochs on a 4070) for a 1-3% perturbation of the targets. Off by
            # default; ignored on GPUs without bf16 (e.g. Xavier).
            "teacher_bf16": False,
            # TF32 tensor cores for fp32 matmuls (teacher, validation).
            # Measured on a 4070: teacher 450 -> 334 ms with target error
            # unchanged (0.18% -> 0.19% vs strict fp32; PyTorch already uses
            # TF32 for convolutions by default). Needs Ampere or newer.
            "tf32": True,
        },
        "student": {
            "backbone": "mbconv",          # mbconv | mobilevit
            "width": 1.0,                  # mbconv only
            "fpn_dim": 128,
            "depth_dim": 64,
            "mobilevit_variant": "s",      # s | xs | xxs
            "mobilevit_init_from": "",     # optional warm start from a student_mobilevit_*.pth
            # group | batch. group is default: the teacher caps batch size
            # near 2-4, where BatchNorm running stats do not stabilise.
            "norm": "group",
            # Keep BatchNorm (inside the MobileViT backbone) in eval mode
            # while training, so train and eval run the same function.
            "freeze_bn": True,
        },
        "data": {
            # Per-eye long side is drawn from this range, aspect preserved,
            # then both eyes take the same crop (h, w; multiples of 32).
            "long_side": [512, 1024],
            "crop": [384, 512],
            # Equal sampling weight per per-eye resolution, so one camera's
            # footage cannot drown another's -- capped so no frame is drawn
            # more than about max_repeat times per epoch.
            "balance_sources": True,
            "max_repeat": 4,
            # End-to-end validation runs on whole frames at these long sides.
            "val_long_side": [512, 640],
            "swap_lr": False,
            "photometric": True,
            "brightness": 0.2,
            "contrast": 0.2,
            "hflip_pair": True,
        },
        "loss": {
            "w_dfeat": [1.0, 1.0, 1.0],
            "w_left": 1.0,
            "w_right": 1.0,
            "w_idepth": 1.0,
            "feature_metric": "relative_mse",
        },
        "arch": {
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
        },
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
        if yaml is None:
            raise ImportError("pyyaml is required to read a config file")
        with open(path, "r", encoding="utf-8") as handle:
            deep_update(cfg, yaml.safe_load(handle) or {})
    stale = [k for k in ("height", "width") if k in cfg["data"]]
    if stale:
        raise SystemExit("data.%s is no longer used: training is resolution-agnostic. "
                         "Use data.long_side and data.crop instead." % "/".join(stale))
    return cfg


def arch_namespace(cfg):
    arch = cfg["arch"]
    return argparse.Namespace(
        dinov2_encoder=arch["dinov2_encoder"], idepth_scale=arch["idepth_scale"],
        hidden_dims=list(arch["hidden_dims"]), corr_implementation=arch["corr_implementation"],
        corr_levels=arch["corr_levels"], corr_radius=arch["corr_radius"],
        scale_list=list(arch["scale_list"]), scale_corr_radius=arch["scale_corr_radius"],
        n_downsample=arch["n_downsample"], context_norm=arch["context_norm"],
        n_gru_layers=arch["n_gru_layers"], shared_backbone=arch["shared_backbone"],
        mixed_precision=arch["mixed_precision"],
    )


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


# OpenCV spawns its own thread pool, which deadlocks or thrashes inside
# DataLoader worker processes. One line, and it must come before any worker
# forks.
cv2.setNumThreads(0)


class StereoPairDataset(Dataset):
    """Side-by-side frames on disk, split into a left/right pair.

    With long_side=None (training) each eye is resized, aspect preserved, to a
    random long side in data.long_side and both take the same random crop of
    data.crop. A fixed resize instead pins the student to one resolution: the
    student trained at 800x600 was measured at feature rel-MSE 0.11 against
    the teacher at 800x600, but 0.30-0.47 at 512x384 and 1024x768. With a
    long_side given (validation) the whole frame is resized to it, uncropped
    and unaugmented -- what deployment runs.

    Geometric augmentation is deliberately restricted. Anything that breaks
    the epipolar relationship between the two views -- an independent crop,
    rotation, or a flip of only one image -- changes the disparity the teacher
    would predict, so the student would be chasing a target that no longer
    corresponds to its input. Horizontal flip is applied to BOTH views with
    the pair order swapped, which is the one geometric transform that keeps a
    rectified pair valid. Photometric jitter is applied identically to both.
    """

    EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, paths, cfg, long_side=None):
        self.paths = list(paths)
        self.cfg = cfg
        train = long_side is None
        d = cfg["data"]
        self.long_side = tuple(d["long_side"]) if train else (long_side, long_side)
        self.crop = tuple(int(v) for v in d["crop"]) if train else None
        self.swap_lr = bool(d["swap_lr"])
        self.photometric = bool(d["photometric"]) and train
        self.brightness = float(d["brightness"])
        self.contrast = float(d["contrast"])
        self.hflip_pair = bool(d["hflip_pair"]) and train

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
        scale = random.uniform(*self.long_side) / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        if self.crop:
            size = (max(size[0], self.crop[1]), max(size[1], self.crop[0]))
        interp = cv2.INTER_AREA if size[0] < w else cv2.INTER_LINEAR
        left = cv2.resize(left, size, interpolation=interp)
        right = cv2.resize(right, size, interpolation=interp)
        if self.crop:
            ch, cw = self.crop
            y, x = random.randint(0, size[1] - ch), random.randint(0, size[0] - cw)
            # The same window in both views: a common shift leaves every
            # disparity of the rectified pair unchanged.
            left, right = left[y:y + ch, x:x + cw], right[y:y + ch, x:x + cw]

        if self.hflip_pair and random.random() < 0.5:
            # Mirroring a rectified pair swaps which camera is which.
            left, right = cv2.flip(right, 1), cv2.flip(left, 1)

        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB).astype(np.float32)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.photometric:
            beta = random.uniform(-self.brightness, self.brightness) * 255.0
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            left = np.clip(left * alpha + beta, 0, 255)
            right = np.clip(right * alpha + beta, 0, 255)

        to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1).contiguous()
        return to_t(left), to_t(right)


def eye_size(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Unreadable image: %s" % path)
    return image.shape[1] // 2, image.shape[0]


def split_sessions(paths, cfg):
    """Train/val split per capture session, plus each frame's source.

    A session is one folder of consecutive frames from one camera; its source
    is that camera's per-eye size, read from its first frame. temporal holds
    out the tail of EACH session -- consecutive survey frames are near
    duplicates, so a random split puts near-copies of training frames into
    validation and reports a number that says nothing about generalisation.
    """
    sessions = collections.OrderedDict()
    for path in paths:
        sessions.setdefault(path.parent, []).append(path)
    frac = cfg["training"]["val_fraction"]
    split = cfg["training"]["val_split"]
    if split not in ("temporal", "random"):
        raise ValueError("training.val_split must be 'temporal' or 'random', got %r" % split)
    rng = random.Random(cfg["training"]["seed"])
    train_paths, val_paths, source_of = [], [], {}
    for folder, items in sessions.items():
        source = "%dx%d" % eye_size(items[0])
        n_val = int(len(items) * frac)
        held = items[len(items) - n_val:] if split == "temporal" else rng.sample(items, n_val)
        held_set = set(held)
        train_paths += [p for p in items if p not in held_set]
        val_paths += held
        for p in items:
            source_of[p] = source
        print("  session %-40s %-10s %5d frames  (%d val)" % (folder.name, source, len(items), len(held)))
    if not val_paths:
        # Every session shorter than 1/val_fraction: hold out the largest one's tail.
        items = max(sessions.values(), key=len)
        val_paths = items[-1:]
        train_paths = [p for p in train_paths if p != items[-1]]
    return train_paths, val_paths, source_of


def source_shares(counts, total, max_repeat):
    """Fraction of each epoch drawn from each source: equal, except that no
    frame is drawn more than max_repeat times per epoch.

    A plain equal share would draw a 1-frame source half of every epoch
    (measured: one frame ~2700 times). Sources that would exceed the cap are
    fixed at it, and what they leave is shared equally among the rest.
    """
    shares, remaining, budget = {}, set(counts), 1.0
    while remaining:
        equal = budget / len(remaining)
        capped = {s for s in remaining if equal * total / counts[s] > max_repeat}
        if not capped:
            shares.update({s: equal for s in remaining})
            break
        for s in capped:
            shares[s] = max_repeat * counts[s] / total
            budget -= shares[s]
        remaining -= capped
    return shares


def discover_images(image_dir):
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError("image_dir not found: %s" % root)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in StereoPairDataset.EXTS)
    if not paths:
        raise FileNotFoundError("No images under %s" % root)
    return paths


# --------------------------------------------------------------------------
# teacher / student plumbing
# --------------------------------------------------------------------------


def build_teacher(cfg, device):
    model = DEFOMStereo(arch_namespace(cfg))
    ckpt_path = Path(cfg["paths"]["teacher_ckpt"])
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def preprocess(model, left, right, n_downsample):
    """Reproduce DEFOM's own encoder preprocessing exactly.

    Any divergence here -- different normalisation constants, a different
    resize -- means the student is trained on inputs the deployed model never
    produces, and the distillation silently targets the wrong function.
    """
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    left = ((left - model.mean) / model.std).contiguous()
    right = ((right - model.mean) / model.std).contiguous()
    height, width = left.shape[-2:]
    ih, iw, oh, ow = get_danv2_io_size(height, width, n_downsample)
    x = torch.cat([left, right], dim=0)
    x = F.interpolate(x, (ih, iw), mode="bilinear", align_corners=True)
    return x, oh, ow, padder


def feature_loss(student, teacher, metric):
    """Per-map distillation loss.

    `relative_mse` is the default, and the reason is measured rather than
    stylistic. The teacher's five outputs span five orders of magnitude:

        d_features[0]  std     93.7
        d_features[1]  std    281.2
        d_features[2]  std    121.9
        left/right     std  20931.1      <-- absmax 272296
        idepth         std     87.9

    Under a plain summed MSE the left/right term evaluates around 1.1e11 while
    the d_features terms sit near 1e6, so with equal weights the student would
    optimise the matching features alone and effectively ignore the rest.
    Dividing each term by the teacher's own energy makes every term a
    dimensionless relative error of order 1, so the configured weights mean
    what they appear to mean and do not need retuning per checkpoint.
    """
    if metric == "cosine":
        return (1.0 - F.cosine_similarity(student, teacher, dim=1)).mean()
    numerator = ((student - teacher) ** 2).sum(dim=1).mean()
    if metric == "mse":
        # sum over channels, mean over batch and space -- raw scale
        return numerator
    denominator = (teacher.float() ** 2).sum(dim=1).mean().clamp_min(1e-6)
    return numerator / denominator


def distill_loss(student_out, teacher_out, cfg):
    weights = cfg["loss"]
    metric = weights["feature_metric"]
    s_feats, s_left, s_right, s_idepth = student_out
    t_feats, t_left, t_right, t_idepth = teacher_out

    parts = {}
    total = 0.0
    for i, (sf, tf) in enumerate(zip(s_feats, t_feats)):
        w = weights["w_dfeat"][i]
        value = feature_loss(sf, tf, metric)
        parts["dfeat%d" % i] = float(value.detach())
        total = total + w * value
    for name, w, sv, tv in (("left", weights["w_left"], s_left, t_left),
                            ("right", weights["w_right"], s_right, t_right)):
        value = feature_loss(sv, tv, metric)
        parts[name] = float(value.detach())
        total = total + w * value
    # idepth is single-channel, so it needs the same scale treatment: its own
    # magnitude (std ~88) is unrelated to the feature terms'.
    if metric in ("relative_mse", "cosine"):
        idepth = F.l1_loss(s_idepth, t_idepth) / t_idepth.abs().mean().clamp_min(1e-6)
    else:
        idepth = F.l1_loss(s_idepth, t_idepth)
    parts["idepth"] = float(idepth.detach())
    total = total + weights["w_idepth"] * idepth
    parts["total"] = float(total.detach())
    return total, parts


@torch.no_grad()
def calibrate_output_scales(teacher_encoder, student, loader, teacher_model, n_downsample,
                            device, batches=8):
    """Measure the teacher's output mean/std and copy them into the student.

    Without this the student's heads have to climb from O(1) init to targets
    of order 2e4, and the relative-loss gradient (~ -2/t) is far too small to
    get there -- measured, the left/right term moved 0.001% in 8 epochs.
    Calibrating first lets every head learn a standardised O(1) function
    while still emitting teacher-scale tensors.
    """
    sums = {k: [0.0, 0.0, 0] for k in ("rn1", "rn2", "rn3", "fuse", "depth")}
    seen = 0
    for left, right in loader:
        if seen >= batches:
            break
        seen += 1
        left, right = left.to(device), right.to(device)
        x, oh, ow, _ = preprocess(teacher_model, left, right, n_downsample)
        d_feats, left_feat, right_feat, idepth = teacher_encoder(x, oh, ow)
        for key, tensor in (("rn1", d_feats[0]), ("rn2", d_feats[1]), ("rn3", d_feats[2]),
                            ("fuse", torch.cat([left_feat, right_feat], dim=0)),
                            ("depth", idepth)):
            t = tensor.float()
            sums[key][0] += t.mean().item()
            sums[key][1] += t.std().item()
            sums[key][2] += 1
    if seen == 0:
        return
    stats = {k: (v[0] / max(1, v[2]), v[1] / max(1, v[2])) for k, v in sums.items()}
    student.set_output_scales(stats)
    print("Calibrated output scales from %d teacher batches:" % seen)
    for key, (mean, std) in stats.items():
        print("    %-6s mean %12.3f  std %12.3f" % (key, mean, std))


@torch.no_grad()
def recalibrate_batchnorm(student, loader, teacher_model, n_downsample, device, batches):
    """Re-estimate the student's BatchNorm running statistics on the training
    distribution before they are frozen (see DefomStudentEncoder.train).

    Loaded statistics come from another input distribution -- DINOv3
    pre-training, or an earlier fixed-800x600 run -- so they are replaced by a
    cumulative average (momentum=None) over `batches` training batches.
    """
    norms = [m for m in student.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms or batches <= 0:
        return
    momenta = [m.momentum for m in norms]
    student.eval()
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    seen = 0
    for left, right in loader:
        if seen >= batches:
            break
        x, oh, ow, _ = preprocess(teacher_model, left.to(device), right.to(device), n_downsample)
        student(x, oh, ow)
        seen += 1
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum
    student.train()
    print("Re-estimated the statistics of %d BatchNorm layers from %d batches" % (len(norms), seen))


# --------------------------------------------------------------------------
# validation: end-to-end disparity agreement
# --------------------------------------------------------------------------


@torch.no_grad()
def validate_disparity(model, student, paths, cfg, device, long_side, cache):
    """Swap the student in, compare full-model disparity against the teacher.

    Runs whole frames at one long side, as the deployed node does. The
    teacher's disparity on a given frame never changes, so it is computed once
    and kept in `cache` (CPU), halving validation time after the first epoch.

    The swap is done in place and reverted in a finally block so one model
    instance serves both roles -- holding a second DEFOM just for validation
    would cost another 1.5 GB of VRAM for no benefit.
    """
    arch = cfg["arch"]
    data = StereoPairDataset(paths, cfg, long_side=long_side)
    original = model.defomencoder.depth_anything
    stats = []
    was_training = student.training
    student.eval()
    try:
        for index in range(len(data)):
            left, right = data[index]
            left = left[None].to(device)
            right = right[None].to(device)
            padder = InputPadder(left.shape, divis_by=32)
            lp, rp = padder.pad(left, right)

            key = (str(data.paths[index]), long_side)
            if key not in cache:
                model.defomencoder.depth_anything = original
                cache[key] = padder.unpad(model(lp, rp, iters=arch["valid_iters"],
                                                scale_iters=arch["scale_iters"],
                                                test_mode=True)).cpu()
            ref = cache[key].to(device)
            model.defomencoder.depth_anything = student
            got = padder.unpad(model(lp, rp, iters=arch["valid_iters"],
                                     scale_iters=arch["scale_iters"], test_mode=True))

            diff = (got - ref).abs().flatten()
            stats.append((diff.mean().item(), diff.median().item(),
                          torch.quantile(diff.float(), 0.9).item(),
                          (diff < 1).float().mean().item(),
                          (diff < 2).float().mean().item(),
                          (diff < 3).float().mean().item()))
    finally:
        model.defomencoder.depth_anything = original
        if was_training:
            student.train()
    if not stats:
        return {}
    arr = np.array(stats)
    keys = ("mean_px", "median_px", "p90_px", "within1", "within2", "within3")
    return dict(zip(keys, arr.mean(axis=0).tolist()))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def append_log(checkpoint_dir, record):
    path = Path(checkpoint_dir) / "training_log.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def resolve_paths(cfg):
    """A copy of cfg whose relative paths resolve against the working directory
    if they exist there, else against the repository root -- so the trainer
    runs from anywhere. Checkpoints store the config as written instead, so
    their paths stay relative and portable to other machines."""
    cfg = copy.deepcopy(cfg)
    entries = [("paths", k) for k in ("image_dir", "teacher_ckpt", "checkpoint_dir",
                                      "init_from", "resume_from")]
    for section, key in entries + [("student", "mobilevit_init_from")]:
        value = cfg[section][key]
        if value and not Path(value).is_absolute():
            cfg[section][key] = str(Path(value).resolve() if Path(value).exists() else ROOT / value)
    return cfg


def train(cfg):
    portable_cfg = cfg
    # Checkpoints rebuild the student from student_cfg and then load full
    # weights, so a warm-start path there is never needed -- and may not
    # exist on the machine that loads them.
    portable_student_cfg = dict(portable_cfg["student"], mobilevit_init_from="")
    cfg = resolve_paths(cfg)
    torch.manual_seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(portable_cfg, handle, indent=2)

    paths = discover_images(cfg["paths"]["image_dir"])
    print("Dataset sessions:")
    train_paths, val_paths, source_of = split_sessions(paths, cfg)
    counts = collections.Counter(source_of[p] for p in train_paths)
    print("Dataset: %d frames  (%d train / %d val, %s split)  sources %s"
          % (len(paths), len(train_paths), len(val_paths), cfg["training"]["val_split"],
             dict(counts)))

    train_set = StereoPairDataset(train_paths, cfg)
    sampler = None
    if cfg["data"]["balance_sources"] and len(counts) > 1:
        shares = source_shares(counts, len(train_paths), cfg["data"]["max_repeat"])
        print("Sampling share per source: %s" % {s: "%.1f%%" % (100 * v) for s, v in shares.items()})
        weights = [shares[source_of[p]] / counts[source_of[p]] for p in train_paths]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_paths), replacement=True)
    loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=sampler is None,
                        sampler=sampler, num_workers=cfg["training"]["num_workers"], drop_last=True,
                        pin_memory=(device.type == "cuda"))

    # Validation frames per source, evenly spaced through the held-out tails.
    val_sets = collections.OrderedDict()
    for source in sorted(set(source_of[p] for p in val_paths)):
        items = [p for p in val_paths if source_of[p] == source]
        step = max(1, len(items) // max(1, cfg["training"]["val_max_frames"]))
        val_sets[source] = items[::step][:cfg["training"]["val_max_frames"]]
    val_keys = ("mean_px", "median_px", "p90_px", "within1", "within2", "within3")
    teacher_cache = {}
    previous_ckpt = None

    print("Loading teacher: %s" % cfg["paths"]["teacher_ckpt"])
    teacher_model = build_teacher(cfg, device)
    teacher_encoder = teacher_model.defomencoder.depth_anything
    out_dim = teacher_model.defomencoder.out_dim

    sc = cfg["student"]
    student = build_student(features=out_dim, width=sc["width"], fpn_dim=sc["fpn_dim"],
                            depth_dim=sc["depth_dim"], backbone=sc["backbone"],
                            mobilevit_variant=sc["mobilevit_variant"],
                            # A warm start or resume replaces every backbone weight, so the
                            # DINOv3 backbone checkpoint would be loaded only to be overwritten
                            # -- and its absolute path need not exist on another machine.
                            mobilevit_init_from=("" if cfg["paths"]["init_from"] or cfg["paths"]["resume_from"]
                                                 else sc["mobilevit_init_from"]),
                            norm=sc["norm"]).to(device)
    print("Student backbone: %s (norm=%s)" % (sc["backbone"], sc["norm"]))
    n_student = count_parameters(student)
    n_teacher = sum(p.numel() for p in teacher_encoder.parameters())
    print("Teacher encoder : %10d params  %7.1f MB" % (n_teacher, n_teacher * 4 / 1e6))
    print("Student encoder : %10d params  %7.1f MB  (%.1fx smaller)"
          % (n_student, n_student * 4 / 1e6, n_teacher / max(1, n_student)))
    core = sum(p.numel() for n, p in teacher_model.named_parameters() if not n.startswith("defomencoder"))
    print("Reused stereo core: %.1f MB -> deployed model ~%.1f MB"
          % (core * 4 / 1e6, (core + n_student) * 4 / 1e6))

    start_epoch = 0
    if cfg["paths"]["init_from"]:
        state = torch.load(cfg["paths"]["init_from"], map_location=device, weights_only=True)
        student.load_state_dict(state["student"] if "student" in state else state)
        print("Initialised student from %s" % cfg["paths"]["init_from"])

    # Must happen before the optimiser is built and before any resume. A
    # resumed or warm-started student keeps the buffers saved with it: its
    # heads were trained against those statistics, and recalibrating would
    # shift every output. Only a student whose buffers are still at their
    # defaults (shift 0, scale 1) is calibrated.
    uncalibrated = student.fuse_scale.item() == 1.0 and student.fuse_shift.item() == 0.0
    if not cfg["paths"]["resume_from"] and uncalibrated:
        calibrate_output_scales(teacher_encoder, student, loader, teacher_model,
                                cfg["arch"]["n_downsample"], device,
                                batches=cfg["training"]["calibration_batches"])
    if not cfg["paths"]["resume_from"] and student.freeze_bn:
        recalibrate_batchnorm(student, loader, teacher_model, cfg["arch"]["n_downsample"], device,
                              cfg["training"]["bn_recalibration_batches"])

    if cfg["training"]["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    teacher_bf16 = bool(cfg["training"]["teacher_bf16"])
    if teacher_bf16 and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
        print("training.teacher_bf16 ignored: this GPU has no bfloat16 support")
        teacher_bf16 = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg["training"]["epochs"] * max(1, len(loader))),
        eta_min=cfg["training"]["min_lr"])
    amp_enabled = bool(cfg["training"]["amp"]) and device.type == "cuda"
    # torch.amp.GradScaler where available (torch >= 2.3); older JetPack
    # builds only have the torch.cuda.amp one.
    scaler = (torch.amp.GradScaler("cuda", enabled=amp_enabled) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp_enabled))

    if cfg["paths"]["resume_from"]:
        state = torch.load(cfg["paths"]["resume_from"], map_location=device, weights_only=True)
        student.load_state_dict(state["student"])
        start_epoch = int(state.get("epoch", 0)) + 1
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            # The restored schedule spans the ORIGINAL run's length. Resuming
            # with more epochs would run past its end, where the cosine wraps
            # and the learning rate climbs back up. Stretch it to this run's
            # length and set the rate to where that longer schedule would be
            # now -- the scheduler steps recursively from the current rate, so
            # changing T_max alone would leave it stuck at min_lr.
            scheduler.T_max = max(1, cfg["training"]["epochs"] * max(1, len(loader)))
            for group, lr in zip(optimizer.param_groups, scheduler._get_closed_form_lr()):
                group["lr"] = lr
        else:
            # Only the newest checkpoint keeps optimizer state (see below).
            print("No optimizer state in %s: AdamW restarts; the LR schedule is "
                  "fast-forwarded to epoch %d" % (cfg["paths"]["resume_from"], start_epoch))
            optimizer.zero_grad(set_to_none=True)
            for _ in range(start_epoch * len(loader)):
                scheduler.step()
        print("Resumed from %s at epoch %d" % (cfg["paths"]["resume_from"], start_epoch))

    n_downsample = cfg["arch"]["n_downsample"]

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        student.train()
        running = {}
        started = time.time()
        # Time spent waiting for the DataLoader: if it is a large share of
        # s/step, decoding/resizing (CPU) is the bottleneck, not the GPU --
        # raise training.num_workers.
        data_wait = 0.0
        fetched = time.time()
        for step, (left, right) in enumerate(loader):
            data_wait += time.time() - fetched
            left, right = left.to(device, non_blocking=True), right.to(device, non_blocking=True)
            x, oh, ow, _ = preprocess(teacher_model, left, right, n_downsample)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=teacher_bf16):
                teacher_out = teacher_encoder(x, oh, ow)
            if teacher_bf16:
                teacher_out = ([t.float() for t in teacher_out[0]],
                               *(t.float() for t in teacher_out[1:]))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp_enabled):
                student_out = student(x, oh, ow)
                loss, parts = distill_loss(student_out, teacher_out, cfg)

            scaler.scale(loss).backward()
            if cfg["training"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), cfg["training"]["grad_clip"])
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Mixed precision skips the optimizer step (and lowers the scale)
            # when gradients overflow, as in the first few steps; the schedule
            # advances only on steps that actually updated the weights.
            if scaler.get_scale() >= scale:
                scheduler.step()

            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value
            if (step + 1) % 20 == 0:
                n = step + 1
                print("  epoch %d  step %4d/%d  total %.4f  idepth %.4f  lr %.2e  (%.2f s/step, %.2f s waiting for data)"
                      % (epoch, n, len(loader), running["total"] / n, running["idepth"] / n,
                         scheduler.get_last_lr()[0], (time.time() - started) / n, data_wait / n))
            fetched = time.time()

        n = max(1, len(loader))
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                  "seconds": round(time.time() - started, 1)}
        record.update({k: round(v / n, 6) for k, v in running.items()})

        validate = epoch % cfg["training"]["val_every"] == 0
        for source, items in val_sets.items():
            for long_side in cfg["data"]["val_long_side"]:
                # End-to-end validation runs the FULL model twice per frame,
                # which peaks higher than training does. Losing it must not
                # lose the epoch's training, so an OOM here is reported and
                # skipped rather than propagated. Keys are written either way
                # (NaN when skipped) so the CSV columns stay stable.
                metrics = {}
                if validate:
                    try:
                        metrics = validate_disparity(teacher_model, student, items, cfg, device,
                                                     long_side, teacher_cache)
                    except torch.cuda.OutOfMemoryError as exc:
                        torch.cuda.empty_cache()
                        print("  [val %s @%d] skipped -- CUDA OOM (%s). Training is unaffected."
                              % (source, long_side, str(exc).split("\n")[0]))
                prefix = "val_%s_L%d_" % (source, long_side)
                record.update({prefix + k: round(metrics[k], 5) if k in metrics else float("nan")
                               for k in val_keys})
                if metrics:
                    print("  [val vs teacher] %-10s @%4d  mean %.2f px | median %.2f | p90 %.2f | "
                          "within 1/2/3px %.0f%%/%.0f%%/%.0f%%"
                          % (source, long_side, metrics["mean_px"], metrics["median_px"],
                             metrics["p90_px"], 100 * metrics["within1"],
                             100 * metrics["within2"], 100 * metrics["within3"]))

        append_log(ckpt_dir, record)
        print("epoch %d done: total %.4f  (%.1f s)" % (epoch, record.get("total", float("nan")),
                                                       record["seconds"]))

        if (epoch + 1) % cfg["training"]["checkpoint_every"] == 0 or epoch == cfg["training"]["epochs"] - 1:
            out = ckpt_dir / ("student_epoch%04d.pth" % epoch)
            torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "config": portable_cfg, "student_cfg": portable_student_cfg,
                        "out_dim": out_dim}, out)
            print("  saved %s" % out)
            # Optimizer state is ~2x the weights and only needed to resume, so
            # only the newest checkpoint keeps it: ~27 MB per older epoch
            # instead of ~82 MB.
            if previous_ckpt is not None and previous_ckpt.exists():
                state = torch.load(str(previous_ckpt), map_location="cpu", weights_only=True)
                state.pop("optimizer", None)
                state.pop("scheduler", None)
                torch.save(state, str(previous_ckpt))
            previous_ckpt = out

    print("Training complete. Checkpoints in %s" % ckpt_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="", help="YAML config path")
    parser.add_argument("--image_dir", default="", help="override paths.image_dir")
    parser.add_argument("--teacher_ckpt", default="", help="override paths.teacher_ckpt")
    parser.add_argument("--checkpoint_dir", default="", help="override paths.checkpoint_dir")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--width", type=float, default=0.0, help="override student.width")
    parser.add_argument("--backbone", default="", choices=["", "mbconv", "mobilevit"],
                        help="override student.backbone")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.image_dir:
        cfg["paths"]["image_dir"] = args.image_dir
    if args.teacher_ckpt:
        cfg["paths"]["teacher_ckpt"] = args.teacher_ckpt
    if args.checkpoint_dir:
        cfg["paths"]["checkpoint_dir"] = args.checkpoint_dir
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.width:
        cfg["student"]["width"] = args.width
    if args.backbone:
        cfg["student"]["backbone"] = args.backbone
    if not cfg["paths"]["image_dir"]:
        raise SystemExit("Set paths.image_dir in the config, or pass --image_dir")
    train(cfg)


if __name__ == "__main__":
    main()
