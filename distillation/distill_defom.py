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
import csv
import json
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
        },
        "student": {
            "backbone": "mbconv",          # mbconv | mobilevit
            "width": 1.0,                  # mbconv only
            "fpn_dim": 128,
            "depth_dim": 64,
            "mobilevit_variant": "s",      # s | xs | xxs
            "mobilevit_scripts_dir": "",   # blank = auto-locate Yolo_Seg/vit_training/scripts
            "mobilevit_init_from": "",     # optional warm start from a student_mobilevit_*.pth
            # group | batch. group is default: the teacher caps batch size
            # near 2-4, where BatchNorm running stats do not stabilise.
            "norm": "group",
        },
        "data": {
            "height": 600,
            "width": 800,
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

    Geometric augmentation is deliberately restricted. Anything that breaks
    the epipolar relationship between the two views -- an independent crop,
    rotation, or a flip of only one image -- changes the disparity the teacher
    would predict, so the student would be chasing a target that no longer
    corresponds to its input. Horizontal flip is applied to BOTH views with
    the pair order swapped, which is the one geometric transform that keeps a
    rectified pair valid. Photometric jitter is applied identically to both.
    """

    EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, paths, cfg, train=True):
        self.paths = list(paths)
        self.cfg = cfg
        self.train = train
        d = cfg["data"]
        self.height, self.width = int(d["height"]), int(d["width"])
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

        if (left.shape[0], left.shape[1]) != (self.height, self.width):
            left = cv2.resize(left, (self.width, self.height), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (self.width, self.height), interpolation=cv2.INTER_AREA)

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
    ckpt_path = ROOT / cfg["paths"]["teacher_ckpt"] if not Path(cfg["paths"]["teacher_ckpt"]).is_absolute() \
        else Path(cfg["paths"]["teacher_ckpt"])
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
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


# --------------------------------------------------------------------------
# validation: end-to-end disparity agreement
# --------------------------------------------------------------------------


@torch.no_grad()
def validate_disparity(model, student, paths, cfg, device, max_frames):
    """Swap the student in, compare full-model disparity against the teacher.

    The swap is done in place and reverted in a finally block so one model
    instance serves both roles -- holding a second DEFOM just for validation
    would cost another 1.5 GB of VRAM for no benefit.
    """
    arch = cfg["arch"]
    data = StereoPairDataset(paths[:max_frames], cfg, train=False)
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

            model.defomencoder.depth_anything = original
            ref = padder.unpad(model(lp, rp, iters=arch["valid_iters"],
                                     scale_iters=arch["scale_iters"], test_mode=True))
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


def train(cfg):
    torch.manual_seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    paths = discover_images(cfg["paths"]["image_dir"])
    n_val = max(1, int(len(paths) * cfg["training"]["val_fraction"]))
    split = cfg["training"]["val_split"]
    if split == "temporal":
        # Survey footage is sequential and consecutive frames are near
        # duplicates, so a RANDOM split puts near-copies of training frames
        # into validation and reports an optimistic number that says nothing
        # about generalisation. capture_stereo.py names files in capture
        # order, so holding out the tail gives a genuinely unseen segment.
        train_paths, val_paths = paths[:-n_val], paths[-n_val:]
    elif split == "random":
        shuffled = paths[:]
        random.Random(cfg["training"]["seed"]).shuffle(shuffled)
        val_paths, train_paths = shuffled[:n_val], shuffled[n_val:]
    else:
        raise ValueError("training.val_split must be 'temporal' or 'random', got %r" % split)
    print("Dataset: %d frames  (%d train / %d val, %s split)"
          % (len(paths), len(train_paths), len(val_paths), split))

    train_set = StereoPairDataset(train_paths, cfg, train=True)
    loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True,
                        num_workers=cfg["training"]["num_workers"], drop_last=True,
                        pin_memory=(device.type == "cuda"))

    print("Loading teacher: %s" % cfg["paths"]["teacher_ckpt"])
    teacher_model = build_teacher(cfg, device)
    teacher_encoder = teacher_model.defomencoder.depth_anything
    out_dim = teacher_model.defomencoder.out_dim

    sc = cfg["student"]
    student = build_student(features=out_dim, width=sc["width"], fpn_dim=sc["fpn_dim"],
                            depth_dim=sc["depth_dim"], backbone=sc["backbone"],
                            mobilevit_scripts_dir=sc["mobilevit_scripts_dir"] or None,
                            mobilevit_variant=sc["mobilevit_variant"],
                            mobilevit_init_from=sc["mobilevit_init_from"],
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
        state = torch.load(cfg["paths"]["init_from"], map_location=device)
        student.load_state_dict(state["student"] if "student" in state else state)
        print("Initialised student from %s" % cfg["paths"]["init_from"])

    # Must happen before the optimiser is built and before any resume, so a
    # resumed run keeps the buffers that were saved with its checkpoint.
    if not cfg["paths"]["resume_from"]:
        calibrate_output_scales(teacher_encoder, student, loader, teacher_model,
                                cfg["arch"]["n_downsample"], device,
                                batches=cfg["training"]["calibration_batches"])

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg["training"]["epochs"] * max(1, len(loader))),
        eta_min=cfg["training"]["min_lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"]["amp"]) and device.type == "cuda")

    if cfg["paths"]["resume_from"]:
        state = torch.load(cfg["paths"]["resume_from"], map_location=device)
        student.load_state_dict(state["student"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0)) + 1
        print("Resumed from %s at epoch %d" % (cfg["paths"]["resume_from"], start_epoch))

    n_downsample = cfg["arch"]["n_downsample"]
    amp_enabled = bool(cfg["training"]["amp"]) and device.type == "cuda"

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        student.train()
        running = {}
        started = time.time()
        for step, (left, right) in enumerate(loader):
            left, right = left.to(device, non_blocking=True), right.to(device, non_blocking=True)
            x, oh, ow, _ = preprocess(teacher_model, left, right, n_downsample)

            with torch.no_grad():
                teacher_out = teacher_encoder(x, oh, ow)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                student_out = student(x, oh, ow)
                loss, parts = distill_loss(student_out, teacher_out, cfg)

            scaler.scale(loss).backward()
            if cfg["training"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value
            if (step + 1) % 20 == 0:
                n = step + 1
                print("  epoch %d  step %4d/%d  total %.4f  idepth %.4f  lr %.2e  (%.2f s/step)"
                      % (epoch, n, len(loader), running["total"] / n, running["idepth"] / n,
                         scheduler.get_last_lr()[0], (time.time() - started) / n))

        n = max(1, len(loader))
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                  "seconds": round(time.time() - started, 1)}
        record.update({k: round(v / n, 6) for k, v in running.items()})

        if val_paths and (epoch % cfg["training"]["val_every"] == 0):
            # End-to-end validation runs the FULL model twice per frame, which
            # peaks higher than training does. Losing it must not lose the
            # epoch's training, so an OOM here is reported and skipped rather
            # than propagated.
            try:
                metrics = validate_disparity(teacher_model, student, val_paths, cfg, device,
                                             cfg["training"]["val_max_frames"])
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                metrics = {}
                print("  [val] skipped -- CUDA OOM during end-to-end validation (%s). "
                      "Training is unaffected. Free VRAM, lower training.val_max_frames, "
                      "or set training.val_every higher." % str(exc).split("\n")[0])
            if metrics:
                record.update({k: round(v, 5) for k, v in metrics.items()})
                print("  [val vs teacher] mean %.2f px | median %.2f | p90 %.2f | "
                      "within 1/2/3px %.0f%%/%.0f%%/%.0f%%"
                      % (metrics["mean_px"], metrics["median_px"], metrics["p90_px"],
                         100 * metrics["within1"], 100 * metrics["within2"], 100 * metrics["within3"]))

        append_log(ckpt_dir, record)
        print("epoch %d done: total %.4f  (%.1f s)" % (epoch, record.get("total", float("nan")),
                                                       record["seconds"]))

        if (epoch + 1) % cfg["training"]["checkpoint_every"] == 0 or epoch == cfg["training"]["epochs"] - 1:
            out = ckpt_dir / ("student_epoch%04d.pth" % epoch)
            torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "config": cfg, "student_cfg": cfg["student"], "out_dim": out_dim}, out)
            print("  saved %s" % out)

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
