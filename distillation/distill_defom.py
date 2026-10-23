#!/usr/bin/env python3
"""Distil DEFOM-Stereo's DepthAnythingV2 encoder into a small MobileViT student.

    python3 distillation/distill_defom.py --config distillation/configs/underwater_vitl.yaml

The student learns to reproduce the frozen teacher encoder's outputs on
unlabelled stereo frames (any camera, any size); DEFOM's stereo core is reused
unchanged. Each epoch is validated end to end against the teacher's disparity,
and the best epoch is kept as student_best.pth. See README.md (how to run) and
ANALYSIS.md (why it works this way).
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
import hashlib
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from core.defom_stereo import DEFOMStereo
from core.utils.utils import InputPadder, get_danv2_io_size
from defom_student import build_student, count_parameters, resolve_student_checkpoint


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def default_config():
    """Every key a config may set; the YAML overrides these."""
    return {
        "run_name": "defom_distill",
        "description": "",
        "paths": {
            "image_dir": "",
            "teacher_ckpt": "checkpoints/defomstereo_vitl_middlebury.pth",
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
            "val_split": "temporal",          # temporal | random
            "val_every": 1,
            "val_max_frames": 80,
            "validate_start": True,
            "amp": True,
            "calibration_batches": 8,
            "bn_recalibration_batches": 50,
            "teacher_bf16": False,
            "tf32": True,
        },
        "student": {
            "backbone": "mobilevit",
            "mobilevit_variant": "s",
            "fpn_dim": 128,
            "depth_dim": 64,
            "mobilevit_init_from": "",
            "norm": "group",
            "freeze_bn": True,
        },
        "data": {
            "long_side": [512, 1024],
            "crop": [384, 512],
            "balance_sources": True,
            "max_repeat": 4,
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
        with open(path, "r", encoding="utf-8") as handle:
            deep_update(cfg, yaml.safe_load(handle) or {})
    stale = [k for k in ("height", "width") if k in cfg["data"]]
    if stale:
        raise SystemExit("data.%s is no longer used: set data.long_side and data.crop" % "/".join(stale))
    if cfg["loss"].get("feature_metric", "relative_mse") != "relative_mse":
        raise SystemExit("loss.feature_metric: only relative_mse is supported")
    if cfg["training"]["val_split"] not in ("temporal", "random"):
        raise SystemExit("training.val_split must be 'temporal' or 'random'")
    return cfg


def resolve_paths(cfg):
    """A copy of cfg with relative paths resolved (cwd if they exist there, else
    the repo root). Checkpoints keep the config as written, so stay portable."""
    cfg = copy.deepcopy(cfg)
    entries = [("paths", k) for k in ("image_dir", "teacher_ckpt", "checkpoint_dir", "init_from", "resume_from")]
    for section, key in entries + [("student", "mobilevit_init_from")]:
        value = cfg[section][key]
        if value and not Path(value).is_absolute():
            cfg[section][key] = str(Path(value).resolve() if Path(value).exists() else ROOT / value)
    return cfg


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

# OpenCV's own thread pool deadlocks or thrashes inside DataLoader workers.
cv2.setNumThreads(0)


class StereoPairDataset(Dataset):
    """Side-by-side stereo frames split into (left, right) RGB float tensors.

    Training (long_side=None): each eye is resized, aspect kept, to a random
    long side in data.long_side, then both take the SAME random crop -- a
    common shift keeps the pair's disparities valid. The only geometric
    augmentation is a mirror of both eyes with their order swapped; photometric
    jitter is identical for both. Validation (long_side given): the whole frame
    at that long side, unaugmented, as deployed.
    """

    EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, paths, cfg, long_side=None):
        self.paths = list(paths)
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
            left, right = left[y:y + ch, x:x + cw], right[y:y + ch, x:x + cw]

        if self.hflip_pair and random.random() < 0.5:
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


def source_shares(counts, total, max_repeat):
    """Share of each epoch per source: equal, but no frame drawn more than
    max_repeat times per epoch (what capped sources leave is shared out)."""
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


# --------------------------------------------------------------------------
# teacher, loss, calibration
# --------------------------------------------------------------------------


def build_teacher(cfg, device):
    model = DEFOMStereo(argparse.Namespace(**cfg["arch"]))
    checkpoint = torch.load(cfg["paths"]["teacher_ckpt"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def preprocess(model, left, right, n_downsample):
    """DEFOM's own encoder preprocessing, exactly -- any divergence trains the
    student on inputs the deployed model never produces."""
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    left = ((left - model.mean) / model.std).contiguous()
    right = ((right - model.mean) / model.std).contiguous()
    ih, iw, oh, ow = get_danv2_io_size(*left.shape[-2:], n_downsample)
    x = F.interpolate(torch.cat([left, right], dim=0), (ih, iw), mode="bilinear", align_corners=True)
    return x, oh, ow, padder


def feature_loss(student, teacher):
    """Squared error summed over channels, averaged over batch and space,
    divided by the teacher's own energy. The teacher's outputs span five orders
    of magnitude, so this makes every term a relative error of order 1."""
    numerator = ((student - teacher) ** 2).sum(dim=1).mean()
    denominator = (teacher.float() ** 2).sum(dim=1).mean().clamp_min(1e-6)
    return numerator / denominator


def distill_loss(student_out, teacher_out, cfg):
    w = cfg["loss"]
    s_feats, s_left, s_right, s_idepth = student_out
    t_feats, t_left, t_right, t_idepth = teacher_out
    parts, total = {}, 0.0
    terms = [("dfeat%d" % i, w["w_dfeat"][i], sf, tf) for i, (sf, tf) in enumerate(zip(s_feats, t_feats))]
    terms += [("left", w["w_left"], s_left, t_left), ("right", w["w_right"], s_right, t_right)]
    for name, weight, sv, tv in terms:
        value = feature_loss(sv, tv)
        parts[name] = float(value.detach())
        total = total + weight * value
    # Raw inverse depth (DEFOM normalises it downstream), relative L1.
    idepth = F.l1_loss(s_idepth, t_idepth) / t_idepth.abs().mean().clamp_min(1e-6)
    parts["idepth"] = float(idepth.detach())
    total = total + w["w_idepth"] * idepth
    parts["total"] = float(total.detach())
    return total, parts


@torch.no_grad()
def calibrate_output_scales(teacher_encoder, student, loader, teacher_model, n_downsample, device, batches):
    """Measure the teacher's output mean/std and store them in the student
    (see DefomStudentEncoder: its heads learn standardised targets)."""
    sums = {k: [0.0, 0.0] for k in ("rn1", "rn2", "rn3", "fuse", "depth")}
    seen = 0
    for left, right in loader:
        if seen >= batches:
            break
        seen += 1
        x, oh, ow, _ = preprocess(teacher_model, left.to(device), right.to(device), n_downsample)
        d_feats, left_feat, right_feat, idepth = teacher_encoder(x, oh, ow)
        for key, tensor in (("rn1", d_feats[0]), ("rn2", d_feats[1]), ("rn3", d_feats[2]),
                            ("fuse", torch.cat([left_feat, right_feat], dim=0)), ("depth", idepth)):
            sums[key][0] += tensor.float().mean().item()
            sums[key][1] += tensor.float().std().item()
    if seen == 0:
        return
    stats = {k: (v[0] / seen, v[1] / seen) for k, v in sums.items()}
    student.set_output_scales(stats)
    print("Calibrated output scales from %d teacher batches:" % seen)
    for key, (mean, std) in stats.items():
        print("    %-6s mean %12.3f  std %12.3f" % (key, mean, std))


@torch.no_grad()
def recalibrate_batchnorm(student, loader, teacher_model, n_downsample, device, batches):
    """Re-estimate the backbone's BatchNorm statistics on this data (cumulative
    average over `batches`), before they are frozen for training."""
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
# validation: end-to-end disparity agreement with the teacher
# --------------------------------------------------------------------------


@torch.no_grad()
def validate_disparity(model, student, paths, cfg, device, long_side, cache):
    """Swap the student into the teacher model and compare disparities on whole
    frames at one long side. The teacher's disparity per frame is cached (CPU).
    One model instance serves both roles; the swap is always reverted."""
    arch = cfg["arch"]
    data = StereoPairDataset(paths, cfg, long_side=long_side)
    original = model.defomencoder.depth_anything
    stats = []
    was_training = student.training
    student.eval()
    try:
        for index in range(len(data)):
            left, right = data[index]
            padder = InputPadder(left[None].shape, divis_by=32)
            lp, rp = padder.pad(left[None].to(device), right[None].to(device))
            key = (str(data.paths[index]), long_side)
            if key not in cache:
                model.defomencoder.depth_anything = original
                cache[key] = padder.unpad(model(lp, rp, iters=arch["valid_iters"],
                                                scale_iters=arch["scale_iters"], test_mode=True)).cpu()
            model.defomencoder.depth_anything = student
            got = padder.unpad(model(lp, rp, iters=arch["valid_iters"],
                                     scale_iters=arch["scale_iters"], test_mode=True))
            diff = (got - cache[key].to(device)).abs().flatten()
            stats.append((diff.mean().item(), diff.median().item(), torch.quantile(diff.float(), 0.9).item(),
                          (diff < 1).float().mean().item(), (diff < 2).float().mean().item(),
                          (diff < 3).float().mean().item()))
    finally:
        model.defomencoder.depth_anything = original
        if was_training:
            student.train()
    if not stats:
        return {}
    keys = ("mean_px", "median_px", "p90_px", "within1", "within2", "within3")
    return dict(zip(keys, np.array(stats).mean(axis=0).tolist()))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


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


def train(cfg):
    portable_cfg = cfg
    # Checkpoints carry every weight, so the warm-start path is never needed to load them.
    portable_student_cfg = dict(portable_cfg["student"], mobilevit_init_from="")
    cfg = resolve_paths(cfg)
    paths_cfg = cfg["paths"]
    if paths_cfg["resume_from"] and paths_cfg["init_from"]:
        print("resume_from is set: ignoring init_from (%s)" % paths_cfg["init_from"])
        paths_cfg["init_from"] = ""
    # A run folder works too: fine-tune from its best epoch, resume from its newest.
    if paths_cfg["init_from"]:
        paths_cfg["init_from"] = str(resolve_student_checkpoint(paths_cfg["init_from"]))
    if paths_cfg["resume_from"]:
        paths_cfg["resume_from"] = str(resolve_student_checkpoint(paths_cfg["resume_from"], prefer_best=False))
        if Path(paths_cfg["resume_from"]).name == "student_best.pth":
            raise SystemExit("resume_from is a student_best.pth: resume from the run's newest "
                             "student_epoch####.pth (or its folder), or fine-tune from it with init_from.")
    tcfg = cfg["training"]
    torch.manual_seed(tcfg["seed"])
    random.seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tcfg["tf32"]:   # before any validation, so every score is computed the same way
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ckpt_dir = Path(paths_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(portable_cfg, handle, indent=2)

    # ---- data
    paths = discover_images(paths_cfg["image_dir"])
    print("Dataset sessions:")
    train_paths, val_paths, source_of = split_sessions(paths, cfg)
    counts = collections.Counter(source_of[p] for p in train_paths)
    print("Dataset: %d frames  (%d train / %d val, %s split)  sources %s"
          % (len(paths), len(train_paths), len(val_paths), tcfg["val_split"], dict(counts)))
    train_set = StereoPairDataset(train_paths, cfg)
    sampler = None
    if cfg["data"]["balance_sources"] and len(counts) > 1:
        shares = source_shares(counts, len(train_paths), cfg["data"]["max_repeat"])
        print("Sampling share per source: %s" % {s: "%.1f%%" % (100 * v) for s, v in shares.items()})
        weights = [shares[source_of[p]] / counts[source_of[p]] for p in train_paths]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_paths), replacement=True)
    loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=sampler is None, sampler=sampler,
                        num_workers=tcfg["num_workers"], drop_last=True, pin_memory=(device.type == "cuda"))
    if len(loader) == 0:
        raise SystemExit("%d training frames is less than one batch (batch_size %d)"
                         % (len(train_paths), tcfg["batch_size"]))

    # Validation frames per source, evenly spaced through the held-out tails.
    val_sets = collections.OrderedDict()
    for source in sorted(set(source_of[p] for p in val_paths)):
        items = [p for p in val_paths if source_of[p] == source]
        step = max(1, len(items) // max(1, tcfg["val_max_frames"]))
        val_sets[source] = items[::step][:tcfg["val_max_frames"]]
    val_keys = ("mean_px", "median_px", "p90_px", "within1", "within2", "within3")
    teacher_cache = {}

    def validate_all():
        """Every source at every long side; NaN where skipped (OOM never stops
        training). val_score = mean within-3px, each line weighted equally."""
        out = {}
        for source, items in val_sets.items():
            for long_side in cfg["data"]["val_long_side"]:
                metrics = {}
                try:
                    metrics = validate_disparity(teacher_model, student, items, cfg, device, long_side, teacher_cache)
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    print("  [val %s @%d] skipped -- CUDA OOM (%s)" % (source, long_side, str(exc).split("\n")[0]))
                prefix = "val_%s_L%d_" % (source, long_side)
                out.update({prefix + k: round(metrics[k], 5) if k in metrics else float("nan") for k in val_keys})
                if metrics:
                    print("  [val vs teacher] %-10s @%4d  mean %.2f px | median %.2f | p90 %.2f | "
                          "within 1/2/3px %.0f%%/%.0f%%/%.0f%%"
                          % (source, long_side, metrics["mean_px"], metrics["median_px"], metrics["p90_px"],
                             100 * metrics["within1"], 100 * metrics["within2"], 100 * metrics["within3"]))
        scores = [v for k, v in out.items() if k.endswith("_within3") and v == v]
        out["val_score"] = round(sum(scores) / len(scores), 5) if scores else float("nan")
        return out

    # ---- best checkpoint. Scores compare only on the same frames, sizes and
    # iterations, so best.json carries a signature of them.
    signature = hashlib.sha1(json.dumps({
        "frames": sorted(str(p) for items in val_sets.values() for p in items),
        "long_sides": list(cfg["data"]["val_long_side"]),
        "iters": [cfg["arch"]["valid_iters"], cfg["arch"]["scale_iters"]],
    }).encode()).hexdigest()[:12]
    best_json, best_pth = ckpt_dir / "best.json", ckpt_dir / "student_best.pth"
    best = {"score": float("-inf"), "epoch": None}
    if best_json.exists():
        previous = json.loads(best_json.read_text())
        if previous.get("signature") == signature:
            best = previous
            print("Best so far (same validation): %s, score %.2f%%" % (best["epoch"], 100 * best["score"]))
        else:
            print("best.json was scored on different validation frames; tracking afresh")

    def update_best(metrics, epoch_label):
        score = metrics.get("val_score", float("nan"))
        if score != score:
            return
        if score <= best["score"]:
            print("  best remains %s (val score %.2f%%; this one %.2f%%)"
                  % (best["epoch"], 100 * best["score"], 100 * score))
            return
        best.update({"score": score, "epoch": epoch_label, "signature": signature,
                     "metrics": {k: v for k, v in metrics.items() if k.startswith("val_")}})
        atomic_save({"student": student.state_dict(), "epoch": epoch_label, "config": portable_cfg,
                     "student_cfg": portable_student_cfg, "out_dim": out_dim, "val_score": score,
                     "val_signature": signature}, best_pth)
        tmp = Path(str(best_json) + ".tmp")
        tmp.write_text(json.dumps(best, indent=2))
        os.replace(str(tmp), str(best_json))
        print("  new best: %s (val score %.2f%%) -> %s" % (epoch_label, 100 * score, best_pth.name))

    # ---- models
    print("Loading teacher: %s" % paths_cfg["teacher_ckpt"])
    teacher_model = build_teacher(cfg, device)
    teacher_encoder = teacher_model.defomencoder.depth_anything
    out_dim = teacher_model.defomencoder.out_dim
    n_downsample = cfg["arch"]["n_downsample"]

    sc = cfg["student"]
    warm_start = paths_cfg["init_from"] or paths_cfg["resume_from"]
    # A checkpoint replaces every backbone weight, so its DINOv3 warm start is skipped.
    student = build_student(features=out_dim, fpn_dim=sc["fpn_dim"], depth_dim=sc["depth_dim"],
                            backbone=sc["backbone"], mobilevit_variant=sc["mobilevit_variant"],
                            mobilevit_init_from="" if warm_start else sc["mobilevit_init_from"],
                            norm=sc["norm"], freeze_bn=sc["freeze_bn"]).to(device)
    n_student = count_parameters(student)
    n_teacher = sum(p.numel() for p in teacher_encoder.parameters())
    core = sum(p.numel() for n, p in teacher_model.named_parameters() if not n.startswith("defomencoder"))
    print("Teacher encoder %.1f MB -> student %.1f MB; deployed model ~%.1f MB (stereo core %.1f MB)"
          % (n_teacher * 4 / 1e6, n_student * 4 / 1e6, (core + n_student) * 4 / 1e6, core * 4 / 1e6))

    if paths_cfg["init_from"]:
        state = torch.load(paths_cfg["init_from"], map_location=device, weights_only=True)
        student.load_state_dict(state["student"] if "student" in state else state)
        print("Initialised student from %s" % paths_cfg["init_from"])
        # Score the checkpoint exactly as saved -- before the BatchNorm statistics
        # are re-estimated on this data -- so student_best.pth only changes if
        # training beats it, and otherwise stays that checkpoint.
        if tcfg["validate_start"]:
            print("Validating the starting checkpoint ...")
            update_best(validate_all(), "start")

    # Before the optimiser and any resume. A loaded student keeps its own
    # output calibration (its heads were trained against it).
    uncalibrated = student.fuse_scale.item() == 1.0 and student.fuse_shift.item() == 0.0
    if not paths_cfg["resume_from"] and uncalibrated:
        calibrate_output_scales(teacher_encoder, student, loader, teacher_model, n_downsample, device,
                                tcfg["calibration_batches"])
    if not paths_cfg["resume_from"] and student.freeze_bn:
        recalibrate_batchnorm(student, loader, teacher_model, n_downsample, device,
                              tcfg["bn_recalibration_batches"])

    teacher_bf16 = bool(tcfg["teacher_bf16"])
    if teacher_bf16 and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
        print("training.teacher_bf16 ignored: this GPU has no bfloat16 support")
        teacher_bf16 = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = max(1, tcfg["epochs"] * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=tcfg["min_lr"])
    amp_enabled = bool(tcfg["amp"]) and device.type == "cuda"
    # torch.amp.GradScaler on torch >= 2.3; older JetPack builds only have torch.cuda.amp's.
    scaler = (torch.amp.GradScaler("cuda", enabled=amp_enabled) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp_enabled))

    start_epoch = 0
    if paths_cfg["resume_from"]:
        state = torch.load(paths_cfg["resume_from"], map_location=device, weights_only=True)
        if not isinstance(state.get("epoch", 0), int):
            raise SystemExit("resume_from %s is not a training epoch (epoch %r)"
                             % (paths_cfg["resume_from"], state["epoch"]))
        student.load_state_dict(state["student"])
        start_epoch = int(state.get("epoch", 0)) + 1
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            # Stretch the restored cosine to this run's length (more epochs would
            # otherwise wrap it back up) and jump to that schedule's current rate.
            scheduler.T_max = total_steps
            for group, lr in zip(optimizer.param_groups, scheduler._get_closed_form_lr()):
                group["lr"] = lr
        else:
            # Older epochs are saved without optimizer state (see below).
            print("No optimizer state in %s: AdamW restarts, LR schedule fast-forwarded to epoch %d"
                  % (paths_cfg["resume_from"], start_epoch))
            for _ in range(start_epoch * len(loader)):
                scheduler.step()
        print("Resumed from %s at epoch %d" % (paths_cfg["resume_from"], start_epoch))

    # A resume whose best.json already matches has scored these weights.
    if tcfg["validate_start"] and paths_cfg["resume_from"] and best["epoch"] is None:
        print("Validating the starting checkpoint ...")
        update_best(validate_all(), "start")

    # ---- epochs
    # The newest epoch on disk keeps its optimizer state until the next one is saved.
    resumed = Path(paths_cfg["resume_from"]) if paths_cfg["resume_from"] else None
    previous_ckpt = resumed if resumed is not None and resumed.parent.resolve() == ckpt_dir.resolve() else None
    for epoch in range(start_epoch, tcfg["epochs"]):
        student.train()
        running = {}
        started = fetched = time.time()
        data_wait = 0.0    # large vs s/step means CPU-bound: raise num_workers
        for step, (left, right) in enumerate(loader):
            data_wait += time.time() - fetched
            left, right = left.to(device, non_blocking=True), right.to(device, non_blocking=True)
            x, oh, ow, _ = preprocess(teacher_model, left, right, n_downsample)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=teacher_bf16):
                teacher_out = teacher_encoder(x, oh, ow)
            if teacher_bf16:
                teacher_out = ([t.float() for t in teacher_out[0]], *(t.float() for t in teacher_out[1:]))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp_enabled):
                loss, parts = distill_loss(student(x, oh, ow), teacher_out, cfg)
            scaler.scale(loss).backward()
            if tcfg["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), tcfg["grad_clip"])
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # The scaler skips steps whose gradients overflowed; the schedule
            # only advances on steps that updated the weights.
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

        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - started, 1)}
        record.update({k: round(v / len(loader), 6) for k, v in running.items()})
        if epoch % tcfg["val_every"] == 0:
            metrics = validate_all()
            record.update(metrics)
            update_best(metrics, "epoch %d" % epoch)
        else:
            record.update({"val_%s_L%d_%s" % (s, L, k): float("nan") for s in val_sets
                           for L in cfg["data"]["val_long_side"] for k in val_keys})
            record["val_score"] = float("nan")
        append_log(ckpt_dir, record)
        print("epoch %d done: total %.4f  (%.1f s)" % (epoch, record.get("total", float("nan")), record["seconds"]))

        if (epoch + 1) % tcfg["checkpoint_every"] == 0 or epoch == tcfg["epochs"] - 1:
            out = ckpt_dir / ("student_epoch%04d.pth" % epoch)
            atomic_save({"student": student.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "epoch": epoch, "config": portable_cfg,
                         "student_cfg": portable_student_cfg, "out_dim": out_dim}, out)
            print("  saved %s" % out)
            # Only the newest epoch keeps optimizer state (~27 MB instead of ~82 MB per epoch).
            if previous_ckpt is not None and previous_ckpt.exists():
                state = torch.load(str(previous_ckpt), map_location="cpu", weights_only=True)
                state.pop("optimizer", None)
                state.pop("scheduler", None)
                atomic_save(state, previous_ckpt)
            previous_ckpt = out

    print("Training complete. Checkpoints in %s" % ckpt_dir)
    if best["epoch"] is not None:
        print("Best: %s, val score %.2f%% -> %s" % (best["epoch"], 100 * best["score"], best_pth))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="", help="YAML config path")
    parser.add_argument("--image_dir", default="", help="override paths.image_dir")
    parser.add_argument("--teacher_ckpt", default="", help="override paths.teacher_ckpt")
    parser.add_argument("--checkpoint_dir", default="", help="override paths.checkpoint_dir")
    parser.add_argument("--init_from", default=None, help="fine-tune from a file or run folder ('' clears it)")
    parser.add_argument("--resume_from", default=None, help="continue a run from a file or run folder")
    parser.add_argument("--lr", type=float, default=0.0, help="override training.lr")
    parser.add_argument("--epochs", type=int, default=0, help="override training.epochs")
    parser.add_argument("--batch_size", type=int, default=0, help="override training.batch_size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key in ("image_dir", "teacher_ckpt", "checkpoint_dir"):
        if getattr(args, key):
            cfg["paths"][key] = getattr(args, key)
    for key in ("init_from", "resume_from"):
        if getattr(args, key) is not None:
            cfg["paths"][key] = getattr(args, key)
    for key in ("lr", "epochs", "batch_size"):
        if getattr(args, key):
            cfg["training"][key] = getattr(args, key)
    if not cfg["paths"]["image_dir"]:
        raise SystemExit("Set paths.image_dir in the config, or pass --image_dir")
    train(cfg)


if __name__ == "__main__":
    main()
