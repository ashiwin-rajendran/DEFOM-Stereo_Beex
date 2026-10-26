#!/usr/bin/env python3
"""Stage 2 of slim DEFOM: train the slim stereo core on the teacher's disparity.

    python3 distillation/teacher_labels.py --config distillation/configs/slim_stage2.yaml   # once
    python3 distillation/train_slim.py     --config distillation/configs/slim_stage2.yaml

The slim model (slim_stereo.py, preset from the config) uses the stage-1
student encoder (frozen by default) and a freshly initialised slim stereo
core. Each training sample is a frame at the label scale with the teacher's
disparity, randomly rescaled (disparity scaled with it) and cropped; the loss
is L1 to the teacher at every refinement iteration, later ones weighted more.
Validation runs whole held-out frames and scores the share of pixels within
3 px of the teacher; the best epoch is kept as student_best.pth + best.json.
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
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from core.utils.utils import InputPadder
from defom_student import load_student
from distill_defom import (StereoPairDataset, append_log, atomic_save, discover_images, load_config,
                           resolve_paths, split_sessions)
from slim_stereo import PRESETS, SlimDEFOM, load_slim
from teacher_labels import label_path, read_label


class LabelledPairs(Dataset):
    """(left, right, teacher disparity) at the label scale. Training adds a
    random rescale (disparity x s) and a random crop shared by all three."""

    def __init__(self, paths, cfg, train):
        self.paths = list(paths)
        d = cfg["data"]
        self.frames = StereoPairDataset(self.paths, cfg, long_side=d["label_long_side"])
        self.labels_dir, self.image_dir = cfg["paths"]["labels_dir"], cfg["paths"]["image_dir"]
        self.train = train
        self.crop = tuple(d["crop"])
        self.jitter = tuple(d["scale_jitter"])
        self.photometric = bool(d["photometric"]) and train
        self.brightness, self.contrast = float(d["brightness"]), float(d["contrast"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        left, right = self.frames[index]                       # 3xHxW float, 0-255
        label = read_label(label_path(self.labels_dir, self.image_dir, self.paths[index]))
        if label is None or label.shape != tuple(left.shape[-2:]):
            raise RuntimeError("missing or mis-sized label for %s: run teacher_labels.py" % self.paths[index])
        label = torch.from_numpy(label)[None]
        if self.train:
            s = random.uniform(*self.jitter)
            ch, cw = self.crop
            h, w = left.shape[-2:]
            s = max(s, ch / h, cw / w)                         # never smaller than the crop
            size = (int(round(h * s)), int(round(w * s)))
            left = F.interpolate(left[None], size, mode="bilinear", align_corners=False)[0]
            right = F.interpolate(right[None], size, mode="bilinear", align_corners=False)[0]
            label = F.interpolate(label[None], size, mode="nearest")[0] * s
            y, x = random.randint(0, size[0] - ch), random.randint(0, size[1] - cw)
            left, right, label = (t[:, y:y + ch, x:x + cw] for t in (left, right, label))
            if self.photometric:
                beta = random.uniform(-self.brightness, self.brightness) * 255.0
                alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
                left = (left * alpha + beta).clamp(0, 255)
                right = (right * alpha + beta).clamp(0, 255)
        return left.contiguous(), right.contiguous(), label.contiguous()


def sequence_loss(predictions, label, gamma):
    """Weighted L1 to the teacher over every iteration's disparity (valid pixels only)."""
    valid = (label > 0).float()
    n = len(predictions)
    total = 0.0
    for i, pred in enumerate(predictions):
        err = (pred - label).abs() * valid
        total = total + gamma ** (n - 1 - i) * err.sum() / valid.sum().clamp_min(1.0)
    with torch.no_grad():
        e = (predictions[-1] - label).abs()[valid > 0]
        metrics = {"epe": float(e.mean()) if e.numel() else float("nan"),
                   "within3": float((e < 3).float().mean()) if e.numel() else float("nan")}
    return total, metrics


@torch.no_grad()
def validate(model, val_sets, cfg, device, iters, scale_iters):
    """Whole frames at the label scale; per source: mean EPE and share within 1/3 px of the teacher."""
    model.eval()
    out = {}
    for source, items in val_sets.items():
        data = LabelledPairs(items, cfg, train=False)
        stats = []
        for i in range(len(data)):
            left, right, label = data[i]
            left, right, label = left[None].to(device), right[None].to(device), label[None].to(device)
            padder = InputPadder(left.shape, divis_by=32)
            pred = padder.unpad(model(*padder.pad(left, right), iters=iters, scale_iters=scale_iters,
                                      test_mode=True)).float()
            valid = label > 0
            e = (pred - label).abs()[valid]
            stats.append((float(e.mean()), float((e < 1).float().mean()), float((e < 3).float().mean())))
        # Plain floats: NumPy scalars would make the checkpoint unloadable with weights_only.
        a = [float(v) for v in np.array(stats).mean(0)] if stats else [float("nan")] * 3
        out.update({"val_%s_epe" % source: round(a[0], 4), "val_%s_within1" % source: round(a[1], 5),
                    "val_%s_within3" % source: round(a[2], 5)})
        print("  [val vs teacher] %-10s EPE %.2f px | within 1px %.1f%% | within 3px %.1f%%"
              % (source, a[0], 100 * a[1], 100 * a[2]))
    scores = [v for k, v in out.items() if k.endswith("_within3") and v == v]
    out["val_score"] = round(sum(scores) / len(scores), 5) if scores else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--resume_from", default=None, help="this run's folder or a slim checkpoint")
    args = ap.parse_args()

    portable_cfg = load_config(args.config)
    if args.epochs:
        portable_cfg["training"]["epochs"] = args.epochs
    if args.resume_from is not None:
        portable_cfg["paths"]["resume_from"] = args.resume_from
    cfg = resolve_paths(portable_cfg)
    for key in ("labels_dir", "student_ckpt"):
        value = cfg["paths"].get(key, "")
        if value and not Path(value).is_absolute():
            cfg["paths"][key] = str(Path(value).resolve() if Path(value).exists() else ROOT / value)
    t = cfg["training"]
    preset = cfg["slim"]["preset"]
    torch.manual_seed(t["seed"]); random.seed(t["seed"]); np.random.seed(t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "resolved_config.json").write_text(json.dumps(portable_cfg, indent=2))

    # ---- data
    paths = discover_images(cfg["paths"]["image_dir"])
    print("Dataset sessions:")
    train_paths, val_paths, source_of = split_sessions(paths, cfg)
    missing = [p for p in paths if not label_path(cfg["paths"]["labels_dir"], cfg["paths"]["image_dir"], p).is_file()]
    if missing:
        raise SystemExit("%d frames have no teacher label (e.g. %s): run teacher_labels.py first"
                         % (len(missing), missing[0]))
    loader = DataLoader(LabelledPairs(train_paths, cfg, train=True), batch_size=t["batch_size"], shuffle=True,
                        num_workers=t["num_workers"], drop_last=True, pin_memory=device.type == "cuda")
    if len(loader) == 0:
        raise SystemExit("fewer training frames than one batch")
    val_sets = collections.OrderedDict()
    for source in sorted(set(source_of[p] for p in val_paths)):
        items = [p for p in val_paths if source_of[p] == source]
        val_sets[source] = items[::max(1, len(items) // t["val_max_frames"])][:t["val_max_frames"]]
    print("Training %d frames (%d steps/epoch), validating %s" % (len(train_paths), len(loader),
                                                                {k: len(v) for k, v in val_sets.items()}))

    # ---- model
    resume = cfg["paths"].get("resume_from") or ""
    init = cfg["paths"].get("init_from") or ""
    if resume or init:
        model, ck, src = load_slim(resume or init, "cpu", prefer_best=not resume)
        if ck["slim"] != dict(PRESETS[preset]):
            raise SystemExit("%s is preset %s, the config says %s" % (src, ck["slim"], preset))
        student_cfg, out_dim = ck["student_cfg"], ck["out_dim"]
        print("%s from %s (epoch %s)" % ("Resuming" if resume else "Fine-tuning", src, ck.get("epoch")))
    else:
        student, sck, src = load_student(cfg["paths"]["student_ckpt"], "cpu")
        if abs(student.input_scale - PRESETS[preset]["student_scale"]) > 1e-6:
            raise SystemExit("stage-1 student %s was trained at input_scale %s; preset %s needs %s"
                             % (src, student.input_scale, preset, PRESETS[preset]["student_scale"]))
        model = SlimDEFOM(cfg["arch"], PRESETS[preset], student)
        student_cfg, out_dim = dict(sck["student_cfg"], input_scale=student.input_scale), sck["out_dim"]
        print("Slim %s with stage-1 student %s (epoch %s); stereo core freshly initialised" % (preset, src, sck.get("epoch")))
    model.to(device)
    model.mixed_precision = bool(t["amp"]) and device.type == "cuda"
    student = model.defomencoder.depth_anything
    core = [p for n, p in model.named_parameters() if not n.startswith("defomencoder.")]
    groups = [{"params": core, "lr": t["lr"]}]
    if t["freeze_student"]:
        for p in student.parameters():
            p.requires_grad = False
    else:
        groups.append({"params": list(student.parameters()), "lr": t["student_lr"]})
    print("Trainable: stereo core %.2f M%s" % (sum(p.numel() for p in core) / 1e6,
                                               "" if t["freeze_student"] else " + student (lr %g)" % t["student_lr"]))

    optimizer = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])
    total_steps = t["epochs"] * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups],
                                                    total_steps=total_steps, pct_start=0.05,
                                                    anneal_strategy="cos",
                                                    final_div_factor=t["lr"] / 25.0 / t["min_lr"])   # ends at min_lr
    scaler = torch.amp.GradScaler("cuda", enabled=model.mixed_precision)
    start_epoch = 0
    if resume:
        if "optimizer" not in ck:
            raise SystemExit("%s has no optimizer state: resume from the run's newest epoch" % src)
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        # Rebuild the one-cycle schedule for this run's length (it may have
        # more epochs) and continue from the step already reached.
        done = start_epoch * len(loader)
        if done >= total_steps:
            raise SystemExit("already trained %d epochs; pass --epochs > %d" % (start_epoch, start_epoch))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups],
                                                        total_steps=total_steps, pct_start=0.05,
                                                        anneal_strategy="cos",
                                                        final_div_factor=t["lr"] / 25.0 / t["min_lr"],
                                                        last_epoch=done - 1)

    # ---- best checkpoint, comparable only on the same frames and iterations
    signature = hashlib.sha1(json.dumps({"frames": sorted(str(p) for v in val_sets.values() for p in v),
                                         "iters": [t["iters"], t["scale_iters"]],
                                         "labels": cfg["paths"]["labels_dir"]}).encode()).hexdigest()[:12]
    best_json, best_pth = ckpt_dir / "best.json", ckpt_dir / "student_best.pth"
    best = {"score": float("-inf"), "epoch": None}
    if best_json.is_file() and json.loads(best_json.read_text()).get("signature") == signature:
        best = json.loads(best_json.read_text())

    def payload(epoch, extra=None):
        d = {"model": model.state_dict(), "slim": dict(PRESETS[preset]), "preset": preset, "epoch": epoch,
             "config": portable_cfg, "student_cfg": student_cfg, "out_dim": out_dim}
        d.update(extra or {})
        return d

    # The newest epoch on disk keeps its optimizer state until the next one is saved.
    previous = Path(src) if resume and Path(src).parent.resolve() == ckpt_dir.resolve() else None
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        if t["freeze_student"]:
            student.eval()
        started, running, n = time.time(), collections.Counter(), 0
        for step, (left, right, label) in enumerate(loader):
            left, right, label = left.to(device), right.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(left, right, iters=t["iters"], scale_iters=t["scale_iters"])
            loss, m = sequence_loss(predictions, label, t["gamma"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            n += 1
            running.update({"loss": float(loss.detach()), "epe": m["epe"], "within3": m["within3"]})
            if (step + 1) % 50 == 0:
                print("  epoch %d step %d/%d  loss %.3f  EPE %.2f px  within3 %.1f%%  lr %.2e  (%.2f s/step)"
                      % (epoch, step + 1, len(loader), running["loss"] / n, running["epe"] / n,
                         100 * running["within3"] / n, scheduler.get_last_lr()[0], (time.time() - started) / n),
                      flush=True)
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - started, 1)}
        record.update({k: round(v / max(1, n), 5) for k, v in running.items()})
        if epoch % t["val_every"] == 0 or epoch == t["epochs"] - 1:
            metrics = validate(model, val_sets, cfg, device, t["iters"], t["scale_iters"])
            record.update(metrics)
            score = metrics["val_score"]
            if score == score and score > best["score"]:
                best = {"score": score, "epoch": "epoch %d" % epoch, "signature": signature, "metrics": metrics}
                atomic_save(payload("epoch %d" % epoch, {"val_score": score}), best_pth)
                tmp = Path(str(best_json) + ".tmp")
                tmp.write_text(json.dumps(best, indent=2))
                os.replace(str(tmp), str(best_json))
                print("  new best: epoch %d (val score %.2f%%)" % (epoch, 100 * score))
            else:
                print("  best remains %s (%.2f%%)" % (best["epoch"], 100 * best["score"]))
        append_log(ckpt_dir, record)
        out = ckpt_dir / ("student_epoch%04d.pth" % epoch)
        atomic_save(payload(epoch, {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}), out)
        if previous is not None and previous.exists():         # only the newest keeps optimizer state
            state = torch.load(str(previous), map_location="cpu", weights_only=True)
            state.pop("optimizer", None); state.pop("scheduler", None)
            atomic_save(state, previous)
        previous = out
        print("epoch %d done: loss %.3f (%.0f s)" % (epoch, record.get("loss", float("nan")), record["seconds"]))
    print("Training complete. Best: %s (%.2f%%) -> %s" % (best["epoch"], 100 * best["score"], best_pth))


if __name__ == "__main__":
    main()
