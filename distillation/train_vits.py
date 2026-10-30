#!/usr/bin/env python3
"""Fine-tune the fast ViT-S DEFOM (slim_stereo.py preset vits_s2) on ViT-L labels.

    python3 distillation/slim_stereo.py --make vits_s2          # the model itself: the ViT-S release, no training
    python3 distillation/teacher_labels.py --config distillation/2_configs/vits_s2_finetune.yaml   # labels, once
    python3 distillation/train_vits.py     --config distillation/2_configs/vits_s2_finetune.yaml

Model: init_from (the vits_s2 checkpoint), or, if empty, the ViT-S release's own
weights (paths.release_ckpt) -- vits_s2 keeps every tensor's shape, so nothing
starts fresh. Only the parameters named in training.train are trained.

Loss, per step, on the same crops:
  disp      L1 to the ViT-L teacher's disparity (labels from teacher_labels.py, the
            same labels as slim v9), at every refinement iteration, later ones
            weighted more;
  internal  relative MSE to the frozen ViT-S release's internal outputs (matching
            features, GRU states and context, monocular disparity), which keeps the
            pretrained matcher anchored. Learning matching from disparity alone is
            what made slim v9 and the retrained ViT-S variants fail on scenes they
            never saw (tank, lab).

Validation, every epoch, vs ViT-L at 32/8 (paths.teacher_ckpt) at val_long_side:
  - the held-out tail of every training session;
  - probe sets (paths.probe_dir/<set>/): new scenes never trained on;
  - a shift test per set: the left image shifted by 30 px as the right image, so
    the true disparity is 30 everywhere; a model off by > 2 px fails.
The release's own result at the deployed iterations is printed as the bar to beat.
val_score = mean "within 3 px" over all sets; an epoch that fails the shift test is
never kept as student_best.pth.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # whole-frame validation fragments memory

import argparse
import collections
import hashlib
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (DEFOM_ARCH, append_log, atomic_save, discover_images, load_config, resolve_paths,
                    split_sessions, subsample, StereoPairDataset)
from core.utils.utils import InputPadder
from slim_stereo import PRESETS, SlimDEFOM, load_defom_weights, load_slim
from teacher_labels import label_path
from train_slim import LabelledPairs

SHIFT_PX = 30


def rel_mse(a, b):
    return ((a.float() - b.float()) ** 2).mean() / (b.float() ** 2).mean().clamp_min(1e-8)


def internal_loss(model, release, w):
    """Relative MSE between the model's and the release's encode() outputs."""
    terms = {"fmap": 0.5 * (rel_mse(model["fmap1"], release["fmap1"]) + rel_mse(model["fmap2"], release["fmap2"]))}
    nets = [rel_mse(a, b) for a, b in zip(model["net"], release["net"])]
    inps = [rel_mse(a, b) for la, lb in zip(model["inp"], release["inp"]) if la is not None for a, b in zip(la, lb)]
    terms["net"] = sum(nets) / len(nets)
    terms["inp"] = sum(inps) / len(inps)
    terms["idepth"] = rel_mse(model["disp"], release["disp"])
    return sum(w["w_" + k] * v for k, v in terms.items()), terms


def sequence_l1(predictions, target, gamma):
    """Mean |disparity - target| (px) over valid pixels (target > 0), averaged
    over iterations with later ones weighted more."""
    n = len(predictions)
    weights = [gamma ** (n - 1 - i) for i in range(n)]
    valid = (target > 0).float()
    count = valid.sum().clamp_min(1.0)
    return sum(w * ((p.float() - target).abs() * valid).sum() / count
               for w, p in zip(weights, predictions)) / sum(weights)


def shift_pair(left, shift):
    """(left, right) cut from one image shift px apart: right(x) = left(x + shift),
    so the true disparity is shift everywhere."""
    w = left.shape[-1]
    return left[..., :w - shift].contiguous(), left[..., shift:].contiguous()


@torch.no_grad()
def run(model, left, right, iters, scale_iters):
    padder = InputPadder(left.shape, divis_by=32)
    return padder.unpad(model(*padder.pad(left, right), iters=iters, scale_iters=scale_iters, test_mode=True)).float()


def agreement(pred, ref):
    e = (pred - ref).abs()
    valid = ref > 1.0
    return float((e < 3).float().mean()), float((e[valid] / ref[valid]).median()) if valid.any() else float("nan")


def build_defom(path, device):
    """An original DEFOM-Stereo release (defomstereo_<vits|vitb|vitl>_<data>.pth) as an
    exact, frozen SlimDEFOM; the encoder comes from the file name."""
    encoder = next((e for e in ("vits", "vitb", "vitl") if e in Path(path).stem), None)
    if encoder is None:
        raise SystemExit("%s: cannot tell the encoder (vits/vitb/vitl) from the file name" % path)
    model = SlimDEFOM(dict(DEFOM_ARCH, dinov2_encoder=encoder), dict(PRESETS["vits"], encoder=encoder))
    fresh = load_defom_weights(model, path)
    if fresh:
        raise SystemExit("%s is not a DEFOM-Stereo %s release (%d tensors missing, e.g. %s)"
                         % (path, encoder, len(fresh), fresh[0]))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class Validator:
    """Whole frames from every validation set with ViT-L's disparity (32/8) and the
    release's own score at the deployed iterations, computed once."""

    def __init__(self, sets, cfg, reference, release, device):
        t = cfg["training"]
        self.device = device
        self.iters, self.scale_iters = t["iters"], t["scale_iters"]
        self.sets = collections.OrderedDict()
        ri, rs = t.get("reference_iters", [32, 8])        # DEFOM's own; fixed so scores stay comparable
        print("Reference disparity for validation: %s at %d/%d iterations ..."
              % (Path(cfg["paths"]["teacher_ckpt"]).name, ri, rs))
        for name, paths in sets.items():
            data = StereoPairDataset(paths, cfg, long_side=t["val_long_side"])
            frames = []
            for i in range(len(data)):
                left, right = (x[None].to(device) for x in data[i])
                frames.append((left.cpu(), right.cpu(), run(reference, left, right, ri, rs).cpu()))
            self.sets[name] = frames
        self.release = self.score(release, verbose=False)
        print("  release @ deployed iters %d/%d (the bar): %s" % (self.iters, self.scale_iters, self.line(self.release)))

    @staticmethod
    def line(m):
        return " | ".join("%s %.1f%% %.2f%%%s" % (k, 100 * v["within3"], 100 * v["rel"],
                                                     "" if v["shift_ok"] else " SHIFT %.1f" % v["shift"])
                          for k, v in m["sets"].items())

    @torch.no_grad()
    def score(self, model, verbose=True):
        model.eval()
        amp, model.mixed_precision = model.mixed_precision, False      # score in fp32, as exported
        try:
            out = {"sets": collections.OrderedDict()}
            for name, frames in self.sets.items():
                stats = [agreement(run(model, l.to(self.device), r.to(self.device), self.iters, self.scale_iters),
                                   ref.to(self.device)) for l, r, ref in frames]
                shifts = []
                for left, _, _ in frames[:: max(1, len(frames) // 3)][:3]:
                    l, r = shift_pair(left.to(self.device), SHIFT_PX)
                    d = run(model, l, r, self.iters, self.scale_iters)[..., 16:-16, SHIFT_PX + 16:-16]
                    shifts.append(float(d.median()))                    # columns with a match
                shift, a = float(np.median(shifts)), np.array(stats)
                out["sets"][name] = {"within3": float(a[:, 0].mean()), "rel": float(np.nanmedian(a[:, 1])),
                                     "shift": shift, "shift_ok": abs(shift - SHIFT_PX) <= 2.0}
            out["val_score"] = float(np.mean([v["within3"] for v in out["sets"].values()]))
            out["shift_ok"] = all(v["shift_ok"] for v in out["sets"].values())
            if verbose:
                print("  [val vs ViT-L] %s | score %.2f%%%s" % (self.line(out), 100 * out["val_score"],
                                                               "" if out["shift_ok"] else "  SHIFT TEST FAILED"))
            return out
        finally:
            model.mixed_precision = amp
            torch.cuda.empty_cache()             # whole-frame validation leaves large cached blocks


def validation_sets(cfg, val_paths, source_of):
    n = cfg["training"]["val_max_frames"]
    sets = collections.OrderedDict()
    for source in sorted(set(source_of[p] for p in val_paths)):
        sets["heldout_" + source] = subsample([p for p in val_paths if source_of[p] == source], n)
    probe = Path(cfg["paths"].get("probe_dir") or "")
    if probe.is_dir():
        for folder in sorted(p for p in probe.iterdir() if p.is_dir()):
            sets["probe_" + folder.name] = subsample(discover_images(folder), n)
    return sets


def batchnorm_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--resume_from", default=None, help="this run's folder or a checkpoint")
    ap.add_argument("--max_steps", type=int, default=0, help="stop each epoch after N steps (smoke test)")
    args = ap.parse_args()

    portable_cfg = load_config(args.config)
    if args.epochs:
        portable_cfg["training"]["epochs"] = args.epochs
    if args.resume_from is not None:
        portable_cfg["paths"]["resume_from"] = args.resume_from
    cfg = resolve_paths(portable_cfg)
    t, w = cfg["training"], cfg["loss"]
    preset = cfg["vits"]["preset"]
    if PRESETS[preset].get("encoder") != "vits":
        raise SystemExit("vits.preset %s is not a ViT-S preset (slim_stereo.py PRESETS)" % preset)
    torch.manual_seed(t["seed"]); random.seed(t["seed"]); np.random.seed(t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = bool(t["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(t["tf32"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "resolved_config.json").write_text(json.dumps(portable_cfg, indent=2))

    # ---- data
    print("Dataset sessions:")
    train_paths, val_paths, source_of = split_sessions(discover_images(cfg["paths"]["image_dir"]), cfg)
    missing = [p for p in train_paths if not label_path(cfg["paths"]["labels_dir"], cfg["paths"]["image_dir"], p).is_file()]
    if missing:
        raise SystemExit("%d frames have no label (e.g. %s): run teacher_labels.py first" % (len(missing), missing[0]))
    loader = DataLoader(LabelledPairs(train_paths, cfg, train=True), batch_size=t["batch_size"], shuffle=True,
                        num_workers=t["num_workers"], drop_last=True, pin_memory=device.type == "cuda")
    sets = validation_sets(cfg, val_paths, source_of)
    steps_per_epoch = min(len(loader), args.max_steps) if args.max_steps else len(loader)
    print("Training %d frames (%d steps/epoch); validation %s" % (len(train_paths), steps_per_epoch,
                                                                 {k: len(v) for k, v in sets.items()}))

    # ---- release (frozen anchor) and model
    # The release runs in fp32 (TF32): DEFOM's fnet.convd exceeds fp16's range (~1e5), so
    # fp16 autocast gives inf/NaN on some frames. The model trains in bf16 (fp32's range).
    release = build_defom(cfg["paths"]["release_ckpt"], device)
    resume = cfg["paths"].get("resume_from") or ""
    init = cfg["paths"].get("init_from") or ""
    if resume or init:
        model, ck, src = load_slim(resume or init, "cpu", prefer_best=not resume)
        if ck["slim"] != dict(PRESETS[preset]):
            raise SystemExit("%s is %s, the config says preset %s" % (src, ck["slim"], preset))
        print("%s from %s (epoch %s)" % ("Resuming" if resume else "Fine-tuning", src, ck.get("epoch")))
    else:
        model = SlimDEFOM(dict(DEFOM_ARCH, dinov2_encoder="vits"), PRESETS[preset])
        fresh = load_defom_weights(model, cfg["paths"]["release_ckpt"])
        if fresh:
            raise SystemExit("preset %s does not fit the release unchanged (e.g. %s)" % (preset, fresh[0]))
        print("%s from the release's weights" % preset)
    model.to(device)
    model.mixed_precision = bool(t["amp"]) and device.type == "cuda"
    model.amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    trainable = tuple(t["train"])
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable (%s): %.2f M of %.2f M parameters" % (", ".join(trainable), sum(p.numel() for p in params) / 1e6,
                                                          sum(p.numel() for p in model.parameters()) / 1e6))
    # the depth encoder (if trained) moves more slowly: it is pretrained and shared by everything after it
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("defomencoder.")]
    core = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("defomencoder.")]
    groups = [g for g in ({"params": core, "lr": t["lr"]},
                          {"params": enc, "lr": t["lr"] * t.get("encoder_lr_factor", 0.1)}) if g["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])
    total_steps = t["epochs"] * steps_per_epoch
    schedule = dict(max_lr=[g["lr"] for g in groups], total_steps=total_steps, pct_start=min(0.5, max(0.05, 3.0 / total_steps)),
                    anneal_strategy="cos", final_div_factor=t["lr"] / 25.0 / t["min_lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=model.mixed_precision and model.amp_dtype == torch.float16)
    start_epoch = 0
    if resume:
        if "optimizer" not in ck:
            raise SystemExit("%s has no optimizer state: resume from the run's newest epoch" % src)
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        done = start_epoch * steps_per_epoch
        if done >= total_steps:
            raise SystemExit("already trained %d epochs; pass --epochs > %d" % (start_epoch, start_epoch))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, last_epoch=done - 1, **schedule)

    reference = build_defom(cfg["paths"]["teacher_ckpt"], device)
    validator = Validator(sets, cfg, reference, release, device)
    del reference
    torch.cuda.empty_cache()
    signature = hashlib.sha1(json.dumps({"frames": sorted(str(p) for v in sets.values() for p in v),
                                         "iters": [t["iters"], t["scale_iters"]], "val_long_side": t["val_long_side"],
                                         "reference": Path(cfg["paths"]["teacher_ckpt"]).name}).encode()).hexdigest()[:12]
    best_json, best_pth = ckpt_dir / "best.json", ckpt_dir / "student_best.pth"
    best = {"score": float("-inf"), "epoch": None}
    if best_json.is_file() and json.loads(best_json.read_text()).get("signature") == signature:
        best = json.loads(best_json.read_text())

    def payload(epoch, extra=None):
        d = {"model": model.state_dict(), "slim": dict(PRESETS[preset]), "preset": preset, "epoch": epoch,
             "config": portable_cfg, "student_cfg": {"backbone": "dinov2_vits"}, "out_dim": model.defomencoder.out_dim,
             "release_val": validator.release}
        d.update(extra or {})
        return d

    def keep_if_best(metrics, label):
        nonlocal best
        score = metrics["val_score"]
        if not metrics["shift_ok"]:
            print("  not eligible as best: shift test failed")
        elif score == score and score > best["score"]:
            best = {"score": score, "epoch": label, "signature": signature, "metrics": metrics,
                    "release_val": validator.release}
            atomic_save(payload(label, {"val_score": score}), best_pth)
            tmp = Path(str(best_json) + ".tmp")
            tmp.write_text(json.dumps(best, indent=2))
            os.replace(str(tmp), str(best_json))
            print("  new best: %s (val score %.2f%%; release %.2f%%)"
                  % (label, 100 * score, 100 * validator.release["val_score"]))
        else:
            print("  best remains %s (%.2f%%)" % (best["epoch"], 100 * best["score"]))

    def flat(metrics):
        out = {"val_score": round(metrics["val_score"], 5), "shift_ok": metrics["shift_ok"]}
        for k, v in metrics["sets"].items():
            out.update({"%s_within3" % k: round(v["within3"], 5), "%s_rel" % k: round(v["rel"], 5),
                        "%s_shift" % k: round(v["shift"], 2)})
        return out

    if not resume:
        print("Validating the starting model ...")
        keep_if_best(validator.score(model), "start")

    previous = Path(src) if resume and Path(src).parent.resolve() == ckpt_dir.resolve() else None
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        batchnorm_eval(model)                 # batch 4: train == deployed function
        started, running, n = time.time(), collections.Counter(), 0
        for step, (left, right, target) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            left, right = left.to(device, non_blocking=True), right.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.no_grad():
                r_enc = release.encode(left, right)
            enc = model.encode(left, right)
            loss, terms = internal_loss(enc, r_enc, w)
            terms["disp"] = sequence_l1(model.refine(enc, t["iters"], t["scale_iters"]), target, t["gamma"])
            loss = loss + w["w_disp"] * terms["disp"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            n += 1
            running.update({"loss": float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()}})
            if (step + 1) % 50 == 0:
                print("  epoch %d step %d/%d  %s  lr %.2e  (%.2f s/step)"
                      % (epoch, step + 1, steps_per_epoch, "  ".join("%s %.4f" % (k, v / n) for k, v in running.items()),
                         scheduler.get_last_lr()[0], (time.time() - started) / n), flush=True)
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - started, 1)}
        record.update({k: round(v / max(1, n), 6) for k, v in running.items()})
        if epoch % t["val_every"] == 0 or epoch == t["epochs"] - 1:
            metrics = validator.score(model)
            record.update(flat(metrics))
            keep_if_best(metrics, "epoch %d" % epoch)
        append_log(ckpt_dir, record)
        out = ckpt_dir / ("student_epoch%04d.pth" % epoch)
        atomic_save(payload(epoch, {"optimizer": optimizer.state_dict()}), out)
        if previous is not None and previous.exists():         # only the newest keeps optimizer state
            state = torch.load(str(previous), map_location="cpu", weights_only=True)
            state.pop("optimizer", None)
            atomic_save(state, previous)
        previous = out
        print("epoch %d done: loss %.4f (%.0f s)" % (epoch, record.get("loss", float("nan")), record["seconds"]))
    print("Training complete. Best: %s (%.2f%%; release at the same iterations %.2f%%) -> %s"
          % (best["epoch"], 100 * best["score"], 100 * validator.release["val_score"], best_pth))


if __name__ == "__main__":
    main()
