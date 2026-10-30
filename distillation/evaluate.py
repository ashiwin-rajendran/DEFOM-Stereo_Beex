#!/usr/bin/env python3
"""Evaluate models against the DEFOM ViT-L teacher: metrics, charts, report.

    python3 distillation/evaluate.py \\
        --models slim_v9=distillation/3_checkpoints/1_slim_model \\
                 vits_s2=distillation/3_checkpoints/2_fast_vits_model@10/4 \\
                 new=distillation/3_checkpoints/slim_v9_finetune \\
        --long_sides 512 800 --out distillation/4_evaluations/<date>_<name>

    # one model at other refinement iterations (name=path@iters/scale_iters):
    python3 distillation/evaluate.py --models it12=<run>@12/4 it8=<run>@8/4

A run folder picks its student_best.pth. The teacher (--teacher, ViT-L) runs at
--iters/--scale_iters (default 32/8); a model without @ at its own training
iterations.

Every model runs end to end on the held-out frames of the training split (so
nothing it trained on), or on every frame of --image_dir (--frames all), and is
compared pixel by pixel against the teacher.

Robustness:
  * every number has a bootstrap 95% confidence interval over frames;
  * models are compared PAIRED, frame by frame, against a reference model,
    with the probability that each is genuinely better;
  * a frame that fails for any model is dropped for all, so pairs stay paired;
  * models distilled against different stereo cores are refused, since they
    cannot share one teacher reference.

Writes to --out:
  report.md            verdict, ranking, per-camera tables, charts
  metrics.csv / .json  every number in the report
  summary.png          within-3px and relative depth error, with 95% CIs
  nearness.png         relative depth error from far to near
  texture.png          agreement on textured vs textureless pixels
  training_curves_*.png  loss and validation over epochs (if a training_log.csv is found)
  samples_*.jpg        left | teacher | each model's disparity, and error maps
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import collections
import csv
import json
import re
import time

import cv2
import numpy as np
import torch

from common import DEFOM_ARCH, ROOT, StereoPairDataset, discover_images, load_config, resolve_paths, split_sessions
from core.defom_stereo import DEFOMStereo
from core.utils.utils import InputPadder
from slim_stereo import load_slim

# Relative-depth-error bands, by disparity as a fraction of image width:
# camera-independent "nearness" (depth ~ 1 / disparity).
NEARNESS_EDGES = [0.0, 0.05, 0.10, 0.20, 0.30, 10.0]
NEARNESS_LABELS = ["<5%\n(far)", "5-10%", "10-20%", "20-30%", ">30%\n(near)"]
PIXELS_PER_FRAME = 4000      # sampled per frame for the nearness bands
ERROR_VIS_MAX_PX = 5.0       # error maps saturate at this many pixels
METRICS = ("within1", "within2", "within3", "epe_mean", "epe_median", "p90",
           "rel_median", "d1", "w3_textured", "w3_textureless")


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def parse_models(specs):
    """'name=path[@iters/scale_iters]' or 'path'. Unnamed models are named after
    their file. Returns {name: (path, (iters, scale_iters) or None)}."""
    models = collections.OrderedDict()
    for spec in specs:
        name, _, path = spec.partition("=") if "=" in spec else ("", "", spec)
        iters = None
        match = re.match(r"^(.*)@(\d+)/(\d+)$", path)
        if match:
            path, iters = match.group(1), (int(match.group(2)), int(match.group(3)))
            if iters[1] > iters[0]:
                raise SystemExit("%s: scale_iters cannot exceed iters" % spec)
        if not name:
            p = Path(path)
            name = p.stem if p.suffix else p.name
        if name in models:
            raise SystemExit("Duplicate model name %r -- name them with name=path" % name)
        models[name] = (path, iters)
    return models


def select_frames(cfg, frames, max_frames):
    """{source: evenly spaced frame paths}. 'heldout' uses exactly the
    trainer's per-session validation split; 'all' takes every frame under
    image_dir (e.g. a new dive no model trained on)."""
    paths = discover_images(cfg["paths"]["image_dir"])
    if frames == "heldout":
        _, pool, source_of = split_sessions(paths, cfg)
    else:
        pool = paths
        _, _, source_of = split_sessions(paths, dict(cfg, training=dict(cfg["training"], val_fraction=0.0)))
    sets = collections.OrderedDict()
    for source in sorted(set(source_of[p] for p in pool)):
        items = [p for p in pool if source_of[p] == source]
        step = max(1, len(items) // max(1, max_frames))
        sets[source] = items[::step][:max_frames]
    return sets


def texture_mask(left):
    """Textured pixels: local gradient above the frame's own median. Stereo is
    ill-posed in textureless water, so agreement there means less."""
    gray = cv2.cvtColor(left.permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    grad = cv2.blur(grad, (9, 9))
    return grad > np.median(grad)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def frame_metrics(pred, teacher, textured):
    """Per-frame agreement of one model's disparity map with the teacher's."""
    finite = np.isfinite(pred)
    err = np.where(finite, np.abs(pred - teacher), np.inf)
    rel = err / np.maximum(teacher, 1e-3)
    out = {
        "within1": float((err < 1).mean()),
        "within2": float((err < 2).mean()),
        "within3": float((err < 3).mean()),
        "epe_mean": float(np.mean(np.where(finite, err, 0.0))) if finite.any() else float("nan"),
        "epe_median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        # Relative disparity error equals relative depth error to first order.
        "rel_median": float(np.median(rel)) * 100.0,
        # KITTI-style outliers: off by more than 3 px AND more than 5%.
        "d1": float(((err > 3) & (rel > 0.05)).mean()) * 100.0,
        "w3_textured": float((err[textured] < 3).mean()),
        "w3_textureless": float((err[~textured] < 3).mean()),
        "nonfinite": float((~finite).mean()),
    }
    return out, rel


def run(args, models, cfg, sets, device):
    loaded = collections.OrderedDict()
    for name, (path, model_iters) in models.items():
        # A whole stereo model (slim v9, ViT-S s2), by default at the iterations it was trained with.
        net, ck, resolved = load_slim(path, device)
        own = (ck["config"]["training"]["iters"], ck["config"]["training"]["scale_iters"])
        loaded[name] = {"net": net, "ck": ck, "path": str(resolved), "iters": model_iters or own}
    teacher_ckpt = Path(args.teacher if Path(args.teacher).is_absolute() else ROOT / args.teacher).resolve()
    print("Teacher / stereo core: %s" % teacher_ckpt)
    model = DEFOMStereo(argparse.Namespace(**DEFOM_ARCH))
    state = torch.load(str(teacher_ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    del state
    model.to(device).eval()
    iters = args.iters or DEFOM_ARCH["valid_iters"]
    scale_iters = args.scale_iters or DEFOM_ARCH["scale_iters"]

    per_frame = collections.defaultdict(list)    # (model, source, L) -> [metrics dict]
    near = collections.defaultdict(list)         # (model, source) -> [(nearness, rel)]
    timing = collections.defaultdict(list)       # (model, L) -> [ms]
    samples = collections.defaultdict(list)      # source -> [(left, teacher, {model: disp})]
    skipped = 0
    rng = np.random.default_rng(args.seed)

    def disparity(net, lp, rp, padder, n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.time()
        with torch.no_grad():
            out = padder.unpad(net(lp, rp, iters=n_iters[0], scale_iters=n_iters[1], test_mode=True))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out[0, 0].float().cpu().numpy(), 1000.0 * (time.time() - started)

    for source, items in sets.items():
        sample_idx = set(np.linspace(0, len(items) - 1, min(args.samples, len(items))).round().astype(int))
        for long_side in args.long_sides:
            data = StereoPairDataset(items, cfg, long_side=long_side)
            for index in range(len(data)):
                try:
                    left, right = data[index]
                except RuntimeError as exc:
                    print("  skip unreadable %s (%s)" % (items[index], exc))
                    skipped += 1
                    continue
                textured = texture_mask(left)
                lp, rp = left[None].to(device), right[None].to(device)
                padder = InputPadder(lp.shape, divis_by=32)
                lp, rp = padder.pad(lp, rp)
                try:
                    teacher, _ = disparity(model, lp, rp, padder, (iters, scale_iters))
                    results = {name: disparity(item["net"], lp, rp, padder, item["iters"])
                               for name, item in loaded.items()}
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("  skip %s @%d: CUDA OOM (dropped for every model)" % (items[index].name, long_side))
                    skipped += 1
                    continue
                width = teacher.shape[1]
                pick = rng.choice(teacher.size, size=min(PIXELS_PER_FRAME, teacher.size), replace=False)
                nearness = (teacher.ravel() / width)[pick]
                for name, (disp, ms) in results.items():
                    metrics, rel = frame_metrics(disp, teacher, textured)
                    per_frame[(name, source, long_side)].append(metrics)
                    near[(name, source)].append(np.stack([nearness, rel.ravel()[pick]], 1))
                    timing[(name, long_side)].append(ms)
                if long_side == max(args.long_sides) and index in sample_idx:
                    rgb = left.permute(1, 2, 0).numpy().astype(np.uint8)
                    samples[source].append((cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), teacher,
                                            {n: r[0] for n, r in results.items()}))
            print("  %-10s @%4d: %d frames" % (source, long_side,
                                               len(per_frame[(next(iter(loaded)), source, long_side)])))
    meta = {"teacher_ckpt": str(teacher_ckpt), "iters": iters,
            "scale_iters": scale_iters, "skipped": skipped,
            "models": {n: {"path": i["path"], "epoch": i["ck"].get("epoch"), "iters": list(i["iters"]),
                           "val_score": i["ck"].get("val_score")} for n, i in loaded.items()}}
    return per_frame, near, timing, samples, meta


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def bootstrap(values, rng, n=2000):
    """95% CI of the mean, resampling frames."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()) if len(values) else float("nan"), float("nan"), float("nan")
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(per_frame, models, lines, reference, rng, n_boot=2000):
    table = {}
    for name in models:
        for line in lines:
            frames = per_frame[(name,) + line]
            row = {k: float(np.nanmean([f[k] for f in frames])) for k in METRICS}
            row["n"] = len(frames)
            row["within3"], row["w3_lo"], row["w3_hi"] = bootstrap([f["within3"] for f in frames], rng, n_boot)
            table[(name,) + line] = row

    # Overall score: mean within-3px over source x size lines, each weighted
    # equally -- the same score the trainer selects student_best.pth by.
    overall = {}
    for name in models:
        draws = np.zeros(n_boot)
        point = []
        for line in lines:
            v = np.array([f["within3"] for f in per_frame[(name,) + line]])
            point.append(v.mean())
            draws += v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
        draws /= len(lines)
        overall[name] = {"score": float(np.mean(point)), "lo": float(np.percentile(draws, 2.5)),
                         "hi": float(np.percentile(draws, 97.5))}

    # Paired comparison against the reference: same frames, so frame-to-frame
    # difficulty cancels and much smaller differences are detectable.
    paired = {}
    for name in models:
        if name == reference:
            continue
        draws = np.zeros(n_boot)
        point = []
        per_line = {}
        for line in lines:
            d = (np.array([f["within3"] for f in per_frame[(name,) + line]])
                 - np.array([f["within3"] for f in per_frame[(reference,) + line]]))
            idx = rng.integers(0, len(d), (n_boot, len(d)))
            line_draws = d[idx].mean(1)
            per_line["%s@%d" % line] = {"diff": float(d.mean()), "lo": float(np.percentile(line_draws, 2.5)),
                                        "hi": float(np.percentile(line_draws, 97.5))}
            point.append(d.mean())
            draws += line_draws
        draws /= len(lines)
        paired[name] = {"diff": float(np.mean(point)), "lo": float(np.percentile(draws, 2.5)),
                        "hi": float(np.percentile(draws, 97.5)), "p_better": float((draws > 0).mean()),
                        "lines": per_line}
    return table, overall, paired


def nearness_bands(near, models, sources):
    out = {}
    for name in models:
        for source in sources:
            pts = np.concatenate(near[(name, source)]) if near[(name, source)] else np.zeros((0, 2))
            idx = np.digitize(pts[:, 0], NEARNESS_EDGES) - 1
            out[(name, source)] = [float(np.median(pts[idx == b, 1]) * 100) if (idx == b).sum() > 50 else float("nan")
                                   for b in range(len(NEARNESS_LABELS))]
    return out


# --------------------------------------------------------------------------
# charts and images
# --------------------------------------------------------------------------


def line_label(line):
    source, long_side = line
    return "%s @%d" % ("Voyis" if source == "2816x2816" else ("DWE" if source in ("800x600", "1600x1200") else source),
                       long_side) + ("" if source in ("2816x2816", "800x600") else "\n(%s)" % source)


def charts(out, models, lines, table, bands, sources, run_logs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: tables only (pip install matplotlib for charts)")
        return []
    written = []
    colours = plt.cm.tab10(np.arange(len(models)) % 10)
    x = np.arange(len(lines))
    width = 0.8 / len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for i, name in enumerate(models):
        w3 = [100 * table[(name,) + l]["within3"] for l in lines]
        lo = [100 * (table[(name,) + l]["within3"] - table[(name,) + l]["w3_lo"]) for l in lines]
        hi = [100 * (table[(name,) + l]["w3_hi"] - table[(name,) + l]["within3"]) for l in lines]
        axes[0].bar(x + (i - (len(models) - 1) / 2) * width, w3, width, yerr=[lo, hi], capsize=3,
                    color=colours[i], label=name)
        rel = [table[(name,) + l]["rel_median"] for l in lines]
        axes[1].bar(x + (i - (len(models) - 1) / 2) * width, rel, width, color=colours[i], label=name)
    axes[0].set_title("Pixels within 3 px of the teacher (higher is better, 95% CI)")
    axes[0].set_ylabel("% of pixels")
    lows = [100 * table[(n,) + l]["w3_lo"] for n in models for l in lines]
    axes[0].set_ylim(max(0, min(lows) - 8), 100)
    axes[1].set_title("Median relative depth error (lower is better)")
    axes[1].set_ylabel("% of depth")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([line_label(l) for l in lines])
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out / "summary.png"), dpi=110)
    plt.close(fig)
    written.append("summary.png")

    fig, axes = plt.subplots(1, len(sources), figsize=(6.5 * len(sources), 4.2), squeeze=False)
    for j, source in enumerate(sources):
        ax = axes[0, j]
        for i, name in enumerate(models):
            ax.plot(range(len(NEARNESS_LABELS)), bands[(name, source)], "o-", color=colours[i], label=name)
        ax.set_xticks(range(len(NEARNESS_LABELS)))
        ax.set_xticklabels(NEARNESS_LABELS, fontsize=8)
        ax.set_xlabel("teacher disparity as % of image width")
        ax.set_ylabel("median relative depth error % (log scale)")
        # Log scale: far open water sits at 10-40%, structure at 0.1-2%.
        ax.set_yscale("log")
        ax.set_title("%s: error from far to near" % line_label((source, 0)).split(" @")[0])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out / "nearness.png"), dpi=110)
    plt.close(fig)
    written.append("nearness.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, name in enumerate(models):
        tex = np.mean([100 * table[(name,) + l]["w3_textured"] for l in lines])
        flat = np.mean([100 * table[(name,) + l]["w3_textureless"] for l in lines])
        ax.bar(np.array([0, 1]) + (i - (len(models) - 1) / 2) * width, [tex, flat], width,
               color=colours[i], label=name)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["textured pixels\n(structure)", "textureless pixels\n(open water)"])
    ax.set_ylabel("% within 3 px of the teacher")
    ax.set_title("Agreement by texture (stereo is ill-posed where there is none)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out / "texture.png"), dpi=110)
    plt.close(fig)
    written.append("texture.png")

    from matplotlib.ticker import MaxNLocator
    for run_name, (rows, best_info) in run_logs.items():
        epochs = [int(r["epoch"]) for r in rows]
        # per-set validation columns: val_<camera>_within3 (train_slim.py), heldout_/probe_<set>_within3 (train_vits.py)
        keys = [k for k in rows[0] if k.endswith("_within3") and k != "within3"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
        axes[0].plot(epochs, [as_float(r.get("loss")) for r in rows], "k-")
        axes[0].set_title("%s: training loss" % run_name)
        axes[0].set_xlabel("epoch")
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].grid(alpha=0.3)
        for k in keys:
            vals = [100 * as_float(r.get(k)) for r in rows]
            axes[1].plot(epochs, vals, "-", alpha=0.6, label=k[:-len("_within3")].replace("val_", ""))
        # Same score the trainer selects by; NaN for epochs without validation.
        mean = [np.mean([100 * as_float(r.get(k)) for k in keys]) for r in rows]
        axes[1].plot(epochs, mean, "k-", linewidth=2, label="mean (selection score)")
        # Mark what training actually saved as student_best.pth (best.json),
        # which may be the starting checkpoint -- not in this log at all.
        label = str((best_info or {}).get("epoch", ""))
        if label.startswith("epoch ") and label[6:].isdigit():
            axes[1].axvline(int(label[6:]), color="g", linestyle="--",
                            label="student_best.pth: %s" % label)
        elif label == "start" and best_info.get("score") is not None:
            axes[1].axhline(100 * best_info["score"], color="g", linestyle="--",
                            label="student_best.pth: the starting checkpoint (%.1f%%)"
                            % (100 * best_info["score"]))
        elif np.isfinite(mean).any():
            best = int(np.nanargmax(mean))
            axes[1].axvline(epochs[best], color="g", linestyle=":",
                            label="highest validation: epoch %d" % epochs[best])
        axes[1].set_title("validation: within 3 px of the teacher")
        axes[1].set_xlabel("epoch")
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].set_ylabel("%")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        name = "training_curves_%s.png" % run_name
        fig.savefig(str(out / name), dpi=110)
        plt.close(fig)
        written.append(name)
    return written


def sample_images(out, samples, models, tile_w=360):
    """Per frame: left | teacher | each model's disparity, then error maps.
    Disparity shares the teacher's 2-98th percentile colour scale, so equal
    colours mean equal disparity across the row."""
    written = []

    def key(h, colormap, left_text, right_text, title):
        """A colour key: the colormap as a horizontal bar with end labels."""
        img = np.full((h, tile_w, 3), 40, np.uint8)
        bar = cv2.applyColorMap(np.tile(np.linspace(0, 255, tile_w - 40).astype(np.uint8), (24, 1)), colormap)
        y = h // 2 - 12
        img[y:y + 24, 20:tile_w - 20] = bar
        for text, org in ((title, (20, y - 14)), (left_text, (20, y + 44)),
                          (right_text, (tile_w - 20 - 9 * len(right_text), y + 44))):
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return img

    def tile(img, text, h):
        img = cv2.resize(img, (tile_w, h), interpolation=cv2.INTER_AREA)
        for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thick, cv2.LINE_AA)
        return img

    for source, items in samples.items():
        blocks = []
        for left, teacher, disps in items:
            h = int(round(tile_w * left.shape[0] / left.shape[1]))
            lo, hi = np.percentile(teacher, 2), np.percentile(teacher, 98)

            def colour(d):
                u8 = np.clip((np.nan_to_num(d) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                return cv2.applyColorMap(u8, cv2.COLORMAP_JET)

            top = [tile(left, "left", h), tile(colour(teacher), "teacher", h)]
            bottom = [key(h, cv2.COLORMAP_JET, "far", "near", "disparity (row above)"),
                      key(h, cv2.COLORMAP_INFERNO, "0 px", "%.0f+ px" % ERROR_VIS_MAX_PX, "|error| vs teacher (below)")]
            for name in models:
                err = np.abs(np.nan_to_num(disps[name], nan=1e6) - teacher)
                w3 = 100 * (err < 3).mean()
                top.append(tile(colour(disps[name]), "%s: %.0f%% within 3px" % (name, w3), h))
                emap = np.clip(err / ERROR_VIS_MAX_PX * 255, 0, 255).astype(np.uint8)
                bottom.append(tile(cv2.applyColorMap(emap, cv2.COLORMAP_INFERNO), "%s error" % name, h))
            blocks.append(np.vstack([np.hstack(top), np.hstack(bottom),
                                     np.full((6, tile_w * len(top), 3), 255, np.uint8)]))
        name = "samples_%s.jpg" % source
        cv2.imwrite(str(out / name), np.vstack(blocks), [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(name)
    return written


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def write_outputs(out, models, lines, table, overall, paired, bands, timing, meta, sets, args, images):
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "source", "long_side", "n_frames", "within1", "within2", "within3",
                         "within3_ci_lo", "within3_ci_hi", "epe_mean_px", "epe_median_px", "p90_px",
                         "rel_depth_err_median_pct", "d1_pct", "within3_textured", "within3_textureless"])
        for name in models:
            for line in lines:
                r = table[(name,) + line]
                writer.writerow([name, line[0], line[1], r["n"]] + ["%.5f" % r[k] for k in (
                    "within1", "within2", "within3", "w3_lo", "w3_hi", "epe_mean", "epe_median", "p90",
                    "rel_median", "d1", "w3_textured", "w3_textureless")])
    (out / "metrics.json").write_text(json.dumps({
        "meta": meta, "overall": overall, "paired_vs_reference": paired, "reference": args.reference,
        "per_line": {"%s|%s|%d" % k: v for k, v in table.items()},
        "nearness_bands": {"%s|%s" % k: v for k, v in bands.items()}, "nearness_labels": NEARNESS_LABELS,
        "ms_per_frame": {"%s|%d" % k: float(np.mean(v)) for k, v in timing.items()}}, indent=2))

    ranked = sorted(models, key=lambda n: overall[n]["score"], reverse=True)
    best = ranked[0]
    big = max(args.long_sides)
    L = []
    L.append("# Model evaluation\n")
    L.append("%s. Every model runs end to end on frames it never "
             "trained on, compared pixel by pixel with the full teacher (`%s`, %d iterations)."
             % (time.strftime("%Y-%m-%d %H:%M"), meta["teacher_ckpt"], meta["iters"]))
    L.append("")
    L.append("Frames: %s (%s), long sides %s%s."
             % (", ".join("%d %s" % (len(v), k) for k, v in sets.items()), args.frames,
                ", ".join(map(str, args.long_sides)),
                "; %d frame(s) skipped for every model" % meta["skipped"] if meta["skipped"] else ""))
    L.append("\n## Verdict\n")
    o = overall[best]
    L.append("**Best: `%s`** -- %.1f%% of pixels within 3 px of the teacher (95%% CI %.1f-%.1f)."
             % (best, 100 * o["score"], 100 * o["lo"], 100 * o["hi"]))
    for name in ranked:
        if name == args.reference:
            continue
        p = paired[name]
        clear = p["lo"] > 0 or p["hi"] < 0
        L.append("- `%s` vs `%s`: %+.1f points (95%% CI %+.1f to %+.1f), P(better) %.0f%% -- %s."
                 % (name, args.reference, 100 * p["diff"], 100 * p["lo"], 100 * p["hi"], 100 * p["p_better"],
                    ("clearly better" if p["diff"] > 0 else "clearly worse") if clear else "no clear difference"))
    L.append("\n## Ranking\n")
    L.append("| rank | model | within 3 px, overall (95%% CI) | vs `%s` | P(better) | median rel. depth error | ms/frame @%d | checkpoint |"
             % (args.reference, big))
    L.append("|---|---|---|---|---|---|---|---|")
    for rank, name in enumerate(ranked, 1):
        o = overall[name]
        rel = np.mean([table[(name,) + l]["rel_median"] for l in lines])
        cmp_ = ("reference" if name == args.reference else
                "%+.1f (%+.1f to %+.1f)" % (100 * paired[name]["diff"], 100 * paired[name]["lo"], 100 * paired[name]["hi"]))
        pb = "--" if name == args.reference else "%.0f%%" % (100 * paired[name]["p_better"])
        info = meta["models"][name]
        epoch = info["epoch"] if isinstance(info["epoch"], str) else "epoch %s" % info["epoch"]
        epoch += ", %d/%d iters" % tuple(info["iters"])
        L.append("| %d | `%s` | **%.1f%%** (%.1f-%.1f) | %s | %s | %.2f%% | %.0f | %s (%s) |"
                 % (rank, name, 100 * o["score"], 100 * o["lo"], 100 * o["hi"], cmp_, pb, rel,
                    np.mean(timing[(name, big)]), Path(info["path"]).name, epoch))
    L.append("\n## Per camera and size\n")
    L.append("Within 3 px (95% CI) / median relative depth error / D1 outliers (>3 px and >5%).\n")
    L.append("| model | " + " | ".join(line_label(l).replace("\n", " ") for l in lines) + " |")
    L.append("|---|" + "---|" * len(lines))
    for name in ranked:
        cells = []
        for line in lines:
            r = table[(name,) + line]
            cells.append("%.1f%% (%.0f-%.0f) / %.2f%% / %.1f%%"
                         % (100 * r["within3"], 100 * r["w3_lo"], 100 * r["w3_hi"], r["rel_median"], r["d1"]))
        L.append("| `%s` | %s |" % (name, " | ".join(cells)))
    L.append("\n## Charts\n")
    captions = {
        "summary.png": "Agreement with the teacher per camera and size. Error bars: 95% CI over frames.",
        "nearness.png": "Relative depth error by distance band. Far (small disparity) is always hardest.",
        "texture.png": "Agreement on structure vs open water. Textured pixels are what matter for "
                       "mapping and inspection; textureless water is ill-posed for any stereo model.",
    }
    for image in images:
        caption = captions.get(image)
        if image.startswith("training_curves_"):
            caption = "Training run `%s`: loss and validation over epochs." % image[len("training_curves_"):-4]
        if image.startswith("samples_"):
            caption = ("Sample frames, %s: left, teacher, then each model (same colour scale; near = red). "
                       "Second row: |error|, black = 0, bright = %.0f px or more." % (image[8:-4], ERROR_VIS_MAX_PX))
        L.append("%s\n\n![%s](%s)\n" % (caption, image, image))
    L.append("## How to read this\n")
    L.append("- **Within 3 px**: share of pixels whose disparity is within 3 px of the teacher's. "
             "The overall score averages every camera and size equally -- the same score training "
             "selects `student_best.pth` by.")
    L.append("- **95% CI**: bootstrap over frames. Two models whose intervals overlap may still differ: "
             "the paired column compares them frame by frame, which cancels frame difficulty.")
    L.append("- **P(better)**: share of bootstrap resamples in which the model beats the reference. "
             "Above 97.5% (or below 2.5%) the difference is unlikely to be noise.")
    L.append("- **Relative depth error**: |disparity error| / disparity, which equals the relative depth "
             "error to first order -- camera independent. Its median is dominated by far, textureless "
             "water, so read it together with the nearness chart.")
    L.append("- **ms/frame**: this evaluation's own speed (eager, full precision, %d iterations), for comparing "
             "models with each other. Deployed TorchScript fp16 at 24 iterations is roughly 2-3x faster."
             % meta["iters"])
    (out / "report.md").write_text("\n".join(L) + "\n")
    return best


def as_float(value):
    """A training-log cell: blank (a column added mid-run) or missing -> NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_run_logs(models):
    logs = collections.OrderedDict()
    for path, _ in models.values():
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        directory = p if p.is_dir() else p.parent
        log = directory / "training_log.csv"
        if log.is_file() and directory.name not in logs:
            rows = list(csv.DictReader(log.open()))
            best = directory / "best.json"
            if rows and any(k.endswith("_within3") for k in rows[0]):
                logs[directory.name] = (rows, json.loads(best.read_text()) if best.is_file() else None)
    return logs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="name=checkpoint or checkpoint; a directory picks its student_best.pth, else latest epoch")
    ap.add_argument("--out", default="", help="output folder (default distillation/4_evaluations/<timestamp>)")
    ap.add_argument("--config", default=str(ROOT / "distillation/2_configs/slim_v9_2_finetune.yaml"),
                    help="training config: dataset location and the held-out split")
    ap.add_argument("--teacher", default="checkpoints/defomstereo_vitl_middlebury.pth",
                    help="the reference model (DEFOM release)")
    ap.add_argument("--image_dir", default="", help="evaluate on another dataset folder instead")
    ap.add_argument("--frames", default="heldout", choices=["heldout", "all"],
                    help="heldout: the training split's validation frames; all: every frame (new data)")
    ap.add_argument("--max_frames", type=int, default=80, help="per source (camera), evenly spaced")
    ap.add_argument("--long_sides", type=int, nargs="+", default=[512, 640])
    ap.add_argument("--iters", type=int, default=0, help="teacher iterations; default 32")
    ap.add_argument("--scale_iters", type=int, default=0)
    ap.add_argument("--reference", default="", help="model the others are compared to (default: first)")
    ap.add_argument("--samples", type=int, default=3, help="sample frames per source in the images")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = parse_models(args.models)
    args.reference = args.reference or next(iter(models))
    if args.reference not in models:
        raise SystemExit("--reference %r is not one of %s" % (args.reference, list(models)))
    cfg = load_config(args.config)
    if args.image_dir:
        cfg["paths"]["image_dir"] = args.image_dir
    cfg = resolve_paths(cfg)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out) if args.out else ROOT / "distillation/4_evaluations" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    print("Selecting frames:")
    sets = select_frames(cfg, args.frames, args.max_frames)
    print("Frames per source: %s" % {k: len(v) for k, v in sets.items()})
    per_frame, near, timing, samples, meta = run(args, models, cfg, sets, device)

    lines = [(s, L) for s in sets for L in args.long_sides if per_frame[(args.reference, s, L)]]
    if not lines:
        raise SystemExit("No frame could be evaluated")
    rng = np.random.default_rng(args.seed)
    table, overall, paired = summarise(per_frame, list(models), lines, args.reference, rng)
    bands = nearness_bands(near, list(models), list(sets))
    images = charts(out, list(models), lines, table, bands, list(sets), read_run_logs(models))
    images += sample_images(out, samples, list(models))
    best = write_outputs(out, list(models), lines, table, overall, paired, bands, timing, meta, sets, args, images)

    print("\nOverall (within 3 px, 95% CI):")
    for name in sorted(models, key=lambda n: overall[n]["score"], reverse=True):
        o = overall[name]
        extra = "" if name == args.reference else "  vs %s %+.1f (P better %.0f%%)" % (
            args.reference, 100 * paired[name]["diff"], 100 * paired[name]["p_better"])
        print("  %-12s %.1f%% (%.1f-%.1f)%s" % (name, 100 * o["score"], 100 * o["lo"], 100 * o["hi"], extra))
    print("Best: %s\nReport: %s" % (best, out / "report.md"))


if __name__ == "__main__":
    main()
