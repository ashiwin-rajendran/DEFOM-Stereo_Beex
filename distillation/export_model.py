#!/usr/bin/env python3
"""Export a distilled student (.pth) for deployment.

    # TorchScript for the ROS node defom_stereo_pointcloud (LibTorch, this PC)
    python3 distillation/export_model.py --ckpt distillation/checkpoints/<run> --format torchscript --size 640x480 --half

    # ONNX for TensorRT on the Orin (the .engine is built there: deploy/orin/)
    python3 distillation/export_model.py --ckpt distillation/checkpoints/<run> --format onnx \\
        --size 1600x1200 --size 1024x768 --size 800x600 --size 640x480

--ckpt takes a run folder (its student_best.pth) or a .pth file. Artifacts go
to distillation/output_models/<label>_<epoch>/:

    defom_<label>_<W>x<H>_<fp16|fp32>.pt  + .json    TorchScript
    defom_<label>_<W>x<H>.onnx            + .json    ONNX (fp32 weights; precision is chosen
    defom_<label>_<W>x<H>_reference.npz              when the engine is built) + test frames
                                                     and expected output for the Orin check

The model takes left and right RGB images (float, 0-255, NCHW) at exactly
W x H and returns disparity in pixels at W x H. Every artifact is checked on
real frames against the fp32 PyTorch model and is not written if it disagrees.
Add a format by adding an export_<format>() function.
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
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.defom_stereo import DEFOMStereo
import core.corr as defom_corr
from defom_student import load_student

OUTPUT_ROOT = ROOT / "distillation" / "output_models"
DATASETS = ROOT / "distillation" / "datasets"
# Short names for the trained runs (others use their folder name).
RUN_LABELS = {
    "underwater_vitl_to_student": "sceneflow",
    "underwater_vitl_middlebury_to_student": "middlebury",
    "underwater_vitl_middlebury_to_student_multiscale": "multiscale",
}
# Limits on disagreement with fp32 PyTorch, as median |disparity difference| (px).
# Two exact-fp32 runtimes differ by up to ~0.1 px on some frames: 24 recurrent
# iterations amplify summation-order noise in ill-posed (textureless) areas. A
# broken export is off by pixels everywhere.
MAX_MEDIAN_PX = {"fp32": 0.25, "fp16": 0.5}
# Layers whose activations exceed this (fp16 max 65504) are marked fp32-only for TensorRT.
FP16_SAFE_LIMIT = 8192.0


# --------------------------------------------------------------------------
# model and data
# --------------------------------------------------------------------------


class DeployWrapper(torch.nn.Module):
    """pad to a multiple of 32 -> DEFOM -> crop: callers deal in W x H only."""

    def __init__(self, model, iters, scale_iters, height, width):
        super().__init__()
        self.model = model
        self.iters, self.scale_iters = iters, scale_iters
        self.pad_h, self.pad_w = (32 - height % 32) % 32, (32 - width % 32) % 32

    def forward(self, left, right):
        if self.pad_h or self.pad_w:
            left = F.pad(left, [0, self.pad_w, 0, self.pad_h], mode="replicate")
            right = F.pad(right, [0, self.pad_w, 0, self.pad_h], mode="replicate")
        disp = self.model(left, right, iters=self.iters, scale_iters=self.scale_iters, test_mode=True)
        if self.pad_h or self.pad_w:
            disp = disp[..., :disp.shape[-2] - self.pad_h, :disp.shape[-1] - self.pad_w]
        return disp


def build_model(ckpt):
    """DEFOM with the student swapped in for the teacher encoder, on the CPU."""
    student, ck, ckpt_path = load_student(ckpt, "cpu")
    recorded = Path(ck["config"]["paths"]["teacher_ckpt"])
    teacher = recorded if recorded.is_absolute() else ROOT / recorded
    if not teacher.is_file():
        teacher = ROOT / "checkpoints" / recorded.name
    if not teacher.is_file():
        raise SystemExit("Stereo core %s not found (recorded as %s)" % (teacher, recorded))
    model = DEFOMStereo(argparse.Namespace(**ck["config"]["arch"]))
    state = torch.load(str(teacher), map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    del state, model.defomencoder.depth_anything
    model.defomencoder.depth_anything = student
    model.args.mixed_precision = False
    print("checkpoint  : %s (epoch %s)" % (ckpt_path, ck.get("epoch")))
    print("stereo core : %s" % teacher)
    return model.eval(), ck, ckpt_path


def output_dir(ckpt_path, ck, label, root=OUTPUT_ROOT):
    label = label or RUN_LABELS.get(ckpt_path.parent.name, ckpt_path.parent.name)
    epoch = ck.get("epoch")
    tag = ("ep%d" % epoch) if isinstance(epoch, int) else str(epoch).replace("epoch ", "ep").replace(" ", "")
    return label, Path(root) / ("%s_%s" % (label, tag))


def load_frames(width, height, count, images=()):
    """Real side-by-side frames, each eye resized to width x height (RGB uint8).
    Frames whose eye aspect matches the export size are preferred."""
    paths = [Path(p) for p in images]
    if not paths:
        found = sorted(p for p in DATASETS.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not found:
            raise SystemExit("No frames under %s; pass --images" % DATASETS)
        target = width / height
        aspect_ok = []
        for p in found[::max(1, len(found) // 200)]:
            im = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8)
            if im is not None and abs((im.shape[1] / 2) / im.shape[0] / target - 1) < 0.02:
                aspect_ok.append(p)
        pool = aspect_ok or found
        paths = pool[::max(1, len(pool) // count)][:count]
    frames = []
    for p in paths:
        image = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if image is None or image.shape[1] % 2:
            continue
        half = image.shape[1] // 2
        eyes = []
        for eye in (image[:, :half], image[:, half:]):
            interp = cv2.INTER_AREA if eye.shape[1] > width else cv2.INTER_LINEAR
            eyes.append(cv2.cvtColor(cv2.resize(eye, (width, height), interpolation=interp), cv2.COLOR_BGR2RGB))
        frames.append((p.name, eyes[0], eyes[1]))
    if not frames:
        raise SystemExit("No readable side-by-side frames")
    return frames


def to_tensor(rgb, device, dtype=torch.float32):
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device=device, dtype=dtype).contiguous()


def compare(out, ref):
    out, ref = np.asarray(out, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    finite = bool(np.isfinite(out).all())
    diff = np.abs(out - ref) if finite else np.full_like(ref, np.inf)
    return {"finite": finite, "median_px": float(np.median(diff)), "p99_px": float(np.percentile(diff, 99)),
            "within1": float((diff < 1).mean()), "disparity_median_px": float(np.median(ref))}


def check(stats, limit, what):
    for s in stats:
        print("  %-14s finite %-5s | vs fp32 PyTorch: median %.4f px  p99 %.3f px  within 1px %.1f%%"
              % (s["frame"], s["finite"], s["median_px"], s["p99_px"], 100 * s["within1"]))
    worst = max(s["median_px"] for s in stats)
    # NaN fails every comparison, so test for the passing case.
    if not all(s["finite"] for s in stats) or not worst <= limit:
        raise SystemExit("%s is non-finite or diverges from fp32 PyTorch (worst median %.4f px > %.2f): "
                         "nothing written." % (what, worst, limit))


def write_json(path, meta):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(str(tmp), str(path))


def source_meta(ckpt_path, ck, label, width, height, args):
    return {"label": label, "source_run": ckpt_path.parent.name, "source_checkpoint": ckpt_path.name,
            "source_epoch": ck.get("epoch"), "val_score": ck.get("val_score"),
            "teacher_ckpt": ck["config"]["paths"]["teacher_ckpt"], "backbone": ck["student_cfg"]["backbone"],
            "width": width, "height": height, "iters": args.iters, "scale_iters": args.scale_iters,
            "input_layout": "NCHW", "input_order": "RGB", "input_range": "0-255",
            "output": "disparity in pixels at width x height (left image)",
            "torch_version": torch.__version__}


# --------------------------------------------------------------------------
# TorchScript (for LibTorch: the defom_stereo_pointcloud ROS node)
# --------------------------------------------------------------------------


def export_torchscript(model, ck, ckpt_path, label, out_dir, width, height, args):
    device = torch.device(args.device)
    model.to(device)
    wrapper = DeployWrapper(model, args.iters, args.scale_iters, height, width).eval()
    dtype = torch.half if args.half else torch.float
    precision = "fp16" if args.half else "fp32"
    frames = load_frames(width, height, args.val_count, args.images)
    print("\n[torchscript %dx%d %s] tracing on %s" % (width, height, precision, device))
    with torch.no_grad():
        model.args.mixed_precision = False
        references = [wrapper(to_tensor(l, device), to_tensor(r, device)).float().cpu() for _, l, r in frames]
        model.args.mixed_precision = args.half   # fp16 where PyTorch's autocast allows, as validated
        example = (to_tensor(frames[0][1], device, dtype), to_tensor(frames[0][2], device, dtype))
        traced = torch.jit.trace(wrapper, example, check_trace=False)
        model.args.mixed_precision = False
        frozen = False
        try:
            traced = torch.jit.freeze(traced)
            frozen = True
        except RuntimeError as exc:   # known for this model on torch 2.x
            print("  note: torch.jit.freeze not applicable (%s)" % str(exc).split("\n")[0][:70])
        stats = []
        for (name, l, r), ref in zip(frames, references):
            out = traced(to_tensor(l, device, dtype), to_tensor(r, device, dtype)).float().cpu()
            stats.append(dict(compare(out.numpy(), ref.numpy()), frame=name))
        check(stats, MAX_MEDIAN_PX[precision], "Traced module")
        latency = time_it(lambda: traced(*example), device)
    print("  latency on this device: %.0f ms (%.1f Hz)" % (latency, 1000 / latency))

    name = "defom_%s_%dx%d_%s" % (label, width, height, precision)
    pt_path = out_dir / (name + ".pt")
    torch.jit.save(traced, str(pt_path) + ".tmp")
    meta = dict(source_meta(ckpt_path, ck, label, width, height, args), name=name, format="torchscript",
                dtype="float16" if args.half else "float32", device_traced_on=device.type,
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                frozen=frozen, latency_ms_on_export_device=round(latency, 1),
                validation_vs_fp32=stats,
                notes="Traced at this size and device; run it on the same device type with LibTorch "
                      "compatible with torch %s." % torch.__version__)
    os.replace(str(pt_path) + ".tmp", str(pt_path))
    write_json(out_dir / (name + ".json"), meta)
    print("  wrote %s (%.1f MB) + .json" % (pt_path, pt_path.stat().st_size / 1e6))
    model.cpu()


def time_it(fn, device, runs=6):
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(1000 * (time.time() - start))
    return float(np.median(times[1:]))


# --------------------------------------------------------------------------
# ONNX (for TensorRT on the Orin)
# --------------------------------------------------------------------------


def corr_matmul(fmap1, fmap2):
    """CorrBlock1D.corr with the einsum written as a batched matmul (same maths)."""
    B, D, H, W1 = fmap1.shape
    W2 = fmap2.shape[3]
    corr = torch.matmul(fmap1.permute(0, 2, 3, 1), fmap2.permute(0, 2, 1, 3))   # B, H, W1, W2
    return corr.reshape(B, H, W1, 1, W2) / float(D) ** 0.5


def upsample_flow_conv(self, flow, mask):
    """DEFOMStereo.upsample_flow with F.unfold written as a one-hot 3x3 conv (same maths)."""
    N, D, H, W = flow.shape
    factor = 2 ** self.args.n_downsample
    mask = torch.softmax(mask.view(N, 1, 9, factor, factor, H, W), dim=2)
    eye = torch.eye(9, dtype=flow.dtype, device=flow.device).view(9, 1, 3, 3)
    up_flow = F.conv2d((factor * flow).reshape(N * D, 1, H, W), eye, padding=1)
    up_flow = up_flow.view(N, D, 9, 1, 1, H, W)
    up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, D, factor * H, factor * W)


class OnnxRewrites:
    """Within this block, DEFOM uses ops TensorRT 8.6 supports natively. The
    maths is identical (checked against the unmodified model on every export)."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.saved_corr = defom_corr.CorrBlock1D.corr
        defom_corr.CorrBlock1D.corr = staticmethod(corr_matmul)
        self.model.upsample_flow = types.MethodType(upsample_flow_conv, self.model)
        return self

    def __exit__(self, *exc):
        defom_corr.CorrBlock1D.corr = self.saved_corr
        del self.model.upsample_flow


def fp16_unsafe_layers(wrapper, frames, device):
    """Conv/linear layers whose input or output exceeds FP16_SAFE_LIMIT on these
    frames, as ONNX/TensorRT layer-name prefixes (e.g. '/model/fnet/convd/conv/')."""
    peaks, hooks = {}, []
    for name, module in wrapper.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            def hook(mod, inp, out, name=name):
                peak = max(float(inp[0].abs().max()), float(out.abs().max()))
                peaks[name] = max(peaks.get(name, 0.0), peak)
            hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        for _, l, r in frames:
            wrapper(to_tensor(l, device), to_tensor(r, device))
    for h in hooks:
        h.remove()
    unsafe = sorted((n for n, v in peaks.items() if v > FP16_SAFE_LIMIT), key=lambda n: -peaks[n])
    return ["/" + n.replace(".", "/") + "/" for n in unsafe], {n: round(peaks[n], 1) for n in unsafe}


def export_onnx(model, ck, ckpt_path, label, out_dir, width, height, args):
    import onnx
    import onnxruntime as ort

    device = torch.device(args.device)
    model.to(device)
    wrapper = DeployWrapper(model, args.iters, args.scale_iters, height, width).eval()
    frames = load_frames(width, height, args.val_count, args.images)
    print("\n[onnx %dx%d] opset %d, iters %d/%d" % (width, height, args.opset, args.iters, args.scale_iters))
    with torch.no_grad():
        references = [wrapper(to_tensor(l, device), to_tensor(r, device)).float().cpu().numpy()
                      for _, l, r in frames]
    fp32_layers, peaks = fp16_unsafe_layers(wrapper, frames, device)
    print("  fp16-unsafe layers (kept fp32 by the Orin builder): %s" % peaks)

    name = "defom_%s_%dx%d" % (label, width, height)
    onnx_path = out_dir / (name + ".onnx")
    tmp_path = str(onnx_path) + ".tmp"
    with OnnxRewrites(model), torch.no_grad():
        # The rewrites must not change the result.
        _, l, r = frames[0]
        rewritten = wrapper(to_tensor(l, device), to_tensor(r, device)).float().cpu().numpy()
        same = compare(rewritten, references[0])
        print("  rewrites vs original model: median %.2e px, p99 %.2e px" % (same["median_px"], same["p99_px"]))
        if not same["median_px"] <= 1e-3:
            raise SystemExit("ONNX rewrites changed the output: nothing written")
        example = (to_tensor(l, device), to_tensor(r, device))
        started = time.time()
        torch.onnx.export(wrapper, example, tmp_path, opset_version=args.opset, do_constant_folding=True,
                          input_names=["left", "right"], output_names=["disparity"])
    model.cpu()
    graph = onnx.load(tmp_path)
    onnx.checker.check_model(graph)
    print("  exported in %.0f s: %d nodes, %.1f MB" % (time.time() - started, len(graph.graph.node),
                                                      os.path.getsize(tmp_path) / 1e6))
    del graph

    # The exported graph, run by onnxruntime (CPU), against the fp32 PyTorch model.
    session = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    stats = []
    for (fname, l, r), ref in zip(frames, references):
        out = session.run(None, {"left": to_tensor(l, "cpu").numpy(), "right": to_tensor(r, "cpu").numpy()})[0]
        stats.append(dict(compare(out, ref), frame=fname))
    del session
    try:
        check(stats, MAX_MEDIAN_PX["fp32"], "ONNX graph")
    except SystemExit:
        os.remove(tmp_path)
        raise

    np.savez_compressed(out_dir / (name + "_reference.npz"),
                        left=np.stack([f[1] for f in frames]), right=np.stack([f[2] for f in frames]),
                        disparity=np.concatenate(references)[:, 0], frames=np.array([f[0] for f in frames]))
    meta = dict(source_meta(ckpt_path, ck, label, width, height, args), name=name, format="onnx",
                onnx=onnx_path.name, reference=name + "_reference.npz", opset=args.opset,
                inputs={"left": [1, 3, height, width], "right": [1, 3, height, width]},
                outputs={"disparity": [1, 1, height, width]},
                fp32_layers=fp32_layers, fp32_layer_peaks=peaks,
                rewrites=["CorrBlock1D.corr: einsum -> matmul", "upsample_flow: unfold -> one-hot conv"],
                validation_onnxruntime_vs_fp32=stats,
                notes="Build the TensorRT engine on the target with deploy/orin/build_engine.py; it keeps "
                      "fp32_layers and all non-conv maths in fp32.")
    os.replace(tmp_path, str(onnx_path))
    write_json(out_dir / (name + ".json"), meta)
    print("  wrote %s + .json + _reference.npz" % onnx_path)


EXPORTERS = {"torchscript": export_torchscript, "onnx": export_onnx}


def parse_size(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 640x480")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="run folder (uses student_best.pth) or a .pth")
    ap.add_argument("--format", nargs="+", default=["torchscript"], choices=sorted(EXPORTERS) + ["all"])
    ap.add_argument("--size", type=parse_size, action="append", help="WxH, repeatable (default 640x480)")
    ap.add_argument("--half", action="store_true", help="torchscript: fp16 (ONNX precision is set on the Orin)")
    ap.add_argument("--iters", type=int, default=24, help="refinement iterations baked into the export")
    ap.add_argument("--scale_iters", type=int, default=8)
    ap.add_argument("--device", default="cuda", help="torchscript: the deployment device type")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--label", default="", help="artifact label (default: short run name)")
    ap.add_argument("--out_root", default=str(OUTPUT_ROOT), help="parent of the <label>_<epoch> output folder")
    ap.add_argument("--images", nargs="*", default=[], help="side-by-side frames for validation")
    ap.add_argument("--val_count", type=int, default=3, help="validation frames per export")
    args = ap.parse_args()
    if args.scale_iters > args.iters:
        raise SystemExit("--scale_iters cannot exceed --iters")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    formats = sorted(EXPORTERS) if "all" in args.format else args.format
    # References must be true fp32: PyTorch's default TF32 convolutions differ
    # from exact fp32 by ~0.1 px after 24 iterations.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    model, ck, ckpt_path = build_model(args.ckpt)
    label, out_dir = output_dir(ckpt_path, ck, args.label, args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("output      : %s" % out_dir)
    for width, height in args.size or [(640, 480)]:
        for fmt in formats:
            EXPORTERS[fmt](model, ck, ckpt_path, label, out_dir, width, height, args)


if __name__ == "__main__":
    main()
