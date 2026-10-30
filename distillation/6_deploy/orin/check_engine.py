#!/usr/bin/env python3
"""Run a TensorRT engine on the exporter's reference frames: accuracy and speed.

    python3 check_engine.py defom_vits_s2_release_512x384_it10s4_fp16b.engine

Compares against the fp32 PyTorch disparity stored in the matching
<model>_reference.npz, times the engine, and writes <engine>_check.json and a
<engine>_check.jpg (left | PyTorch | engine | |difference|). Needs only what
JetPack ships: tensorrt, numpy, cv2 and libcudart.
"""

import argparse
import ctypes
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

MAX_MEDIAN_PX = 0.5      # median |engine - fp32 PyTorch| disparity
MIN_WITHIN3 = 0.95       # share of pixels within 3 px of fp32 PyTorch
# Some models (e.g. DEFOM ViT-S) amplify plain fp16 rounding through their iterations:
# PyTorch's own fp16 already exceeds the limits above. When the exporter stored that
# result (disparity_fp16), a frame also passes if the engine is no further from fp32
# than PyTorch fp16 is, within a margin. On frames PyTorch fp16 already moves by more
# than FP16_UNSTABLE_PX median (e.g. murky open water: nothing to match, every model
# guesses) the margin is wider. A broken engine is off by pixels everywhere and still
# fails. (Checked: on such frames the engines agree with ViT-L as well as fp32 does.)
FP16_MEDIAN_FACTOR, FP16_MEDIAN_EXTRA, FP16_WITHIN3_DROP = 1.5, 0.1, 0.03
FP16_UNSTABLE_PX, FP16_UNSTABLE_DROP = 0.25, 0.08


class Cuda:
    """The four CUDA runtime calls needed, via ctypes (no pycuda on the SVC)."""

    H2D, D2H = 1, 2

    def __init__(self):
        for name in ("libcudart.so", "libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise SystemExit("libcudart not found (is CUDA installed?)")
        self.stream = ctypes.c_void_p()
        self._ok(self.lib.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")

    @staticmethod
    def _ok(code, what):
        if code != 0:
            raise RuntimeError("%s failed with CUDA error %d" % (what, code))

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._ok(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
        return ptr

    def to_device(self, ptr, array):
        array = np.ascontiguousarray(array)
        self._ok(self.lib.cudaMemcpy(ptr, array.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(array.nbytes),
                                     self.H2D), "cudaMemcpy H2D")

    def to_host(self, array, ptr):
        self._ok(self.lib.cudaMemcpy(array.ctypes.data_as(ctypes.c_void_p), ptr, ctypes.c_size_t(array.nbytes),
                                     self.D2H), "cudaMemcpy D2H")

    def sync(self):
        self._ok(self.lib.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def free(self, ptr):
        self.lib.cudaFree(ptr)


def colour(disp, lo, hi):
    u8 = np.clip((np.nan_to_num(disp) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--reference", default="", help="default: <model>_reference.npz beside the engine")
    ap.add_argument("--runs", type=int, default=20, help="timed runs")
    args = ap.parse_args()

    engine_path = Path(args.engine).resolve()
    stem = engine_path.stem.rsplit("_", 1)[0]            # defom_<label>_<W>x<H>
    ref_path = Path(args.reference) if args.reference else engine_path.with_name(stem + "_reference.npz")
    ref = np.load(str(ref_path))
    left, right, expected = ref["left"], ref["right"], ref["disparity"]
    torch_fp16 = ref["disparity_fp16"] if "disparity_fp16" in ref.files else None
    names = [str(n) for n in ref["frames"]]

    cuda = Cuda()
    logger = trt.Logger(trt.Logger.WARNING)
    # The ONNX parser maps InstanceNorm to a TensorRT plugin; register plugins first.
    trt.init_libnvinfer_plugins(logger, "")
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise SystemExit("Could not load %s (built by another TensorRT version or GPU?)" % engine_path)
    context = engine.create_execution_context()
    buffers, shapes = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shapes[name] = tuple(engine.get_tensor_shape(name))
        if engine.get_tensor_dtype(name) != trt.float32:
            raise SystemExit("expected float32 I/O, %s is %s" % (name, engine.get_tensor_dtype(name)))
        buffers[name] = cuda.malloc(int(np.prod(shapes[name])) * 4)
        context.set_tensor_address(name, buffers[name].value)
    out = np.empty(shapes["disparity"], np.float32)

    def infer():
        context.execute_async_v3(cuda.stream.value)
        cuda.sync()

    print("%s | TensorRT %s | inputs %s" % (engine_path.name, trt.__version__, shapes["left"]))
    stats, panels = [], []
    for k, name in enumerate(names):
        for tensor, image in (("left", left[k]), ("right", right[k])):
            cuda.to_device(buffers[tensor], image.astype(np.float32).transpose(2, 0, 1)[None])
        infer()
        cuda.to_host(out, buffers["disparity"])
        got, want = out[0, 0], expected[k]
        finite = bool(np.isfinite(got).all())
        diff = np.abs(got.astype(np.float64) - want)
        s = {"frame": name, "finite": finite, "median_px": float(np.median(diff)),
             "p99_px": float(np.percentile(diff, 99)), "within1": float((diff < 1).mean()),
             "within3": float((diff < 3).mean()), "disparity_median_px": float(np.median(want))}
        if torch_fp16 is not None:
            d16 = np.abs(torch_fp16[k].astype(np.float64) - want)
            s["pytorch_fp16_median_px"], s["pytorch_fp16_within3"] = float(np.median(d16)), float((d16 < 3).mean())
        stats.append(s)
        print("  %-14s finite %-5s | vs fp32 PyTorch: median %.3f px  p99 %.2f px  within 1px %.1f%%  3px %.1f%%"
              % (name, finite, s["median_px"], s["p99_px"], 100 * s["within1"], 100 * s["within3"])
              + ("" if torch_fp16 is None else "  (PyTorch fp16: median %.3f px, 3px %.1f%%)"
                 % (s["pytorch_fp16_median_px"], 100 * s["pytorch_fp16_within3"])))
        lo, hi = np.percentile(want, 2), np.percentile(want, 98)
        err = cv2.applyColorMap(np.clip(diff / 3.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        panels.append(np.hstack([cv2.cvtColor(left[k], cv2.COLOR_RGB2BGR), colour(want, lo, hi),
                                 colour(got, lo, hi), err]))

    for _ in range(3):
        infer()
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        infer()
        times.append(1000 * (time.perf_counter() - start))
    latency = float(np.median(times))
    print("  latency: median %.1f ms (%.2f Hz), p90 %.1f ms" % (latency, 1000 / latency, np.percentile(times, 90)))

    result = {"engine": engine_path.name, "tensorrt": trt.__version__, "latency_ms_median": round(latency, 1),
              "latency_ms_p90": round(float(np.percentile(times, 90)), 1), "hz": round(1000 / latency, 2),
              "accuracy_vs_fp32_pytorch": stats}
    check_json = engine_path.with_name(engine_path.stem + "_check.json")
    check_json.write_text(json.dumps(result, indent=2))
    scale = min(1.0, 2400 / panels[0].shape[1])
    cv2.imwrite(str(engine_path.with_name(engine_path.stem + "_check.jpg")),
                cv2.resize(np.vstack(panels), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    print("  wrote %s (+ .jpg: left | PyTorch | engine | |diff| 0-3 px)" % check_json.name)
    for ptr in buffers.values():
        cuda.free(ptr)
    # Same limits as the exporter's fp16 check: a correct engine is within ~0.1 px.
    def ok(s):
        if not s["finite"]:
            return False
        if s["median_px"] <= MAX_MEDIAN_PX and s["within3"] >= MIN_WITHIN3:
            return True
        if "pytorch_fp16_median_px" not in s:
            return False
        unstable = s["pytorch_fp16_median_px"] > FP16_UNSTABLE_PX
        drop = max(FP16_UNSTABLE_DROP if unstable else FP16_WITHIN3_DROP, 0.5 * (1 - s["pytorch_fp16_within3"]))
        return (s["median_px"] <= FP16_MEDIAN_FACTOR * s["pytorch_fp16_median_px"] + FP16_MEDIAN_EXTRA
                and s["within3"] >= s["pytorch_fp16_within3"] - drop)
    bad = [s["frame"] for s in stats if not ok(s)]
    if bad:
        raise SystemExit("ACCURACY FAIL on %s (limits: median <= %.1f px, >= %.0f%% within 3 px): this "
                         "engine is wrong -- check that the model's .json was present when it was built."
                         % (", ".join(bad), MAX_MEDIAN_PX, 100 * MIN_WITHIN3))
    if any(s["median_px"] > MAX_MEDIAN_PX or s["within3"] < MIN_WITHIN3 for s in stats):
        print("  ACCURACY OK (as close to fp32 as PyTorch's own fp16 is; this model is fp16-sensitive)")
    else:
        print("  ACCURACY OK")


if __name__ == "__main__":
    main()
