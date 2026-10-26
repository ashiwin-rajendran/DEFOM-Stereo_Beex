#!/usr/bin/env python3
"""Build a TensorRT engine from an exported ONNX model, on the device that will run it.

    python3 build_engine.py defom_multiscale_640x480.onnx                 # fp16b (default)
    python3 build_engine.py defom_multiscale_640x480.onnx --precision fp16
    python3 build_engine.py defom_multiscale_640x480.onnx --precision fp32

Uses only the TensorRT Python package that ships with JetPack. Writes
<onnx stem>_<precision>.engine and <onnx stem>_<precision>_engine.json next
to the ONNX file. An engine
only runs on the GPU and TensorRT version that built it.

Precision:
  fp32   everything fp32 (reference; slowest)
  fp16   convolutions, matrix multiplies and activations in fp16, everything
         else in fp32 -- like PyTorch mixed precision. Coordinate and
         disparity maths stay exact, and the layers the exporter measured
         above fp16's range (the .json's fp32_layers) are kept fp32.
  fp16b  (default) "balanced": fp16 inside the networks (fnet, cnet, the GRU update
         blocks, MobileViT -- incl. their normalisation and gating maths),
         fp32 only where it matters: the blocks around the fp32_layers convs
         (their normalisation sees ~1e8), the student's output heads (~1e7),
         LayerNorm layers,
         and every op at the model's top level -- the correlation lookup,
         coordinate, disparity-update and upsampling maths. Fewer fp32/fp16
         conversions: 2x faster than fp16 at the same accuracy (Orin NX).
"""

import argparse
import json
import time
from pathlib import Path

import tensorrt as trt

# Layer types allowed to run in fp16 in "fp16" mode. Everything else that
# computes on floats is pinned to fp32.
FP16_TYPES = {trt.LayerType.CONVOLUTION, trt.LayerType.DECONVOLUTION, trt.LayerType.MATRIX_MULTIPLY,
              trt.LayerType.ACTIVATION}
if hasattr(trt.LayerType, "FULLY_CONNECTED"):
    FP16_TYPES.add(trt.LayerType.FULLY_CONNECTED)
# Pure data movement: no maths, left to TensorRT.
FREE_TYPES = {trt.LayerType.CONSTANT, trt.LayerType.SHAPE, trt.LayerType.IDENTITY}
FLOAT_TYPES = (trt.float32, trt.float16)


def float_layer(layer):
    outputs = [layer.get_output(i) for i in range(layer.num_outputs)]
    inputs = [layer.get_input(i) for i in range(layer.num_inputs) if layer.get_input(i) is not None]
    return outputs and all(t.dtype in FLOAT_TYPES for t in outputs) and all(t.dtype in FLOAT_TYPES for t in inputs)


def balanced_pins(fp32_layers):
    """fp16b: name tests for the layers kept fp32 (ONNX node names, e.g.
    '/model/fnet/convd/conv/Conv', '/model/Add_45', '/model/GridSample_3')."""
    blocks = tuple(p.rstrip("/").rsplit("/", 1)[0] + "/" for p in fp32_layers)   # conv -> its whole block
    student = "/model/defomencoder/"
    backbone = "/model/defomencoder/depth_anything/backbone/"

    def pinned(layer):
        name = layer.name
        if any(name.startswith(b) for b in blocks):
            return True
        if layer.type == trt.LayerType.NORMALIZATION:                 # LayerNorm: sum of squares overflows fp16
            return True
        if name.startswith("/model/") and name.count("/") == 2:      # top-level op: coords, corr, disparity
            return True
        if name.startswith(student) and not name.startswith(backbone) and layer.type not in FP16_TYPES:
            return True                                                # student heads, denormalisation
        return False
    return pinned


def pin_balanced(network, fp32_layers):
    """fp16b: pin only the layers balanced_pins() selects. Returns (pinned, free)."""
    test = balanced_pins(fp32_layers)
    pinned = free = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in FREE_TYPES or not float_layer(layer):
            continue
        if test(layer):
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            pinned += 1
        else:
            free += 1
    return pinned, free


def pin_fp32(network, fp32_layers):
    """Pin every float-computing layer except FP16_TYPES (and the fp32_layers
    convs) to fp32. Returns (pinned, fp16-eligible) counts."""
    pinned = eligible = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in FREE_TYPES or not float_layer(layer):
            continue
        if layer.type in FP16_TYPES and not any(p in layer.name for p in fp32_layers):
            eligible += 1
            continue
        layer.precision = trt.float32
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float32)
        pinned += 1
    return pinned, eligible


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--precision", choices=["fp16b", "fp16", "fp32"], default="fp16b")
    ap.add_argument("--workspace_gb", type=float, default=4.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    meta_path = onnx_path.with_suffix(".json")
    if not meta_path.is_file():
        raise SystemExit("%s is missing: copy it from output_models together with the .onnx. It lists "
                         "the layers that overflow fp16 (without it an fp16 engine is silently wrong)."
                         % meta_path.name)
    meta = json.loads(meta_path.read_text())
    fp32_layers = meta.get("fp32_layers")
    if args.precision != "fp32" and not fp32_layers:
        raise SystemExit("%s has no fp32_layers: re-export with distillation/export_model.py" % meta_path.name)
    engine_path = onnx_path.with_name("%s_%s.engine" % (onnx_path.stem, args.precision))

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    print("TensorRT %s | parsing %s" % (trt.__version__, onnx_path.name))
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("  ", parser.get_error(i))
        raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    if args.precision != "fp32":
        if not builder.platform_has_fast_fp16:
            print("note: this GPU has no fast fp16")
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        pin = pin_balanced if args.precision == "fp16b" else pin_fp32
        pinned, eligible = pin(network, fp32_layers)
        print("layers: %d total, %d may run fp16, %d pinned fp32 (incl. %s)"
              % (network.num_layers, eligible, pinned, ", ".join(fp32_layers) or "none"))

    print("building %s engine (this can take several minutes) ..." % args.precision)
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("Engine build failed")
    build_s = time.time() - started
    tmp = Path(str(engine_path) + ".tmp")
    tmp.write_bytes(bytes(serialized))
    tmp.replace(engine_path)
    info = {"engine": engine_path.name, "onnx": onnx_path.name, "precision": args.precision,
            "tensorrt": trt.__version__, "build_seconds": round(build_s, 1),
            "width": meta.get("width"), "height": meta.get("height"), "iters": meta.get("iters"),
            "fp32_layers": fp32_layers, "source": {k: meta.get(k) for k in
                                                   ("source_run", "source_checkpoint", "source_epoch")}}
    # <engine>_engine.json: distinct from the TorchScript export's <model>_fp16.json.
    engine_path.with_name(engine_path.stem + "_engine.json").write_text(json.dumps(info, indent=2))
    print("wrote %s (%.1f MB) in %.0f s" % (engine_path, engine_path.stat().st_size / 1e6, build_s))


if __name__ == "__main__":
    main()
