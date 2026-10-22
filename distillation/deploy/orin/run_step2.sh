#!/bin/bash
# Build and check a TensorRT engine for every exported ONNX model in a folder.
#
#   bash run_step2.sh <folder with *.onnx, *.json, *_reference.npz>
#   PRECISIONS="fp16 fp32" bash run_step2.sh <folder>      # also build fp32 (slower)
#
# Writes <folder>/step2_results.txt (send this back) plus per-engine
# _check.json/_check.jpg. Existing engines are reused; delete one to rebuild.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:?usage: run_step2.sh <folder with the exported models>}"
PRECISIONS="${PRECISIONS:-fp16}"
LOG="$DIR/step2_results.txt"

{
  echo "===== $(date) on $(hostname)"
  cat /proc/device-tree/model 2>/dev/null; echo
  nvpmodel -q 2>/dev/null | head -1
  python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)"
  for onnx in "$DIR"/*.onnx; do
    for p in $PRECISIONS; do
      engine="${onnx%.onnx}_${p}.engine"
      echo; echo "----- $(basename "$onnx") $p"
      if [ ! -f "$engine" ]; then
        python3 "$HERE/build_engine.py" "$onnx" --precision "$p" || { echo "BUILD FAILED"; continue; }
      fi
      python3 "$HERE/check_engine.py" "$engine" || echo "CHECK FAILED"
    done
  done
  echo; echo "===== done $(date)"
} 2>&1 | tee -a "$LOG"
echo "Results: $LOG"
