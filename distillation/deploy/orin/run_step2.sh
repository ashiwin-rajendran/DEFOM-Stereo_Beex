#!/bin/bash
# Build and check a TensorRT engine for the exported ONNX models in a folder,
# smallest size first.
#
#   bash run_step2.sh <folder>                        # every size
#   bash run_step2.sh <folder> 640x480 800x600        # only these sizes, in this order
#   PRECISIONS="fp16 fp32" bash run_step2.sh <folder> # also build fp32 (slower)
#
# Writes <folder>/step2_results.txt (send this back) plus per-engine
# _check.json/_check.jpg. Existing engines are reused; delete one to rebuild.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:?usage: run_step2.sh <folder with the exported models> [WxH ...]}"
shift
PRECISIONS="${PRECISIONS:-fp16}"
LOG="$DIR/step2_results.txt"

if [ $# -gt 0 ]; then
  MODELS=()
  for size in "$@"; do
    found=("$DIR"/*_"$size".onnx)
    [ -f "${found[0]}" ] || { echo "no ONNX model for $size in $DIR"; exit 1; }
    MODELS+=("${found[0]}")
  done
else
  # Every model, ordered by pixel count (the size is the last _WxH of the name).
  mapfile -t MODELS < <(for f in "$DIR"/*.onnx; do
    s="${f##*_}"; s="${s%.onnx}"; echo "$(( ${s%x*} * ${s#*x} )) $f"; done | sort -n | cut -d' ' -f2-)
fi

{
  echo "===== $(date) on $(hostname)"
  cat /proc/device-tree/model 2>/dev/null; echo
  nvpmodel -q 2>/dev/null | head -1
  python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)"
  for onnx in "${MODELS[@]}"; do
    if [ ! -f "${onnx%.onnx}.json" ] || [ ! -f "${onnx%.onnx}_reference.npz" ]; then
      echo; echo "----- $(basename "$onnx"): SKIPPED, its .json or _reference.npz is missing"; continue
    fi
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
