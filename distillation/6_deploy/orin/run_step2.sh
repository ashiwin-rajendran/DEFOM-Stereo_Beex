#!/bin/bash
# Build and check a TensorRT engine for the exported ONNX models in a folder,
# smallest size first.
#
#   bash run_step2.sh <folder>                        # every model, smallest first
#   bash run_step2.sh <folder> 640x480 800x600_it16s6 # models whose name contains these, in this order
#   PRECISIONS="fp16b fp32" bash run_step2.sh <folder> # also an fp32 reference (slower)
#
# Writes <folder>/step2_results.txt (send this back) plus per-engine
# _check.json/_check.jpg. Existing engines are reused; delete one to rebuild.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:?usage: run_step2.sh <folder with the exported models> [WxH ...]}"
shift
PRECISIONS="${PRECISIONS:-fp16b}"
LOG="$DIR/step2_results.txt"

if [ $# -gt 0 ]; then
  MODELS=()
  for pattern in "$@"; do
    found=("$DIR"/*"$pattern"*.onnx)
    [ -f "${found[0]}" ] || { echo "no ONNX model matching '$pattern' in $DIR"; exit 1; }
    MODELS+=("${found[@]}")
  done
else
  # Every model, ordered by pixel count (the WxH in its name), then name.
  mapfile -t MODELS < <(for f in "$DIR"/*.onnx; do
    s=$(basename "$f" | grep -oE '[0-9]+x[0-9]+' | head -1)
    echo "$(( ${s%x*} * ${s#*x} )) $f"; done | sort -n -k1,1 -k2 | cut -d' ' -f2-)
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
