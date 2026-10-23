# On the Orin (SVC)

Tools copied to the Jetson. They use only what JetPack 6 ships (python3,
TensorRT 8.6, numpy, OpenCV, libcudart): nothing to install, no sudo, nothing
outside the folder they run in is changed.

| file | does |
|---|---|
| `build_engine.py` | ONNX -> TensorRT `.engine` for this GPU (fp16 or fp32) |
| `check_engine.py` | runs an engine on the exporter's reference frames: accuracy vs fp32 PyTorch, latency |
| `run_step2.sh` | both of the above for the models in a folder (or chosen sizes), smallest first; writes `step2_results.txt` |
| `check_rectification.py` | is the raw 1600x1200 stream rectified? (with or without the `.dwecal` maps) |

## 1. Engines: speed and accuracy

On the PC (training repo):

```bash
python3 distillation/export_model.py --ckpt distillation/checkpoints/<run> --format onnx \
    --size 640x480 --size 800x600 --size 1024x768
scp -r distillation/output_models/<label>_<epoch> distillation/deploy/orin dwe@192.168.4.5:~/defom/
```

(The multiscale student is refused at 1600x1200: it was trained on long sides
of 512-1024 and is numerically unstable there -- two exact-fp32 runs differ by
0.6 px median -- while the teacher is stable. It needs fine-tuning at that scale.)

Copy the whole folder: every model needs its `.onnx`, `.json` (it lists the
layers that must stay fp32) and `_reference.npz` side by side. Without the
`.json` the builder refuses; it used to build an engine that ran but was wrong.

On the SVC (stop DWE's stream so the GPU is free), smallest size first:

```bash
cd ~/defom
nohup bash orin/run_step2.sh <label>_<epoch> 640x480 800x600 > step2.log 2>&1 &   # chosen sizes, in order
nohup bash orin/run_step2.sh <label>_<epoch> > step2.log 2>&1 &                   # every size, smallest first
PRECISIONS=fp32 bash orin/run_step2.sh <label>_<epoch> 640x480                    # optional fp32 reference
tail -f step2.log
```

Each build takes a long time on the Orin NX (1024x768: 45 min). Per model it
writes `<model>_fp16.engine`, `<model>_fp16_engine.json` (build info) and
`<model>_fp16_check.json/.jpg` (left | PyTorch | engine | difference); each
check ends with `ACCURACY OK` or `ACCURACY FAIL`. Send back
`<label>_<epoch>/step2_results.txt` and a `_check.jpg`.

**fp16** runs convolutions, matrix multiplies and activations in fp16 and
everything else in fp32, like PyTorch mixed precision. The four
convolutions where the depth features enter the stereo core reach ~1e8,
beyond fp16's 65504; the exporter measures them and lists them in the `.json`
(`fp32_layers`), and they stay fp32. Measured on an RTX 4070 with TensorRT
8.6.1 at 640x480: median 0.02-0.12 px from fp32 PyTorch, 99.6-99.9% of pixels
within 3 px, 129 ms.

An engine only runs on the GPU and TensorRT version that built it. A runtime
loading it must register TensorRT's plugins first
(`init_libnvinfer_plugins` / `initLibNvInferPlugins`): InstanceNorm is a plugin.

## 2. Is the raw 1600x1200 stream rectified?

```bash
scp ~/DWE/RTPSender/calibration/e3d1008.dwecal dwe@192.168.4.5:~/defom/
# on the SVC, cameras UNDERWATER, looking at a still textured scene 1-3 m away:
python3 orin/check_rectification.py --dwecal e3d1008.dwecal --out rect_check
```

It reports the vertical offset between matched left/right features (0 for a
rectified pair) on the raw frames and after the `.dwecal` maps, and saves the
pairs (`rect_check/raw_*.png` are real 1600x1200 test frames for later). The
calibration is for water: in air even a correct calibration does not
rectify. It also prints both cameras' serial numbers: confirm the `.dwecal`
belongs to this camera (`e3d1008`).
