# On the Orin (SVC)

Tools copied to the Jetson. They use only what JetPack 6 ships (python3,
TensorRT 8.6, numpy, OpenCV, libcudart): nothing to install, no sudo, nothing
outside the folder they run in is changed.

| file | does |
|---|---|
| `build_engine.py` | ONNX -> TensorRT `.engine` for this GPU (fp16 or fp32) |
| `check_engine.py` | runs an engine on the exporter's reference frames: accuracy vs fp32 PyTorch, latency |
| `run_step2.sh` | both of the above for every ONNX model in a folder; writes `step2_results.txt` |
| `check_rectification.py` | is the raw 1600x1200 stream rectified? (with or without the `.dwecal` maps) |

## 1. Engines: speed and accuracy

On the PC (training repo):

```bash
python3 distillation/export_model.py --ckpt distillation/checkpoints/<run> --format onnx \
    --size 640x480 --size 800x600 --size 1024x768 --size 1600x1200
scp -r distillation/output_models/<label>_<epoch> distillation/deploy/orin dwe@192.168.4.5:~/defom/
```

On the SVC (stop DWE's stream so the GPU is free):

```bash
cd ~/defom
bash orin/run_step2.sh <label>_<epoch>                        # fp16 engines, all sizes
PRECISIONS=fp32 bash orin/run_step2.sh <label>_<epoch>        # optional fp32 reference
```

Each engine build takes several minutes (longer for 1600x1200). Send back
`<label>_<epoch>/step2_results.txt`; the `*_check.jpg` images show left |
PyTorch | engine | difference.

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
