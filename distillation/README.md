# DEFOM-Stereo on the Orin NX: train, evaluate, deploy, compare

Live stereo depth from the DWE cameras on the SVC (Jetson Orin NX), at ~6 Hz, delivered at
1600x1200 to the AIB over ROS. Two models are deployed (TensorRT fp16b on the Orin NX, measured):

| model | checkpoint / export folder | inference | Orin NX | strength |
|---|---|---|---|---|
| **slim v9** | `1_slim_model` | 800x600, 12/4 iterations | 172 ms, 5.8 Hz | best on dive sites like its training data; full 800x600 detail |
| **ViT-S s2** | `2_fast_vits_model` | 512x384, 10/4 iterations | 165 ms, 6.05 Hz | holds up on unfamiliar scenes (tank, lab) |

- **Slim v9:** a slim stereo core plus a 0.75x MobileViT encoder student, distilled from
  DEFOM ViT-L.
- **ViT-S s2:** DEFOM's own ViT-S release with every weight unchanged; only its feature
  networks start at 1/2 resolution.
- **The teacher and quality reference for both** is DEFOM ViT-L Middlebury
  (`defomstereo_vitl_middlebury.pth`). No ground truth is needed: the teacher's disparity on
  your own rectified footage is the training target.

Results: every model tested, with figures, and why the two models are built this way:
[8_results/RESULTS.md](8_results/RESULTS.md). One-page status: [8_results/BRIEF_UPDATE.md](8_results/BRIEF_UPDATE.md).

**Contents**
1. [Layout](#layout)
2. [Setup](#1-setup): [Thor](#11-jetson-thor-training), [PC](#12-pc-export-evaluation-rosbags), [files](#13-files-to-copy-in)
3. [Data](#2-data): record, labels
4. [Train](#3-train): [slim v9](#31-slim-v9), [ViT-S s2](#32-vit-s-s2)
5. [Evaluate](#4-evaluate)
6. [Export](#5-export-onnx)
7. [Deploy on the Orin](#6-deploy-on-the-orin): [engines](#61-build-and-check-the-engine), [rectification](#62-rectification-check), [SVC service](#63-svc-defom_svc), [AIB receiver](#64-aib-defom_receiverpy), [test without the vehicle](#65-test-without-the-vehicle)
8. [Compare models on rosbags (PC)](#7-compare-models-on-rosbags-pc)
9. [Config reference](#config-reference) · [Troubleshooting](#troubleshooting)

---

## Layout

```
distillation/                  the folders are numbered in workflow order
  README.md                    this guide
  common.py                    config, stereo frames, the ViT-L teacher, checkpoint/log writing
  slim_stereo.py               SlimDEFOM and its presets (v9, vits_s2); --make builds vits_s2 from the release
  defom_student.py             the MobileViT encoder student (slim v9's encoder)
  mobilevit.py                 MobileViT backbone (self-contained copy)
  teacher_labels.py            the teacher's disparity for every frame (labels, shared by both models)
  train_slim.py                train slim v9
  train_vits.py                fine-tune ViT-S s2, anchored to the ViT-S release
  evaluate.py                  compare models against the teacher: report + charts
  export_model.py              ONNX for TensorRT (and TorchScript)

  1_datasets/                  (git-ignored)
      underwater/<session>/        training frames
      probe/<set>/                 new scenes, validation only
      teacher_labels_800/          teacher_labels.py output
  2_configs/
      slim_v9_1_from_vitl.yaml     slim v9 from scratch: a new stereo core on 1_slim_model's encoder
      slim_v9_2_finetune.yaml      slim v9 fine-tune: from 1_slim_model (or step 1's run)
      vits_s2_finetune.yaml        ViT-S s2 fine-tune: from 2_fast_vits_model
  3_checkpoints/<run>/         one folder per run (git-ignored); 1_slim_model, 2_fast_vits_model deployed
      student_best.pth             the best validated epoch  <- use this one
      best.json, student_epoch####.pth, training_log.csv
  4_evaluations/<name>/        evaluate.py reports (git-ignored)
  5_output_models/<run>/       exports + Orin engines, named like 3_checkpoints/<run> (git-ignored)
  6_deploy/
      orin/                        copied to the SVC: build + check TensorRT engines, rectification check
      defom_svc/                   defom_svc.cpp (SVC: cameras -> engine -> UDP), defom_receiver.py (AIB -> ROS)
  7_ros_tools/                 PC, ROS Noetic
      capture_stereo.py            record training frames from a ROS stereo topic
      capture_public_stereo.py     the same from separate left / right camera topics (public bags), rectified
      compare_beex_bags.py         any model on DWE / Voyis recordings, next to the camera's own depth
      compare_public_bags.py       the same for public stereo bags (separate, unrectified cameras)
  8_results/
      RESULTS.md                   every model tested, analysis, pitfalls, next steps
      BRIEF_UPDATE.md              one-page status
      figures/                     charts and side-by-side panels

checkpoints/                   (repo root) DEFOM releases: defomstereo_vitl_middlebury.pth,
                               defomstereo_vits_rvc.pth (git-ignored)
```

Scripts run from any directory; relative paths resolve against the repository root.

---

## 1. Setup

### 1.1 Jetson Thor (training)

JetPack 7, Python 3.12, CUDA 13. Run long jobs inside `tmux`, so they survive a disconnect.

```bash
conda create -n defomstereo python=3.12 -y && conda activate defomstereo
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
pip install --no-deps nvidia-cudss-cu13
pip install nvpl-lapack nvpl-blas numpy opencv-python pillow pyyaml scipy timm opt_einsum einops matplotlib
SITE=$CONDA_PREFIX/lib/python3.12/site-packages
conda env config vars set LD_LIBRARY_PATH=$SITE/nvpl/lib:$SITE/nvidia/cu13/lib PYTHONNOUSERSITE=1
conda deactivate && conda activate defomstereo
python -c "import torch; print(torch.cuda.get_arch_list())"   # must include sm_110
```

### 1.2 PC (export, evaluation, rosbags)

- **Python:** `torch`, `torchvision`, `numpy`, `opencv-python`, `pillow`, `pyyaml`, `scipy`,
  `timm`, `opt_einsum`, `einops`, `matplotlib`, plus `onnx` and `onnxruntime` for export.
  `xformers` is optional.
- **ROS Noetic** for `7_ros_tools/`. No BeeX workspace needs to be sourced.
- **Training also runs on the PC** (RTX 4070, 12 GB): set `training.batch_size: 2` for the slim
  configs.

### 1.3 Files to copy in

These are git-ignored, so they are not in a clone:

| file | where | needed for |
|---|---|---|
| `defomstereo_vitl_middlebury.pth` (teacher) | `checkpoints/` | everything |
| `defomstereo_vits_rvc.pth` (ViT-S release) | `checkpoints/` | ViT-S s2 |
| `1_slim_model/` | `distillation/3_checkpoints/` | slim v9: fine-tune, or its encoder for a fresh core |
| `2_fast_vits_model/` | `distillation/3_checkpoints/` | ViT-S s2 fine-tune |
| training frames | `distillation/1_datasets/underwater/<session>/` | training |
| new-scene frames | `distillation/1_datasets/probe/<set>/` | ViT-S validation |

**Check the install:** `python3 distillation/slim_stereo.py` must end with `EQUIVALENT`. It
confirms that SlimDEFOM with DEFOM's own sizes reproduces DEFOMStereo exactly.

---

## 2. Data

### 2.1 Record

From a BeeX recording (`.beex` files are rosbags) or the live stream:

```bash
rosbag play --topics /ikan/explore3d/depth_image -- <dive>.beex                      # DWE
python3 distillation/7_ros_tools/capture_stereo.py --out distillation/1_datasets/underwater

rosbag play --topics /ikan/voyis/stereo_concatenated/compressed -- <dive>.beex        # Voyis
python3 distillation/7_ros_tools/capture_stereo.py --out distillation/1_datasets/underwater \
    --topic /ikan/voyis/stereo_concatenated/compressed --session voyis_dive3

rosbag play bus_outside.bag                                                            # public bag
python3 distillation/7_ros_tools/capture_public_stereo.py --out distillation/1_datasets/public \
    --left /slave1/image_raw/compressed --right /slave2/image_raw/compressed --session bus_outside
```

- **Format:** one folder per session of **rectified, side-by-side** frames (left | right), any
  camera and size: DWE 800x600, Voyis 2816x2816, public datasets. Copied-in datasets work
  the same way: one folder per sequence.
- **Topics:** any stereo topic the compare tools read works: DWE `DepthImage`, side-by-side
  `CompressedImage` / `Image`, `StereoImage`.
- **Public bags** (one raw topic per camera, e.g. `cemetery.bag`, `bus_outside.bag`):
  `capture_public_stereo.py` pairs the two cameras by stamp (`--max_dt` 0.02 s), rectifies them
  from `<topic>/cam_info` to `--width` (800, i.e. 800x600 per eye) and saves the same
  side-by-side sessions. The left camera is the one whose `P[0,3]` is 0; swapped topics are
  refused. `--no_rectify` for bags that are already rectified. Keep public data in its own folder
  (`1_datasets/public`) unless it is meant to be trained on with the dives: the configs' `image_dir`
  takes every session under it.
- **Recording behaviour:** near-duplicates are skipped (`--min_mad 2.0`), so record while
  moving. It stops at `--max_frames` (4000). Reusing `--session` appends to it.
- **Validation split:** the last 5% of every session is held out.
- **Probe sets** (`1_datasets/probe/<set>/`, e.g. tank, lab in air) are never trained on. They are
  how a model's behaviour on unfamiliar scenes is caught: held-out frames come from the same
  sites as training and miss it.

### 2.2 Labels: the teacher's disparity

Run once per dataset. It is resumable, and after adding sessions only the new frames are done.
All three configs share the same labels.

```bash
python3 -u distillation/teacher_labels.py --config distillation/2_configs/slim_v9_2_finetune.yaml
```

- **Format:** ViT-L at 32/8 iterations, frames resized to 800 long side; 16-bit PNG holding
  disparity x 64 in `1_datasets/teacher_labels_800/<session>/`.
- **Time:** ~0.5-1 s per frame.
- **The labels are fixed to one teacher and size** (`labels.json`). For another, use another
  `labels_dir`.

---

## 3. Train

All training runs work the same way:
- **Validation:** whole held-out frames, every epoch. The score is the share of pixels within
  3 px of the teacher.
- **Best epoch:** kept as `3_checkpoints/<run>/student_best.pth` + `best.json`.
- **Fine-tunes** score their starting model first, so `student_best.pth` only changes if
  training beats it. Give every run its own `checkpoint_dir`.
- **Options:** `--epochs N`; `--resume_from <run>` continues an interrupted run from its newest
  epoch; `--max_steps N` runs a quick smoke test.

### 3.1 Slim v9

**Fine-tune the deployed model** -- the usual case, e.g. after adding data:

```bash
python3 -u distillation/train_slim.py --config distillation/2_configs/slim_v9_2_finetune.yaml 2>&1 | tee slim_finetune.log
```

- **What trains:** starts from `1_slim_model`; the stereo core and the encoder student train
  together, the student at a 5x lower learning rate.
- **Time:** ~30 min/epoch on the Thor (batch 6), 10 epochs.

**From scratch:** a new stereo core on `1_slim_model`'s encoder, then the fine-tune:

```bash
python3 -u distillation/train_slim.py --config distillation/2_configs/slim_v9_1_from_vitl.yaml 2>&1 | tee slim_scratch.log
# then slim_v9_2_finetune.yaml with  init_from: distillation/3_checkpoints/slim_v9_from_vitl
```

- **Time:** ~26 min/epoch on the Thor; validation plateaus after ~20-40 of the 60 epochs.

**Watch for:** each epoch prints per-camera agreement with the teacher, e.g.
`[val vs teacher] 800x600 EPE 1.14 px | within 3px 92.1%`, then `new best` or `best remains`.

**Warning:** slim v9 learns matching from these frames alone. Trained on our dives only, it
fails on new scenes (0% in the lab, 33-47% in the tank). Add varied data, such as public
stereo datasets, and check the probe sets with `evaluate.py --image_dir`.

### 3.2 ViT-S s2

**The deployed model needs no training.** It is the release with unchanged weights:

```bash
python3 distillation/slim_stereo.py --make vits_s2 --out distillation/3_checkpoints/<new_folder>
```

(It refuses to overwrite an existing `student_best.pth`; `2_fast_vits_model` already holds it.)

**Fine-tune** (optional, e.g. with public datasets added):

```bash
python3 -u distillation/train_vits.py --config distillation/2_configs/vits_s2_finetune.yaml 2>&1 | tee vits_finetune.log
```

- **Loss:**
  - L1 to ViT-L's labels;
  - plus the model's internal outputs (matching features, GRU states and context) kept close to
    the frozen ViT-S release.

  The anchor keeps it general: every attempt to retrain DEFOM's GRU on our dives alone lost
  the tank and the lab.
- **What trains:** fnet/cnet by default (`training.train`). Add the GRU only with data beyond
  our dives.
- **Validation:** every epoch vs ViT-L at 32/8, at 512 wide, on held-out frames, the probe sets,
  and a 30 px shift test. An epoch failing the shift test is never kept. The release's own
  score is printed as the bar to beat.
- **Time and memory:** ~1 h/epoch on an RTX 4070 over 11k frames, ~11 GB at batch 4.
- **Precision:** the release and ViT-L run in fp32 (DEFOM's `fnet.convd` reaches ~1e5 and
  overflows fp16); the model trains in bf16.
- **Status:** this config has only had smoke runs so far.

---

## 4. Evaluate

```bash
python3 distillation/evaluate.py \
    --models slim_v9=distillation/3_checkpoints/1_slim_model \
             vits_s2=distillation/3_checkpoints/2_fast_vits_model@10/4 \
             new=distillation/3_checkpoints/slim_v9_finetune \
    --long_sides 512 800 --out distillation/4_evaluations/<date>_<name>
# a new dive or a probe set:   --image_dir distillation/1_datasets/probe/<set> --frames all
```

- **Reference:** every model is compared pixel by pixel with ViT-L at 32/8, on the held-out
  frames (or every frame of `--image_dir`).
- **Iterations:** `name=path@iters/scale` sets a model's iterations, as deployed (ViT-S s2:
  `@10/4`).
- **Paired comparison:** the first model is the reference the others are compared to
  (`--reference` to change it).
- **`report.md`:**
  - a verdict, and a ranking with 95% confidence intervals and P(better);
  - per-camera tables;
  - charts by distance and by texture;
  - sample disparity and error maps.

  Trust a difference only when the report says **clearly better**.
- **Time:** ~10-15 min for a few models, 80 frames per camera, 2 sizes, on the 4070.

To look at models on whole recordings, see [section 7](#7-compare-models-on-rosbags-pc).

---

## 5. Export (ONNX)

```bash
python3 distillation/export_model.py --ckpt distillation/3_checkpoints/1_slim_model --format onnx \
    --size 800x600 --iters 12 --scale_iters 4
python3 distillation/export_model.py --ckpt distillation/3_checkpoints/2_fast_vits_model --format onnx \
    --size 512x384 --iters 10 --scale_iters 4
```

- **Output:** `5_output_models/<run>/`, named like the checkpoint folder:
  `defom_<label>_<W>x<H>_it<i>s<s>.onnx` + `.json` + `_reference.npz`. Copy all three together.
  - The `.json` lists the layers that must stay fp32.
  - The `.npz` holds test frames with the fp32 and fp16 PyTorch results.
- **Size:** `--size` is the inference size. `defom_svc` upsamples the disparity to 1600x1200,
  and smaller is much faster.
- **Checked on export:** every export is compared with fp32 PyTorch on real frames and is not
  written if it disagrees. The exporter rewrites einsum to matmul and unfold to conv for
  TensorRT (identical maths, checked).
- **Other targets:** an original release exports the same way
  (`--ckpt checkpoints/defomstereo_vits_rvc.pth`). TorchScript (`--format torchscript --half`)
  is for the PC ROS node `defom_stereo_pointcloud`.

---

## 6. Deploy on the Orin

```
SVC (Orin NX)                                                        AIB (ROS)
/dev/video0 + /dev/video2, raw MJPEG 1600x1200 @ 60 fps
  -> pair L/R by capture time (<= 12 ms apart)
  -> rectify with e3d1008.dwecal
  -> resize to the engine's size -> TensorRT fp16b (slim v9 800x600 | ViT-S s2 512x384)
  -> disparity -> depth, upsampled to 1600x1200 (edge-aware)
  -> UDP :5610 ── depth float32 + intrinsics + left JPEG ──>  defom_receiver.py -> /ikan/explore3d/defom/*
```

Everything on the SVC uses only what JetPack 6 ships: python3, TensorRT 8.6.2, numpy, OpenCV
and CUDA. Nothing needs installing, and nothing outside `~/defom` is changed.

### 6.1 Build and check the engine

An engine only runs on the GPU and TensorRT version that built it, so it is built on the SVC
itself. Copy the export folder and the tools from the PC:

```bash
scp -r distillation/5_output_models/2_fast_vits_model distillation/6_deploy/orin dwe@192.168.4.5:~/defom/
```

On the SVC, stop DWE's stream first so the GPU is free (`sudo systemctl stop rtpsender`), then:

```bash
cd ~/defom
nohup bash orin/run_step2.sh 2_fast_vits_model 512x384_it10s4 > step2.log 2>&1 &   # chosen models, in order
nohup bash orin/run_step2.sh 2_fast_vits_model > step2.log 2>&1 &                  # every model, smallest first
tail -f step2.log
```

- **Time:** each build takes 25-45 min. One build at a time.
- **Harmless warnings:** "tactic device request ... insufficient memory", INT64 weights cast
  to INT32, and FP16 subnormal weights.
- **Output, per model:**
  - `<model>_fp16b.engine`;
  - `_engine.json` (build info);
  - `_check.json` / `_check.jpg` (left | PyTorch | engine | difference).

  Every check ends with `ACCURACY OK` (or `FAIL`) and the measured latency. All results
  collect in `<run>/step2_results.txt`. Existing engines are reused; delete one to rebuild it.
- **Precision options:** `PRECISIONS="fp16b fp32" bash orin/run_step2.sh ...` also builds an
  fp32 reference. Single steps: `build_engine.py <onnx> --precision fp16b|fp16|fp32` and
  `check_engine.py <engine>`.

**Precision.** The few convolutions where the depth features enter the stereo core exceed
fp16's range; the export lists them, and they always stay fp32. **fp16b** (the default) runs
everything else in fp16, except the correlation, coordinate and disparity maths and
LayerNorm. It is 2x faster than plain fp16 at the same accuracy.

**ViT-S is fp16-sensitive.** On murky frames with nothing to match, PyTorch's own fp16 already
differs from fp32 by up to ~1 px, yet it agrees with ViT-L just as well. `check_engine.py`
therefore also accepts an engine that is no further from fp32 than PyTorch's fp16 is, and says
so: `ACCURACY OK (as close to fp32 as PyTorch's own fp16 is; ...)`. Copy the current
`6_deploy/orin/` with every model.

Measured (TensorRT 8.6.2, MAXN):

| model | size, iterations | latency |
|---|---|---|
| slim v9 (`1_slim_model`) | 800x600, 12/4 | 172 ms (5.8 Hz) |
| ViT-S s2 (`2_fast_vits_model`) | 512x384, 10/4 | 165 ms (6.05 Hz) |
| ViT-S s2 | 640x480, 12/4 | 298 ms (3.4 Hz) |

### 6.2 Rectification check

The raw 1600x1200 stream is **not** rectified: rows are offset by a median ~2 px. The
`.dwecal` maps are needed, and `defom_svc` applies them exactly as DWE does for its own stream.
The calibration is for water, so judge it **underwater**: in air even a correct calibration
does not line up.

```bash
scp ~/DWE/RTPSender/calibration/e3d1008.dwecal dwe@192.168.4.5:~/defom/
# on the SVC, underwater, looking at a still textured scene 1-3 m away:
python3 orin/check_rectification.py --dwecal e3d1008.dwecal --out rect_check
```

- **Reports:** the vertical offset between matched left/right features, raw and after the
  maps; it should be ~1-1.5 px at 1600 wide.
- **Checks:** the cameras' serial numbers, so you can confirm the `.dwecal` belongs to
  `e3d1008`.
- **Saves:** the pairs, in `rect_check/raw_*.png`, for offline tests.
- **`defom_svc` checks on its own too:** it runs the same check at every start (`--rectify
  auto`) and switches the maps off only if the raw stream is already aligned. To run only the
  check: `./defom_svc --dwecal ../e3d1008.dwecal --check_rectification`.

### 6.3 SVC: `defom_svc`

**Copy and build** (once, ~1 min; g++, OpenCV, CUDA and TensorRT come with JetPack):

```bash
# PC
scp -r distillation/6_deploy/defom_svc dwe@192.168.4.5:~/defom/
# SVC
cd ~/defom/defom_svc
g++ -O3 -std=c++17 -o defom_svc defom_svc.cpp -I/usr/include/opencv4 -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d \
    -lnvinfer -lnvinfer_plugin -lcudart -lpthread
```

The `-Wdeprecated` notes from TensorRT's headers are harmless.

**Run.** Only one program can use a camera, so stop DWE's stream first
(`sudo systemctl stop rtpsender`; `start` restores it):

```bash
cd ~/defom/defom_svc
./defom_svc --engine ../2_fast_vits_model/defom_vits_s2_release_512x384_it10s4_fp16b.engine \
            --dwecal ../e3d1008.dwecal --aib 192.168.4.1                     # ViT-S s2
./defom_svc --engine ../1_slim_model/defom_slim_v9_joint_800x600_it12s4_fp16b.engine \
            --dwecal ../e3d1008.dwecal --aib 192.168.4.1                     # slim v9

nohup ./defom_svc --engine ... --dwecal ../e3d1008.dwecal --aib 192.168.4.1 > defom_svc.log 2>&1 &   # keeps running
pkill -x defom_svc                                                                                     # stop
```

Engines built before 2026-10-30 may still sit in the SVC's older folders (`vits_s2_release/`,
`slim_v9_joint_ep2/`).

It prints a status line every 5 s (measured with ViT-S s2, 2026-10-30):

```
5.80 Hz | prep 118 ms, infer 170 ms, depth+jpeg 20 ms, send 129 ms | capture->sent 340 ms | L/R skew 0.1 ms | dropped 0 | send errors 0
```

- **prep:** waiting for a pair, decoding, rectifying and resizing. It overlaps with inference,
  which sets the rate.
- **capture->sent:** the latency from exposure until the frame leaves the SVC.
- **L/R skew:** the gap between the two cameras' exposures. They are not hardware-synchronised.
- **dropped:** a frame was replaced before the next stage took it.
- **Exit:** with code 1 if a camera is busy or disconnected, or no pair arrives for 2 s.

| option | default | |
|---|---|---|
| `--rectify auto\|on\|off` | auto | see 6.2 |
| `--left`, `--right`, `--swap_lr` | /dev/video0, /dev/video2 | the start-up check warns of negative disparity (eyes swapped) |
| `--cam_fps`, `--max_skew_ms` | 60, 12 | camera rate; largest L/R exposure gap accepted |
| `--max_hz` | 0 | cap the output rate (less heat); 0 = as fast as possible |
| `--aib`, `--port` | 192.168.4.1, 5610 | receiver |
| `--send_mbps`, `--packet_bytes` | 500, 8192 | UDP pacing and datagram size |
| `--baseline_m`, `--edge_px` | 0.1, 1.5 | stereo baseline; depth-edge threshold for upsampling (model pixels) |
| `--jpeg_quality` | 90 | left image |
| `--input_dir DIR` | | test input: side-by-side raw 3200x1200 images, looped |
| `--save_dir DIR`, `--save_frames N` | | save the first N frames (left PNG, disparity and depth float32) |

### 6.4 AIB: `defom_receiver.py`

**Once, as root:** let a socket hold whole frames (~8 MB each). Without this the receiver warns
"receive buffer below 8 MB", and frames drop as soon as anyone subscribes.

```bash
sudo sysctl -w net.core.rmem_max=67108864
echo 'net.core.rmem_max=67108864' | sudo tee /etc/sysctl.d/90-defom.conf    # keep after reboot
```

**Copy and run.** The AIB runs ROS Melodic, so use `python2`; the script also runs on
`python3` with Noetic. It needs numpy, plus rospy and cv2 for ROS.

```bash
scp distillation/6_deploy/defom_svc/defom_receiver.py aib@<aib>:~/
python2 defom_receiver.py --no_ros                    # first: is anything arriving?
python2 defom_receiver.py                             # ROS (roscore running)
```

Its status line every 5 s shows the rate, the SVC's timings, network time and the
`complete / dropped / skipped` frame counts. **dropped** means a frame missed datagrams; raise
`rmem_max`.

| topic (under `--ns /ikan/explore3d/defom`) | |
|---|---|
| `depth` | Image 32FC1, metres, NaN = invalid, 1600x1200 |
| `camera_info` | rectified intrinsics at 1600x1200 |
| `left/compressed` | rectified left JPEG, pixel-aligned with depth |
| `points` | PointCloud2 XYZRGB, every `--stride` (2) pixel; built only while subscribed |
| `depth/compressed` | JET-coloured depth; built only while subscribed |

- **Options:**
  - `--frame_id` (default `explore3d_left_optical`): the TF frame of the left camera's
    *optical* axes;
  - `--min_depth`/`--max_depth` (metres; 0 = no limit): cut depth and points;
  - `--save DIR`: also saves the first frames.
- **Stamps:** the SVC's clock is not set (it reads 1970), so each message is stamped with its
  arrival time minus the frame's age on the SVC.
- **Topic names:** these are the same names as the PC node `defom_stereo_pointcloud`, so don't
  run both.
- **Wire format:** one frame = a 128-byte header (size, intrinsics, offsets, timings), then the
  depth (float32 metres), then the left JPEG. It is cut into datagrams with 32-byte headers.
  `FrameHeader` / `PacketHeader` in `defom_svc.cpp` and `FRAME` / `PACKET` in
  `defom_receiver.py` define it; change both together.
- **Bandwidth:** each frame is ~8.1 MB (7.68 MB of it depth), ~370 Mbit/s at 5.7 Hz.

### 6.5 Test without the vehicle

On any machine with an engine built for its own GPU:

```bash
./defom_svc --engine <engine> --dwecal e3d1008.dwecal --input_dir raw_pairs/ --aib 127.0.0.1 --max_hz 3
python3 defom_receiver.py --no_ros --save rx/
```

`raw_pairs/` holds side-by-side raw 3200x1200 frames, e.g. the `rect_check/raw_*.png` saved by
`check_rectification.py`.

---

## 7. Compare models on rosbags (PC)

Run any model on recorded dives and see it next to the camera's own depth, before building
anything for the Orin.

```bash
roscore
rosbag play --loop --topics /ikan/explore3d/depth_image -- <dive>.beex            # DWE (play only this topic)
python3 distillation/7_ros_tools/compare_beex_bags.py --show --input /ikan/explore3d/depth_image

rosbag play --loop <voyis>.beex                                                      # Voyis
python3 distillation/7_ros_tools/compare_beex_bags.py --show --input /ikan/voyis/stereo_concatenated/compressed

rosbag play --loop cemetery.bag                                                      # public stereo bag
python3 distillation/7_ros_tools/compare_public_bags.py --show
```

- **Recordings:** BeeX `.beex` recordings are ordinary rosbags. No workspace needs to be
  sourced: messages are decoded with the definition stored in the bag.
- **Models**, numbered at start:
  - **our exports:** every `.json` + `.onnx` under `5_output_models/<run>/`, run from
    `3_checkpoints/<run>/` with the export's iterations;
  - **baselines:** every `checkpoints/defomstereo_*.pth`, at 32/8. ViT-L is the reference.

  The `.engine` only runs on the Orin, so on the PC each model runs in PyTorch.
- **Timings:** the `ms` shown is PyTorch on this PC, not the Orin.
- **GPU memory:** at most 3 models stay loaded (`--max_loaded`).
- **Switch model:** keys `1`-`9`, `n` / `p` in the window (`q` quits), or:

  ```bash
  rostopic pub -1 /defom_compare/select std_msgs/String "vitl"           # part of the name
  rostopic pub -1 /defom_compare/select std_msgs/String "data: '2'"      # its number
  ```

  **Write live values as `"data: '...'"`.** `rostopic` reads a bare `2` as a number and `off`
  as false, and rejects both.

| filter (off by default) | live | effect |
|---|---|---|
| `--range MIN MAX` | `~range "data: '0 3'"` / `"data: 'off'"` | model depth outside MIN..MAX metres becomes invalid (NaN, black) |
| `--texture T` | `~texture "data: '6'"` / `"data: 'off'"` | model depth where the left image has no texture (murky water, where models guess) becomes invalid; 6 suits Voyis |
| `--min_depth` / `--max_depth` | | point clouds only |

| output (under `/defom_compare`, or `/defom_compare_public`) | |
|---|---|
| `compare` | left \| camera's own depth \| model depth, near = red (`--vis_min/--vis_max`, 0-3 m) |
| `depth`, `depth/color`, `points`, `camera_info` | model depth: 32FC1 metres, coloured, XYZRGB cloud, intrinsics |
| `ref/depth/color`, `ref/points` | the camera's own depth (DWE `refined_pro`, or Voyis RGBD matched by time) |
| `rectified` | public bags: left \| right after rectification, with row lines |

- **Inputs:**
  - **DWE:** `/ikan/explore3d/depth_image`, 800x600 left\|right + DWE depth.
  - **Voyis:** `/ikan/voyis/stereo_concatenated/compressed`, with its depth from
    `/ikan/voyis/stereo/data`. Not `stereo_concatenated_raw`, which is unrectified.
  - **Public bags:** `/cam_fl/image_raw/compressed` + `/cam_fr/image_raw/compressed`, rectified
    from each `<topic>/cam_info`. Use `--no_rectify --fx --cx --cy --baseline` if already
    rectified.
- **Intrinsics:** DWE uses `e3d1008.dwecal` at 800x600, baseline 0.1 m; Voyis 180114006,
  baseline 0.170 m. Override with `--fx --fy --cx --cy --baseline`.
- **"Differs from DWE/Voyis"** compares with the camera's own depth, which is not ground truth.

---

## Config reference

| key | meaning |
|---|---|
| `paths.init_from` | model to fine-tune (a run folder uses its `student_best.pth`) |
| `paths.resume_from` / `--resume_from` | continue an interrupted run (newest epoch); `--epochs N` extends it |
| `paths.checkpoint_dir` | output folder: one per run |
| `paths.student_ckpt` (slim, scratch) | the slim checkpoint whose encoder student a new core is built on |
| `paths.release_ckpt` (ViT-S) | the frozen ViT-S release (internal anchor) |
| `paths.labels_dir`, `paths.probe_dir` | teacher labels; new-scene validation sets |
| `slim.preset` / `vits.preset` | `v9` / `vits_s2` (`slim_stereo.py` PRESETS) |
| `training.iters` / `scale_iters` | refinement iterations, as deployed (slim 12/4, ViT-S 10/4) |
| `training.freeze_student` (slim) | keep the encoder student fixed |
| `training.train` (ViT-S) | parameter-name prefixes that train |
| `training.val_max_frames` | held-out frames per camera (per set for ViT-S) |
| `data.label_long_side` / `scale_jitter` / `crop` | label scale, random rescale, crop |

## Troubleshooting

| symptom | fix |
|---|---|
| CUDA out of memory in training | lower `training.batch_size` (2 on a 12 GB GPU); stop other GPU processes |
| `frames have no teacher label` | run `teacher_labels.py` with the same config |
| `holds labels made with different settings` | those labels came from another teacher/size: use another `labels_dir` |
| `best.json was scored on different validation frames` | expected after adding data: the held-out set changed |
| `is preset ..., the config says ...` | `init_from` points at the other model's checkpoint |
| `libnvpl_*` / `libcudss` not found (Thor) | the NVPL / cuDSS steps in 1.1 |
| Orin check `ACCURACY FAIL` on a ViT-S engine | copy the current `6_deploy/orin/` and re-export with fp16 references (6.1) |
| `defom_svc`: cannot set MJPEG / STREAMON failed | DWE's `rtpsender` holds the cameras: `sudo systemctl stop rtpsender` |
| receiver: `receive buffer below 8 MB`, frames dropped | `rmem_max` (6.4) |
| `rostopic pub` to `~select` / `~range` / `~texture` ignored | write the value as `"data: '...'"` |
