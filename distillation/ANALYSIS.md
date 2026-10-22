# DEFOM-Stereo student distillation — analysis

What the distillation does, why each design choice was made, and what has been
measured. For the commands, see [README.md](README.md).

Contents: [1 What is distilled](#1-what-is-distilled) ·
[2 The student](#2-the-student) · [3 The loss](#3-the-loss) ·
[4 Data and resolution](#4-data-and-resolution) · [5 Numerics](#5-numerics-fp16-bf16-tf32) ·
[6 Validation and model selection](#6-validation-and-model-selection) ·
[7 Results](#7-results) · [8 Fine-tuning vs resuming](#fine-tuning-vs-resuming) ·
[9 Limitations and next steps](#9-limitations-and-next-steps)

---

## 1. What is distilled

DEFOM-Stereo = a DepthAnythingV2 (DINOv2-L) encoder that gives monocular depth
features, plus a stereo core (feature/context nets, correlation, GRU updates).

```
DINOv2-L backbone   1217.5 MB   ┐
depth_head           123.8 MB   ├─ REPLACED by a 27 MB MobileViT student
depth_feat           122.5 MB   ┘
fnet / cnet / GRU     66.8 MB   ── KEPT, unchanged, from defomstereo_vitl_middlebury.pth
total               1530.5 MB   →  ~94 MB deployed
```

The student only has to reproduce the encoder's **output tuple** — three
`d_features` maps, the left/right matching features, and an inverse-depth prior
— so it never has to learn stereo matching itself. No ground truth is needed:
the frozen teacher's encoder outputs are the targets, so any unlabelled footage
from the deployment domain is training data.

## 2. The student

**Backbone: MobileViT-S** (`mobilevit.py`), tapped at /4, /8 and /16, then a
light FPN with separate heads per output. It is a frozen copy of the
architecture used in the DINOv3→MobileViT distillation, so those checkpoints
warm-start it (304/304 tensors), and verified bit-identical to the original.
It lives in this repository so edits elsewhere cannot silently change it.

- **Input divisible by 32.** MobileViT unfolds 2×2 patches on the /16 map,
  which must be even. /16 crashed at 1064×1400 (1600×1200 input) and at 336×448
  (a training crop); /32 works everywhere.
- **Output calibration.** The teacher's outputs span five orders of magnitude
  (`left_feat` reaches ~2×10⁷ for this teacher). A fresh head emits O(1) values,
  so the gradient towards a 10⁷ target is ~10⁻⁴ and that term never trains —
  measured: 0.001% movement in 8 epochs. Before training, the teacher's output
  mean/std are measured and stored as buffers; heads predict standardised
  values and are de-normalised on output. With it, the same 8 epochs moved the
  matching-feature terms 19% and the depth prior 81%.
- **Normalisation.** The FPN uses GroupNorm: with BatchNorm at the 2–4 batch the
  teacher allows, train and eval outputs differed by 28–45%, so training
  optimised a different function from the one deployed. MobileViT itself has
  BatchNorm; unfrozen it showed the same problem (**19–22%** train/eval gap).
  So its statistics are re-estimated on the training data and then frozen
  (`student.freeze_bn`): train and eval are now identical (gap exactly 0) and
  the BN affine weights still learn.

## 3. The loss

Five teacher tensors are matched. Feature maps: squared error summed over
channels, averaged over batch and space, **divided by the teacher's own energy**
(`relative_mse`). Without that division the matching-feature term (~10¹¹) would
drown every other term (~10⁶); with it, every term is a relative error of
order 1 and the configured weights mean what they say. The depth prior is
matched raw with L1 — DEFOM normalises it downstream, so normalising here would
discard the scale the teacher emits.

## 4. Data and resolution

**Any camera, any size, mixed.** Frames live in one folder per capture session.
Each session's camera is identified by its per-eye size (800×600 DWE,
1600×1200 DWE raw, 2816×2816 Voyis). No intrinsics are used anywhere in
training: the loss is label-free and the encoder runs on each eye separately.
Calibration matters only for rectification (before capture) and for depth (in
the ROS node's camera profile).

**Resolution-agnostic training.** Each eye is resized, aspect preserved, to a
random long side in 512–1024, and both eyes take the same random 384×512 crop
(a common crop window keeps the pair's disparities valid). This spans the
sizes the ROS node runs at. The old fixed 800×600 resize had two problems: it
squashed 1:1 Voyis frames to 4:3, and it tied the student to one scale — its
feature error against the teacher was 0.11 at 800×600 but 0.30–0.47 at
512×384 and 1024×768.

**Balanced sampling.** Each camera gets an equal share of every epoch, capped so
no frame is drawn more than 4× per epoch (an uncapped equal share once made a
single frame half of an epoch, drawn ~2,700 times).

**Rectification** was measured on the data: the DWE SVC stream has a median
vertical offset of 0.24 px between eyes, i.e. rectified. The raw 1600×1200
stream is likely not (the SVC's rectification maps move pixels by up to 27 px),
so raw frames should be rectified before being used.

## 5. Numerics: fp16, bf16, TF32

- **fp16 was broken, now fixed.** The teacher's matching features (~2×10⁷)
  overflow fp16's 65,504 at the first stereo-core conv that reads them, which
  turned 100% of the disparity into NaN — the black depth image seen in the ROS
  node. `ConvBlock` (exactly where depth features enter the stereo core) now
  always runs that conv in fp32. fp32 output is bit-identical; fp16 is finite,
  ~1.55× faster, and within 0.05 px median of fp32 (larger differences only in
  textureless water).
- **bf16 teacher** (`teacher_bf16`): bf16 keeps fp32's range, so it cannot
  overflow. Teacher pass 456 → 272 ms on a 4070 for a 1–3% perturbation of the
  targets; on the Thor the loss curve matched the fp32 run to three decimals.
  Not available on Xavier.
- **TF32** (`tf32`): teacher 450 → 334 ms with the target error unchanged
  (0.18% → 0.19%).

## 6. Validation and model selection

**What is measured.** Feature loss is only a proxy. Validation swaps the student
into the real DEFOM model and compares its **disparity** against the unmodified
teacher on held-out frames: the share of pixels within 1/2/3 px, mean, median
and 90th-percentile error. The **selection score** is within-3px averaged over
every camera × size, each weighted equally.

**Held-out frames.** The last 5% of **each session** (temporal split).
Consecutive survey frames are near-duplicates, so a random split would put
near-copies of training frames into validation and report a flattering number.

**Why 80 frames.** The 40-epoch run validated 24 frames per camera. With so
few, one hard frame moves a line by ~4%: the log suggested validation had
stalled (and that Voyis @640 had got worse), while a 160-frame evaluation
showed a clear +5.6 point gain over the original student. 80 frames per camera
keeps the noise at ~1–2%.

**Best checkpoint.** After every epoch the selection score is compared with the
best so far; improvements are saved as `student_best.pth` with a `best.json`.
When starting from a checkpoint, that checkpoint is validated first, so the
best can only change if training actually beats it. A score is only
comparable on the same held-out frames, sizes and iterations, so `best.json`
stores a signature of them; after adding data the held-out set changes and
tracking restarts.

**`evaluate.py`** answers the same question offline, more thoroughly: any
number of checkpoints, bootstrap 95% confidence intervals, a **paired**
comparison against a reference (the same frames for both models, so
frame-to-frame difficulty cancels), error by texture and by distance, sample
disparity and error maps, and training curves.

## 7. Results

Evaluated with `evaluate.py` on 80 Voyis + 80 DWE held-out frames at long
sides 512 and 640 (report: `distillation/evaluations/2026-10-22_multiscale_run/`).
`orig` is the fixed-800×600 student (epoch 49); `epN` are epochs of the
40-epoch multi-scale run (Thor, bf16 + TF32, warm-started from `orig`).

| model | within 3 px, overall (95% CI) | vs `orig` (95% CI) | Voyis @512 | Voyis @640 | DWE @512 | DWE @640 |
|---|---|---|---|---|---|---|
| **`ep20`** (selected) | **76.8%** (74.1–79.4) | **+5.6** (+4.1 to +7.1) | 75.2% | 71.0% | 82.9% | 78.1% |
| `ep35` | 76.4% (73.7–79.0) | +5.2 (+3.7 to +6.7) | 75.2% | 70.1% | 82.2% | 78.0% |
| `ep0` | 75.7% (73.0–78.2) | +4.5 (+3.0 to +6.0) | 74.5% | 70.2% | 80.4% | 77.7% |
| `orig` | 71.2% (68.7–73.5) | reference | 69.3% | 65.9% | 75.6% | 73.9% |

**What this shows**

1. **Multi-scale, multi-camera training clearly helped**: +4.5 to +5.6 points
   over the original student, every line, P(better) 100%.
2. **Almost all of the gain came in the first epoch.** Epochs 0, 20 and 35 are
   within each other's confidence intervals; training loss kept falling
   (1.34 → 0.95) without end-to-end disparity following. Fine-tunes therefore
   need far fewer epochs (the config now uses 10).
3. **Where the students disagree with the teacher is open water.** On Voyis,
   `ep20` matches the teacher on 93–94% of textured pixels but only 49–57% of
   textureless ones; stereo is ill-posed there for any model. On DWE the split
   is smaller (80–84% vs 76–82%).
4. **By distance**, error on structure (disparity 5–30% of the image width) is
   0.1–1.4% of depth for all multi-scale epochs (the original: up to 1.7%).
   The far band (< 5%) is the hardest for every model (10–30%). The original
   student broke down at very close range on DWE (36% error where disparity
   exceeds 30% of the width); the multi-scale students reduce that to 1.1–1.6%.
5. **Median relative depth error** improves on DWE (e.g. 4.4% → 3.3% at 512)
   but is slightly worse on Voyis (3.5–3.6% → 3.7–4.4%). That median is
   dominated by far, textureless water, so the nearness and texture charts are
   the better guide.

**Speed** (deployed TorchScript, fp16, 24 iterations, RTX 4070): 118 ms at
640×480, 114 ms at 544×544. Training: 0.53 s/step on a 4070, 0.62 s/step on
the Thor; ~25–30 min per epoch over ~11k frames.

## Fine-tuning vs resuming

Both start from an existing checkpoint, but they mean different things:

| | `init_from` (fine-tune) | `resume_from` (continue) |
|---|---|---|
| use it to | train further on new or changed data | finish a run that was interrupted |
| optimizer, learning-rate schedule | fresh (new `epochs`, `lr`) | restored; extending `epochs` stretches the schedule |
| BatchNorm statistics | re-estimated on the current data | kept from the checkpoint |
| output calibration | kept from the checkpoint | kept |
| best checkpoint | the starting student is validated first | continues the run's best.json if the validation set is unchanged |

Adding datasets is a fine-tune: `init_from` the best student into a **new**
`checkpoint_dir`. Resuming would continue an old schedule (and its frozen
BatchNorm statistics) built for different data.

## 9. Limitations and next steps

- **The metric is agreement with the teacher**, not accuracy against ground
  truth: the student cannot exceed the teacher, and where the teacher is wrong
  the student is scored for copying it. An absolute check needs ground-truth
  depth (e.g. a measured object, like the 22 cm limpet mine).
- **Open water dominates the remaining gap** and is ill-posed for stereo.
  Masking or down-weighting textureless pixels in the loss is worth trying.
- **Voyis data is thin** (1,737 frames, 2 sessions). More Voyis sessions are the
  most direct way to improve the weakest line (Voyis @640).
- **Raw 1600×1200 DWE frames** should be rectified with the SVC's calibration
  maps before training on them.
- Training is GPU-bound on both machines; `teacher_bf16` + `tf32` is the
  fastest accurate setting.
