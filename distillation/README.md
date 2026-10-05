# DEFOM-Stereo encoder distillation

Shrinks `defomstereo_vitl_*` from **1530.5 MB to ~75.6 MB (20.2×)** by replacing
its DepthAnythingV2 encoder with a small student, while reusing DEFOM's trained
stereo core unchanged.

**No ground-truth disparity is needed.** The student regresses the frozen
teacher's encoder outputs, so any unlabelled stereo footage from the deployment
domain is valid training data.

---

## Read this before starting: the official `vits` may already be enough

| | vitl | vits | distilled student |
|---|---:|---:|---:|
| Total size | 1530.5 MB | **173.2 MB** | **~75.6 MB** |
| Time/frame (32 iters, 800×600) | 0.805 s | 0.472 s | not yet measured |
| Training required | — | **none** | yes |

`checkpoints/defomstereo_vits_sceneflow.pth` already exists on disk, loads
cleanly (`missing=0, unexpected=0`), and is 8.8× smaller with zero work.
Measured against vitl on explore3d footage it tracks closely — median
disagreement ~1 px, with the differences concentrated in textureless water
rather than on structure.

So distillation buys roughly **2.3× beyond vits**, not 20× beyond nothing.
Run `demo_v3.py` first and decide whether that is worth a training cycle.

---

## Relationship to the DINOv3→MobileViT pipeline

The existing distillation in `Yolo_Seg/vit_training` targets a single tensor:
`x_norm_patchtokens` → `[B, G, G, 1024]`, L2-normalised. The **target
contract** does not carry over, but the **architecture does** — MobileViT is
selectable as a backbone here (see below).

What actually differs, tested rather than assumed:

| | status |
|---|---|
| **Four outputs, not one** — `(d_features, left_feat, right_feat, idepth)` feeding two heads | **the real blocker**; solved by the FPN head in `defom_student.py` |
| **L2 normalisation** — DPT heads consume raw features, magnitude carries information | real, one line |
| **Patch 14 vs 16** — DEFOM feeds 532×700 | minor: MobileViT needs /16, so the wrapper resizes to 544×704 |
| ~~Non-square input~~ | **not a problem** — verified `(544,704) → (2,34,44,1024)` |
| ~~Fixed resolution at construction~~ | **not a problem** — a model built at 544×704 runs at 608×800 unchanged |

(The last two were claims I made early on and later disproved by testing. They
are listed so the record is accurate.)

`defom_student.py` matches DepthAnythingV2's *interface* and resizes to the
requested `(out_h, out_w)` at the end — which is what the teacher's heads
already do internally — so either backbone satisfies the same contract.

---

## What gets replaced

Measured parameter budget:

```
DINOv2-L backbone   1217.5 MB   frozen      ┐
depth_head           123.8 MB   frozen      ├─ REPLACED by an 8.9 MB student
depth_feat           122.5 MB   trained     ┘
fnet / cnet / GRU     66.8 MB   trained     ─── KEPT, loaded from the vitl ckpt
────────────────────────────────────────
total               1530.5 MB        →      ~75.6 MB
```

Because the 66.8 MB stereo core is reused verbatim, the student only has to
reproduce the encoder's output tuple — it never has to learn stereo matching.

---

## Choosing a backbone

Both backbones emit the identical output contract and share the same FPN head,
so they are interchangeable via one config line.

| `student.backbone` | student | deployed | dependency |
|---|---:|---:|---|
| `mbconv` width 0.5 | 4.8 MB | ~71.6 MB | none |
| **`mbconv` width 1.0** (default) | **8.9 MB** | **~75.7 MB** | none |
| `mbconv` width 1.5 | 15.5 MB | ~82.3 MB | none |
| `mobilevit` (-S) | 27.2 MB | ~94.0 MB | `Yolo_Seg/vit_training/scripts` |

```yaml
student:
  backbone: mobilevit
  mobilevit_variant: s          # s | xs | xxs
  mobilevit_init_from: ""       # optional warm start
```
or `--backbone mobilevit` on the command line.

`mobilevit` reuses MobileViT-S from your DINOv3 distillation unchanged — the
wrapper only *reads* its submodules to tap `/4`, `/8` and `/16` intermediates,
so existing checkpoints stay layer-name compatible.

### Warm-starting from your DINOv3 RGB checkpoints

Verified working against
`weights/old/Distilled_Dinov3_RGB/checkpoints_512_from_448_stage2/`:

```
[MobileViTBackbone] warm start from student_mobilevit_2M_Dinov3_based_epoch0002.pth
    loaded 304/304 tensors  (missing 0, unexpected 6 -- conv2/fc expected here)
```

**304/304** — a complete warm start. The 6 unexpected tensors are `conv2`
(160 → 1024, the DINOv3 projection), which this head does not use: it taps
`mvit[2]` at 160 channels upstream, so `conv2` is replaced with `Identity`.

Two things to get right:

- **Use the `.pth`, not the `.pt`.** The `.pt` in the same folder is a
  TorchScript export, not a state dict. The loader raises a clear `TypeError`
  saying so rather than failing obscurely.
- **The weights live under `model_state_dict`**, not the `model`/`student` key
  used elsewhere in this repo. The loader searches all four, strips any
  `module.` prefix, prints how many tensors actually matched, and **raises if
  fewer than half load** — a silent no-op here would waste the whole run.

That checkpoint came from `input_mode: rgb` at 512×512 against DINOv3 ViT-L/16.
Same modality as this camera, but a different teacher (DINOv3 vs DINOv2) and a
different target (1024-d patch tokens vs DEFOM's four outputs), so treat the
transfer as a head start on low-level features rather than a free win —
`mobilevit_init_from: ""` trains from scratch for comparison.

MobileViT requires input divisible by 16 and DEFOM feeds 532×700, so the
wrapper resizes to 544×704 on the way in. (Contrary to what I first thought,
non-square and variable resolution are both fine — only the /16 constraint is
real, and it costs one resize.)

## Files

| File | Purpose |
|---|---|
| `defom_student.py` | Student encoder (both backbones). Drop-in for `DepthAnythingV2`. **Run it directly to verify every contract.** |
| `capture_stereo.py` | Record training frames from the live ROS topic. |
| `distill_defom.py` | Trainer. |
| `configs/underwater_vitl.yaml` | Config. |

---

## Dependencies on other folders

The default path imports **nothing** from `Yolo_Seg/vit_training`:

| file | non-pip imports |
|---|---|
| `capture_stereo.py` | `rospy`, `sensor_msgs` |
| `defom_student.py` | none (`mobilevit_distill` only when `backbone: mobilevit`) |
| `distill_defom.py` | `core` (DEFOM's own), `defom_student` |

`backbone: mobilevit` is the single exception, and deliberately so — importing
the real module keeps it the single source of truth and keeps warm starts
layer-name compatible. It auto-locates `Yolo_Seg/vit_training/scripts` by
walking up from this folder; override with `student.mobilevit_scripts_dir`. If
it is missing you get a clear error naming `backbone: mbconv` as the way out.

## How to run

### 1. Record training data

```bash
cd DEFOM_Stereo/DEFOM-Stereo_Beex
python3 distillation/capture_stereo.py \
  --out distillation/datasets/underwater \
  --max_frames 4000
```

Writes 1600×600 side-by-side PNGs. Frames are deduplicated by mean absolute
difference (`--min_mad`, default 2.0) — survey footage at 6.5 Hz is highly
redundant, and without this you get thousands of near-identical frames that
inflate epoch time and bias the student toward wherever the vehicle lingered.

Play a bag first if you are not live. More data is strictly better; aim for a
few thousand frames spanning the scene variety you care about.

### 2. Check the student size you want

```bash
python3 distillation/defom_student.py
```

```
width 0.5 :  1194133 params    4.8 MB fp32
width 1.0 :  2212777 params    8.9 MB fp32
width 1.5 :  3871741 params   15.5 MB fp32
```

Add ~66.8 MB of reused stereo core to get the deployed size.

### 3. Train

```bash
python3 distillation/distill_defom.py --config distillation/configs/underwater_vitl.yaml
```

Overrides without editing the config:

```bash
python3 distillation/distill_defom.py \
  --config distillation/configs/underwater_vitl.yaml \
  --image_dir distillation/datasets/underwater \
  --epochs 30 --batch_size 2 --width 1.5
```

**Stop any running `demo_v2.py` / `demo_v3.py` first.** The teacher alone needs
~1.5 GB of weights plus activation, and end-to-end validation runs the full
model twice. Training was observed to OOM with only ~250 MB free.

Outputs to `paths.checkpoint_dir`: `student_epoch*.pth`, `training_log.csv`,
`resolved_config.json`.

### 4. Deploy

Load the vitl checkpoint, then swap the encoder:

```python
model = DEFOMStereo(args)                       # args.dinov2_encoder = 'vitl'
model.load_state_dict(torch.load('checkpoints/defomstereo_vitl_sceneflow.pth')['model'])

state = torch.load('distillation/checkpoints/.../student_epoch0019.pth')
# student_cfg and out_dim are stored in the checkpoint, so the student is
# rebuilt exactly as trained -- backbone included.
student = build_student(features=state['out_dim'], **state['student_cfg'])
student.load_state_dict(state['student'])      # calibration buffers come with it
model.defomencoder.depth_anything = student     # 1530.5 MB -> 75.6 MB
```

Everything downstream — `DefomEncoder.forward`, `fnet`, `cnet`, the GRU blocks
— is untouched, so `demo_v2.py` works as-is on the result.

---

## How the loss works, and the one thing that will bite you

Five teacher tensors are matched. Feature maps use the same reduction as your
DINOv3 distillation: **sum over channels, mean over batch and space**.

The default metric is `relative_mse`, and that default is load-bearing. The
teacher's outputs span five orders of magnitude:

| output | std | absmax |
|---|---:|---:|
| `d_features[0]` | 93.7 | 663 |
| `d_features[1]` | 281.2 | 1,551 |
| `d_features[2]` | 121.9 | 606 |
| `left_feat` / `right_feat` | **20,931** | **272,297** |
| `idepth` | 87.9 | 287 |

Under plain `mse` the left/right term evaluates around **1.1e11** while the
`d_features` terms sit near **1e6** — so with equal weights the student
optimises the matching features alone and ignores everything else. Dividing
each term by the teacher's own energy makes every term a dimensionless relative
error of order 1, so a well-scaled untrained student starts near `total ≈ 6.0`
(six terms × ~1.0) and the configured weights mean what they look like.

Only set `feature_metric: mse` if you deliberately want that imbalance.

`idepth` is matched **raw**, with L1. `DefomEncoder` normalises it by its own
per-sample max downstream, so normalising here would discard the scale the
teacher actually emits.

### Normalisation: GroupNorm, not BatchNorm

The teacher needs ~1.5 GB of weights plus activation, which caps the usable
batch size around 2-4. BatchNorm would see 4-8 samples per forward and its
running statistics never stabilise. Measured on a real in-distribution frame
with BatchNorm at batch 2:

| output | train vs eval relative difference |
|---|---:|
| `d_features[0]` | 0.45 |
| `left_feat` | 0.40 |
| `idepth` | 0.28 |

That is not cosmetic. Validation and deployment both run `eval()`, so a 28-45%
gap means they execute a measurably different function than the one being
optimised. With `norm: group` the measured train-vs-eval difference is exactly
`0.00e+00` -- GroupNorm has no running statistics and is batch-size
independent.

`norm: batch` is still selectable, but only makes sense at batch sizes this
setup cannot reach. The `mobilevit` backbone has BatchNorm inside the upstream
MobileViT blocks regardless; only the FPN head honours this setting.

### Output calibration — do not disable this

Balancing the loss terms is not sufficient on its own. The student's heads also
have to *reach* those magnitudes, and a freshly initialised head emits O(1)
against a `left_feat` target of ~2e4. The relative-loss gradient there is
roughly `-2/t ≈ 1e-4`, so that term simply does not train. Measured over
8 epochs without calibration:

| term | epoch 0 | epoch 7 | moved |
|---|---:|---:|---:|
| `left` | 0.999998 | 0.999985 | **0.001%** |
| `right` | 0.999998 | 0.999992 | **0.001%** |
| total | 5.9984 | 5.9801 | 0.3% |

So before training, `calibration_batches` teacher batches are sampled to
measure each output's mean and std, and those are stored as buffers in the
student. The heads then predict a **standardised** target and denormalise on
output — the external contract is unchanged, but each head only has to learn
an O(1) function. Same 8 epochs, with calibration:

| term | epoch 0 | epoch 6 | moved |
|---|---:|---:|---:|
| `left` | 1.045 | 0.844 | **19%** |
| `right` | 1.042 | 0.847 | **19%** |
| `idepth` | 0.428 | 0.082 | **81%** |
| total | 5.1698 | 3.8807 | 25% |

End-to-end disparity over the same run: within-3px **28% → 61%**, median
12.74 → 1.94 px.

The buffers travel in the checkpoint, and calibration is skipped on resume so a
resumed run keeps the statistics it was trained with.

---

## Validation split: temporal, not random

`val_split: temporal` holds out the last frames captured. This matters for the
same reason frame deduplication does: consecutive survey frames are
near-duplicates, so a random split drops near-copies of training frames into
validation and reports a number that looks good while measuring nothing about
generalisation. `capture_stereo.py` writes files in capture order, so the tail
is a genuinely unseen segment.

`random` is available if your frames are already independent.

## Validation

Feature loss is a proxy. What matters is end-to-end disparity, so validation
temporarily swaps the student into the real DEFOM model and compares its output
against the unmodified teacher on held-out frames — the same metric
`demo_v3.py` reports live:

```
[val vs teacher] mean 1.84 px | median 0.71 | p90 5.20 | within 1/2/3px 61%/74%/82%
```

The swap is in-place and reverted in a `finally`, so one model instance serves
both roles rather than holding a second 1.5 GB copy.

Use `demo_v3.py` for the final check — it runs the distilled model against vitl
and vits on the same live frame with a shared colour scale.

---

## Verified vs. not

Confirmed on this machine, on a free GPU:

- Output contract matches the teacher exactly for **both** backbones and all
  `mbconv` widths (`python3 distillation/defom_student.py` asserts this)
- Encoder surgery: 1530.5 MB → 75.6 MB, end-to-end disparity finite over the
  full frame
- Multi-epoch training converges: total loss 5.17 → 3.88, every one of the six
  terms decreasing
- End-to-end validation runs each epoch; disparity agreement improved
  within-3px 28% → 61%, median 12.74 → 1.94 px
- MobileViT backbone trains (5.17 → 4.45) and reports ~93.9 MB deployed
- Checkpoint round-trip: `strict=True` load, calibration buffers preserved
  (`fuse_scale` 20777.8), full deploy path rebuilt to 75.6 MB
- Resume restores epoch, optimiser and scheduler, and correctly skips
  recalibration
- Validation OOM is caught and skipped without losing the epoch

Not yet measured:

- Student inference speed vs vits (needs a trained student)
- Accuracy on a real dataset — all of the above used a 6-frame synthetic set
  purely to exercise the machinery. The absolute numbers mean nothing; only
  the fact that they move does.

Run step 3 on real data and watch the `[val vs teacher]` line.

---

## Tuning notes

| Symptom | Knob |
|---|---|
| CUDA OOM in training | `training.batch_size` to 1; each sample puts 2 images through the teacher |
| CUDA OOM only in validation | `training.val_max_frames` down, or `training.val_every` up — it is already non-fatal |
| Features match but disparity is off | Raise `loss.w_idepth` — the prior seeds the disparity field before the GRU refines it |
| Student underfits | `student.width` to 1.5, or more data |
| Epochs too slow | The teacher forward dominates; `vits` as teacher is ~7× faster, at lower teacher quality |
| A loss term sits at ~1.0 and will not move | Calibration did not run. Check for the `Calibrated output scales` line; raise `training.calibration_batches` |
| `mobilevit` backbone not found | Set `student.mobilevit_scripts_dir`, or use `backbone: mbconv` |
| Validation looks great but deployment does not | Likely `val_split: random` on sequential footage, or `norm: batch` |
