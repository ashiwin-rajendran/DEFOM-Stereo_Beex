# DEFOM-Stereo student distillation — how to run

Replaces DEFOM-Stereo's 1.5 GB DepthAnythingV2 encoder with a 27 MB MobileViT
student, keeping DEFOM's trained stereo core: **1530 MB → ~94 MB**. Training
needs no ground truth — the student learns to reproduce the teacher's encoder
outputs on your own unlabelled stereo footage, from any camera and at any
resolution.

This page is the step-by-step guide. **Why** things work the way they do, and
the measured results, are in [ANALYSIS.md](ANALYSIS.md).

---

## Layout

```
distillation/
  distill_defom.py      train a student (any cameras / sizes mixed)
  evaluate.py           compare students against the teacher: report + charts
  defom_student.py      the student model, and checkpoint loading
  mobilevit.py          MobileViT backbone (self-contained copy)
  capture_stereo.py     record training frames from a ROS stereo topic
  live_compare.py       teacher vs student side by side on a live ROS stream
  configs/underwater_vitl.yaml   the training config

  datasets/underwater/<session>/        training frames            (git-ignored)
  checkpoints/<run>/                    one folder per run         (git-ignored)
      student_epoch####.pth   weights of every epoch (newest also holds optimizer state)
      student_best.pth        the epoch with the best validation  <- use this one
      best.json               which epoch that is, and its scores
      training_log.csv        loss and validation per epoch
  checkpoints/pretrained/               DINOv3 MobileViT warm start (from scratch only)
  evaluations/<name>/                   evaluate.py reports        (git-ignored)

demo_v3.py   (repo root)   ROS node running a student
demo_v2.py   (repo root)   ROS node running the full teacher
```

Everything runs from the repository root; relative paths in configs resolve
against it. Nothing outside this repository is needed.

---

## 0. Setup

**Python packages** (any machine): `torch`, `torchvision`, `numpy`,
`opencv-python`, `pillow`, `pyyaml`, `scipy`, `timm`, `opt_einsum`, `einops`;
plus `matplotlib` for `evaluate.py` charts. `xformers` is optional.

**Jetson Thor** (JetPack 7, Python 3.12, CUDA 13):

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

**Files to copy in** (git-ignored, so not in a clone):

| file | where |
|---|---|
| `defomstereo_vitl_middlebury.pth` (the teacher) | `checkpoints/` |
| the student run(s) you start from | `distillation/checkpoints/<run>/` |
| `student_mobilevit_2M_Dinov3_based_epoch0002.pth` (scratch only) | `distillation/checkpoints/pretrained/` |
| training frames | `distillation/datasets/underwater/<session>/` |

Check: `python3 distillation/defom_student.py` must end with `all contracts OK`.

---

## 1. Record training data

```bash
python3 distillation/capture_stereo.py --out distillation/datasets/underwater --max_frames 4000
python3 distillation/capture_stereo.py --out distillation/datasets/underwater --max_frames 4000 \
    --topic /ikan/voyis/stereo_concatenated/compressed --session voyis_dive3
```

- Each run writes its own session folder (default name `<eyeW>x<eyeH>_<time>`).
- Any side-by-side size works: DWE 1600×600 or 3200×1200, Voyis 5632×2816.
- Near-duplicate frames are skipped (`--min_mad 2.0`), so a static camera
  records almost nothing — record while the vehicle moves, underwater.
- Frames must be **rectified**.

---

## 2. Train

The config is set up for the next step: **fine-tune the best multi-scale
student on all the data in `datasets/underwater/`** (including anything new you
added), into a new run folder.

```bash
python3 -u distillation/distill_defom.py --config distillation/configs/underwater_vitl.yaml \
    2>&1 | tee distill_finetune.log
```

On the Thor, run it inside `tmux` so it survives a disconnect.

Other cases, by command-line override:

| goal | add |
|---|---|
| fine-tune from another student | `--init_from distillation/checkpoints/<run>` (a folder uses its `student_best.pth`) |
| keep each run separate | `--checkpoint_dir distillation/checkpoints/<new_run_name>` |
| continue an **interrupted** run | `--resume_from distillation/checkpoints/<run> --checkpoint_dir distillation/checkpoints/<run>` (a folder uses its newest epoch; `init_from` is then ignored) |
| more / fewer epochs, other learning rate | `--epochs 15`, `--lr 5e-5` |
| other data folder | `--image_dir /path/to/sessions` |

Use `--init_from` (fine-tune) to add data; use `--resume_from` only to finish
an interrupted run, never from `student_best.pth` (the trainer refuses). The
difference is explained in [ANALYSIS.md](ANALYSIS.md#fine-tuning-vs-resuming).
Give every fine-tune its own `--checkpoint_dir`, so an earlier run's
`student_best.pth` is never overwritten.

**What you will see:**

```
Validating the starting checkpoint ...
  [val vs teacher] 800x600    @ 512  mean 2.29 px | median 1.91 | p90 4.27 | within 1/2/3px 70%/79%/82%
  new best: start (val score 77.12%) -> student_best.pth
  epoch 0  step  20/2714  total 1.20  idepth 0.11  lr 1.00e-04  (0.62 s/step, 0.00 s waiting for data)
  ...
  new best: epoch 3 (val score 78.40%) -> student_best.pth        # or: best remains ...
```

- **val score** = mean "within 3 px" over every camera and size; higher is
  better. `student_best.pth` is updated only when it improves — including
  over the starting checkpoint, so a fine-tune can never make it worse.
- **waiting for data** near 0 means the GPU is the limit (expected). If it
  grows, raise `training.num_workers`.
- Speed: ~0.53 s/step on an RTX 4070, ~0.62 s/step on the Thor
  (`teacher_bf16` + `tf32`); an epoch over ~11k frames is ~25–30 min.

---

## 3. Evaluate

Compare any students against the teacher on held-out frames, with a report:

```bash
python3 distillation/evaluate.py \
    --models orig=distillation/checkpoints/underwater_vitl_middlebury_to_student/student_epoch0049.pth \
             multiscale=distillation/checkpoints/underwater_vitl_middlebury_to_student_multiscale \
             finetune=distillation/checkpoints/underwater_vitl_middlebury_to_student_finetune \
    --out distillation/evaluations/finetune_vs_multiscale
```

- A directory picks its `student_best.pth`. Name models `name=path`.
- The first model is the **reference** the others are compared to
  (`--reference` to change it).
- New dive the students never saw: `--image_dir /path/to/new_sessions --frames all`.
- ~10–15 min for 4 models × 80 frames per camera × 2 sizes on a 4070.

Open `distillation/evaluations/<name>/report.md`. It has:

| section | tells you |
|---|---|
| **Verdict** | the best model, and whether each beats the reference beyond noise |
| **Ranking** | overall score with 95% CI, paired gain vs the reference, P(better), speed |
| **Per camera and size** | where each model is strong or weak |
| `summary.png` | agreement and depth error per camera/size, with error bars |
| `nearness.png` | depth error from far to near |
| `texture.png` | agreement on structure vs open water |
| `training_curves_<run>.png` | loss and validation per epoch, best epoch marked |
| `samples_<camera>.jpg` | left / teacher / each model's disparity, and their error maps |

Rule of thumb: trust a difference only when the report says **clearly
better** (P(better) above 97.5%).

---

## 4. Deploy

Export to TorchScript with the ROS package's exporter (in
`defom_stereo_pointcloud/scripts/`), on the machine that will run it:

```bash
CK=$DEFOM_REPO/distillation/checkpoints/<run>/student_best.pth
python3 scripts/export_torchscript.py --repo $DEFOM_REPO --student_ckpt $CK --half --name defom_<run>_640x480_fp16
python3 scripts/export_torchscript.py --repo $DEFOM_REPO --student_ckpt $CK --half --width 544 --height 544 --name defom_<run>_544x544_fp16
```

640×480 for the DWE camera, 544×544 for Voyis.

Check a student live on the robot's stream:

```bash
python3 demo_v3.py --student_ckpt distillation/checkpoints/<run>        # student as a ROS node
python3 distillation/live_compare.py --student_ckpt distillation/checkpoints/<run>   # vs teacher, live
```

Without `--student_ckpt`, both use the multi-scale run's `student_best.pth`
(epoch 20).

---

## Config quick reference

| key | default | meaning |
|---|---|---|
| `paths.init_from` | multi-scale `student_best.pth` | student to start from (fine-tune) |
| `paths.checkpoint_dir` | `..._finetune` | output folder; one per run |
| `training.epochs` / `lr` | 10 / 1e-4 | fine-tune length and rate |
| `training.batch_size` | 4 | pairs per step (each runs 2 images through the teacher) |
| `training.val_max_frames` | 80 | held-out frames validated per camera |
| `training.teacher_bf16` | true | ~1.7× faster teacher; ignored on GPUs without bf16 (Xavier) |
| `training.tf32` | true | faster fp32 matmuls, same accuracy |
| `data.long_side` / `crop` | [512, 1024] / [384, 512] | training scales and crop |
| `data.balance_sources` / `max_repeat` | true / 4 | equal share per camera, capped |
| `student.backbone` | mobilevit | `mbconv` is a smaller, self-contained alternative |

## Troubleshooting

| symptom | fix |
|---|---|
| CUDA out of memory in training | `training.batch_size` 2; stop other GPU processes |
| out of memory in validation only | lower `training.val_max_frames`; it is skipped, not fatal |
| `best.json was scored on different validation frames` | expected after adding data: the held-out set changed, so the old best is not comparable and tracking restarts |
| `libnvpl_*` / `libcudss` not found (Thor) | the NVPL / cuDSS steps in Setup |
| a loss term stuck near 1.0 | output calibration did not run — training from scratch needs `training.calibration_batches` > 0 |
| `No student_best.pth or student_epoch*.pth` | wrong run folder |
