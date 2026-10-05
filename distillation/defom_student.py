#!/usr/bin/env python3
"""Lightweight student encoder: a drop-in replacement for DepthAnythingV2
inside DEFOM-Stereo's DefomEncoder.

WHY A PURPOSE-BUILT STUDENT RATHER THAN THE EXISTING MobileViT
---------------------------------------------------------------
The DINOv3 -> MobileViT distillation in Yolo_Seg/vit_training targets a single
tensor: `x_norm_patchtokens` -> [B, G, G, 1024], L2-normalised, square input,
patch 16. None of that holds here:

  * DEFOM needs FOUR outputs, not one, and they feed two different heads.
  * DINOv2 is patch-14, and MobileViT's stride product is a power of two, so
    it cannot produce a /14 grid by restriding.
  * Input is non-square (532x700 at your 800x600 stream) and its size CHANGES
    with input resolution via get_danv2_io_size().
  * The DPT heads consume raw features; L2-normalising destroys the magnitude
    they rely on.

So this student matches DepthAnythingV2's *interface* instead, and sidesteps
the grid problem entirely by resizing to the requested (out_h, out_w) at the
end -- exactly what the teacher's DPT heads already do internally.

INTERFACE (verified against depth_anything_v2/dpt.py)
------------------------------------------------------
    forward(x, out_h, out_w) -> (d_features, left_feat, right_feat, idepth)

    x            [2B, 3, ih, iw]   left and right concatenated on the batch dim
    d_features   list of 3, LEFT only, 256ch, at (oh,ow), (oh/2,ow/2), (oh/4,ow/4)
    left_feat    [B, 256, oh, ow]
    right_feat   [B, 256, oh, ow]
    idepth       [B, 1, oh, ow]    LEFT only, RAW -- DefomEncoder applies its own
                                   max-normalisation afterwards, so do not
                                   normalise here

Swapping this in leaves DefomEncoder.forward, fnet, cnet and the GRU update
blocks completely untouched, so the trained stereo core from the vitl
checkpoint is reused as-is.

BACKBONES
---------
    mbconv      self-contained MBConv/FPN. Zero dependencies outside this
                folder. 8.9 MB at width 1.0.
    mobilevit   reuses MobileViT-S from Yolo_Seg/vit_training/scripts, the
                same architecture as the DINOv3->MobileViT distillation.
                24.3 MB. Imports across repos, which is the point: it is the
                single source of truth for that architecture, so a warm start
                from an existing student_mobilevit_*.pth stays layer-name
                compatible.

MobileViT needs its input divisible by 16 and DEFOM feeds 532x700, so the
wrapper resizes to the next multiple (544x704) on the way in. Both backbones
expose four feature maps and share the same FPN head, so the output contract
above is identical either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels, kind="group"):
    """Normalisation layer.

    GroupNorm is the default, and the reason is measured. The teacher needs
    ~1.5 GB of weights plus activation, which caps the usable batch size
    around 2-4 -- so BatchNorm would see only 4-8 samples per forward and its
    running statistics never stabilise. Measured with BatchNorm at batch 2,
    on a real in-distribution frame, train() and eval() outputs diverged by
    28-45% relative. That matters twice over: validation would score a
    different function than the one being optimised, and deployment runs
    eval().

    GroupNorm has no running statistics and is batch-size independent, so
    train and eval are identical by construction.
    """
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    groups = 32
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _conv_bn(inp, oup, stride=1, kernel=3, norm="group"):
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, pad, bias=False),
        _norm(oup, norm),
        nn.SiLU(inplace=True),
    )


class MBConv(nn.Module):
    """Inverted residual block (MobileNetV2 style)."""

    def __init__(self, inp, oup, stride=1, expansion=4, norm="group"):
        super().__init__()
        hidden = int(inp * expansion)
        self.use_residual = stride == 1 and inp == oup
        layers = []
        if expansion != 1:
            layers += [nn.Conv2d(inp, hidden, 1, 1, 0, bias=False),
                       _norm(hidden, norm), nn.SiLU(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            _norm(hidden, norm), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),
            _norm(oup, norm),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_residual else self.block(x)


def _stage(inp, oup, blocks, stride, expansion, norm="group"):
    layers = [MBConv(inp, oup, stride, expansion, norm)]
    layers += [MBConv(oup, oup, 1, expansion, norm) for _ in range(blocks - 1)]
    return nn.Sequential(*layers)


class MBConvBackbone(nn.Module):
    """Self-contained MBConv trunk. Emits /4, /8, /16, /32."""

    def __init__(self, width=1.0, in_channels=3, norm="group"):
        super().__init__()
        c = [max(8, int(round(v * width))) for v in (24, 32, 64, 128, 192)]
        c = [v + (-v % 8) for v in c]   # keep channels divisible for GroupNorm
        self.stem = _conv_bn(in_channels, c[0], stride=2, norm=norm)                  # /2
        self.stage1 = _stage(c[0], c[1], blocks=2, stride=2, expansion=2, norm=norm)  # /4
        self.stage2 = _stage(c[1], c[2], blocks=3, stride=2, expansion=4, norm=norm)  # /8
        self.stage3 = _stage(c[2], c[3], blocks=4, stride=2, expansion=4, norm=norm)  # /16
        self.stage4 = _stage(c[3], c[4], blocks=3, stride=2, expansion=4, norm=norm)  # /32
        self.out_channels = (c[1], c[2], c[3], c[4])

    def forward(self, x):
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class MobileViTBackbone(nn.Module):
    """MobileViT-S from Yolo_Seg/vit_training, tapped for intermediates.

    The upstream `_extract_features` is a flat chain that returns only the
    final /16 map, so the stages are re-run here explicitly to expose /4, /8
    and /16. Nothing in the upstream file is modified -- this only reads its
    submodules, so an existing student_mobilevit_*.pth still loads into
    `self.net` by its original layer names.

    Measured taps at 544x704 input:
        mv2[1..3] ->  64 ch  /4
        mvit[0]   ->  96 ch  /8
        mvit[1]   -> 128 ch  /16
        mvit[2]   -> 160 ch  /16   (mv2[6] is stride-1 upstream)
    """

    MULTIPLE = 16

    def __init__(self, scripts_dir=None, variant="s", in_channels=3, init_from=""):
        super().__init__()
        scripts_dir = Path(scripts_dir) if scripts_dir else _default_scripts_dir()
        if not scripts_dir.is_dir():
            raise FileNotFoundError(
                "MobileViT backbone needs Yolo_Seg/vit_training/scripts. Not found at %s. "
                "Set student.mobilevit_scripts_dir in the config, or use "
                "student.backbone: mbconv to avoid the dependency entirely." % scripts_dir)
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        try:
            import mobilevit_distill as mvd
        except ImportError as exc:
            raise ImportError("Could not import mobilevit_distill from %s: %s" % (scripts_dir, exc))

        builder = {"s": mvd.mobilevit_s_distill,
                   "xs": mvd.mobilevit_xs_distill,
                   "xxs": mvd.mobilevit_xxs_distill}[variant]
        # image_size only fixes the positional assert, not runtime shape --
        # verified: a model built at 544x704 runs at 608x800 unchanged.
        self.variant = variant
        self.net = builder(image_size=(544, 704), in_channels=in_channels)
        # conv2 maps to the 1024-d DINOv3 target, which is irrelevant here --
        # the FPN head consumes the stage features instead. Dropping it saves
        # 0.16 M params and avoids a dead branch in the graph.
        self.net.conv2 = nn.Identity()
        self.out_channels = (64, 96, 128, 160)

        if init_from:
            self.load_pretrained(init_from)

    def load_pretrained(self, path):
        """Warm start from a DINOv3-distilled MobileViT checkpoint.

        Those checkpoints wrap the weights under `model_state_dict`, not the
        `model`/`student` key used elsewhere in this repo, so the key is
        searched for explicitly. Getting that wrong loads nothing at all while
        still "succeeding", which is exactly the kind of silent no-op that
        wastes a training run -- hence the hard check at the end.

        `conv2` (160 -> 1024) is reported as unexpected by design: it projects
        to the DINOv3 target dimension, which nothing here consumes. The FPN
        head taps mvit[2] at 160 channels upstream of it, so it is replaced
        with Identity in __init__.
        """
        blob = torch.load(path, map_location="cpu")
        if not isinstance(blob, dict):
            raise TypeError(
                "%s is a %s, not a state dict. The .pt files in these checkpoint folders are "
                "TorchScript exports -- use the matching .pth instead."
                % (path, type(blob).__name__))
        state = None
        for key in ("model_state_dict", "model", "student", "state_dict"):
            if key in blob and isinstance(blob[key], dict):
                state = blob[key]
                break
        if state is None:
            state = blob
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items() if isinstance(v, torch.Tensor)}

        own = self.net.state_dict()
        loaded = sum(1 for k, v in state.items() if k in own and own[k].shape == v.shape)
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        print("[MobileViTBackbone] warm start from %s" % path)
        print("    loaded %d/%d tensors  (missing %d, unexpected %d -- conv2/fc expected here)"
              % (loaded, len(own), len(missing), len(unexpected)))
        if loaded < 0.5 * len(own):
            raise RuntimeError(
                "Only %d of %d tensors matched. The checkpoint does not fit this MobileViT "
                "variant -- check student.mobilevit_variant (this is '%s')."
                % (loaded, len(own), self.variant))

    def forward(self, x):
        h, w = x.shape[-2:]
        ph = (self.MULTIPLE - h % self.MULTIPLE) % self.MULTIPLE
        pw = (self.MULTIPLE - w % self.MULTIPLE) % self.MULTIPLE
        if ph or pw:
            x = F.interpolate(x, (h + ph, w + pw), mode="bilinear", align_corners=True)

        net = self.net
        y = net.mv2[0](net.conv1(x))
        y = net.mv2[3](net.mv2[2](net.mv2[1](y)))
        f1 = y                                   # /4   64
        y = net.mvit[0](net.mv2[4](y))
        f2 = y                                   # /8   96
        y = net.mvit[1](net.mv2[5](y))
        f3 = y                                   # /16  128
        y = net.mvit[2](net.mv2[6](y))
        f4 = y                                   # /16  160
        return [f1, f2, f3, f4]


def _default_scripts_dir():
    # DEFOM_Stereo/DEFOM-Stereo_Beex/distillation -> workspace root
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Yolo_Seg" / "vit_training" / "scripts"
        if candidate.is_dir():
            return candidate
    return here.parent / "mobilevit_scripts"


class DefomStudentEncoder(nn.Module):
    """MBConv backbone + light FPN, emitting DepthAnythingV2's output tuple.

    `width` scales the whole network. The defaults land around 5 M parameters
    (~20 MB fp32); combined with DEFOM's ~67 MB stereo core that gives a
    ~87 MB deployed model, against 173 MB for the official vits and 1530 MB
    for vitl.
    """

    def __init__(self, features=256, width=1.0, fpn_dim=128, depth_dim=64,
                 backbone="mbconv", mobilevit_scripts_dir=None, mobilevit_variant="s",
                 mobilevit_init_from="", norm="group"):
        super().__init__()
        self.out_dim = features
        self.backbone_name = backbone

        if backbone == "mbconv":
            self.backbone = MBConvBackbone(width=width, norm=norm)
        elif backbone == "mobilevit":
            self.backbone = MobileViTBackbone(scripts_dir=mobilevit_scripts_dir,
                                              variant=mobilevit_variant,
                                              init_from=mobilevit_init_from)
        else:
            raise ValueError("backbone must be 'mbconv' or 'mobilevit', got %r" % backbone)
        c1, c2, c3, c4 = self.backbone.out_channels

        self.lat4 = nn.Conv2d(c4, fpn_dim, 1)
        self.lat3 = nn.Conv2d(c3, fpn_dim, 1)
        self.lat2 = nn.Conv2d(c2, fpn_dim, 1)
        self.lat1 = nn.Conv2d(c1, fpn_dim, 1)
        self.smooth3 = _conv_bn(fpn_dim, fpn_dim, norm=norm)
        self.smooth2 = _conv_bn(fpn_dim, fpn_dim, norm=norm)
        self.smooth1 = _conv_bn(fpn_dim, fpn_dim, norm=norm)

        # Three reassembly heads, one per d_features scale. Separate weights
        # because the teacher's three maps come from different ViT depths and
        # are genuinely different representations, not resized copies.
        self.rn1 = nn.Conv2d(fpn_dim, features, 1)
        self.rn2 = nn.Conv2d(fpn_dim, features, 1)
        self.rn3 = nn.Conv2d(fpn_dim, features, 1)

        # Fused matching feature, applied to BOTH images (siamese by virtue of
        # left/right sharing the batch dim).
        self.fuse = nn.Sequential(_conv_bn(fpn_dim, fpn_dim, norm=norm), nn.Conv2d(fpn_dim, features, 1))

        # OUTPUT DENORMALISATION -- load-bearing, not cosmetic.
        #
        # The teacher's outputs are wildly out of scale with each other
        # (left/right std ~20931 and absmax ~272297, against d_features std
        # ~94-281). A freshly initialised head emits O(1), so to hit a target
        # of ~2e4 the relative-loss gradient is about -2/t ~ 1e-4: training
        # stalls at relative error 1.0 no matter how long it runs. Measured:
        # 8 epochs moved the left/right term by 0.001%.
        #
        # So the heads predict a STANDARDISED target, and these buffers map
        # back to the teacher's real range here in forward(). The external
        # contract is unchanged -- callers still get teacher-scale tensors --
        # but the head itself only ever has to learn an O(1) function.
        #
        # Calibrated once from the teacher by calibrate_output_scales();
        # they are buffers, so they travel in the checkpoint.
        for name in ("rn1", "rn2", "rn3", "fuse", "depth"):
            self.register_buffer("%s_shift" % name, torch.zeros(1))
            self.register_buffer("%s_scale" % name, torch.ones(1))
        self.calibrated = False

        self.depth = nn.Sequential(
            _conv_bn(fpn_dim, depth_dim, norm=norm),
            nn.Conv2d(depth_dim, depth_dim // 2, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(depth_dim // 2, 1, 1),
            # No ReLU: the head predicts a STANDARDISED value, which is
            # legitimately negative below the mean. Non-negativity of the
            # final inverse depth is restored by the clamp in forward().
        )

    def _denorm(self, x, name):
        return x * getattr(self, "%s_scale" % name) + getattr(self, "%s_shift" % name)

    @torch.no_grad()
    def set_output_scales(self, stats):
        """stats: {'rn1': (mean, std), ...} measured from the teacher."""
        for name, (mean, std) in stats.items():
            getattr(self, "%s_shift" % name).fill_(float(mean))
            getattr(self, "%s_scale" % name).fill_(max(float(std), 1e-6))
        self.calibrated = True

    @staticmethod
    def _up_add(top, lateral):
        return lateral + F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x, out_h, out_w):
        batch = x.shape[0]
        half = batch // 2

        f1, f2, f3, f4 = self.backbone(x)

        p4 = self.lat4(f4)
        p3 = self.smooth3(self._up_add(p4, self.lat3(f3)))
        p2 = self.smooth2(self._up_add(p3, self.lat2(f2)))
        p1 = self.smooth1(self._up_add(p2, self.lat1(f1)))

        def to(t, h, w):
            return F.interpolate(t, (h, w), mode="bilinear", align_corners=True)

        # Teacher emits these for the LEFT image only (DPTFeat: `[:bs//2]`).
        d_features = [
            self._denorm(to(self.rn1(p1), out_h, out_w), "rn1")[:half],
            self._denorm(to(self.rn2(p2), out_h // 2, out_w // 2), "rn2")[:half],
            self._denorm(to(self.rn3(p3), out_h // 4, out_w // 4), "rn3")[:half],
        ]

        fused = self._denorm(to(self.fuse(p1), out_h, out_w), "fuse")
        left_feat, right_feat = fused[:half], fused[half:]

        # RAW inverse depth: DefomEncoder divides by its own max afterwards.
        # Clamped non-negative, which the dropped ReLU used to guarantee.
        idepth = self._denorm(to(self.depth(p1), out_h, out_w), "depth")[:half]
        idepth = idepth.clamp_min(0.0)

        return d_features, left_feat, right_feat, idepth


def build_student(features=256, width=1.0, fpn_dim=128, depth_dim=64,
                  backbone="mbconv", mobilevit_scripts_dir=None,
                  mobilevit_variant="s", mobilevit_init_from="", norm="group"):
    return DefomStudentEncoder(features=features, width=width, fpn_dim=fpn_dim,
                               depth_dim=depth_dim, backbone=backbone,
                               mobilevit_scripts_dir=mobilevit_scripts_dir,
                               mobilevit_variant=mobilevit_variant,
                               mobilevit_init_from=mobilevit_init_from, norm=norm)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _report(label, model):
    n = count_parameters(model)
    # DEFOM's stereo core (fnet/cnet/GRU) is reused from the vitl checkpoint.
    core_mb = 66.8
    print("%-22s %9d params  %6.1f MB   deployed ~%.1f MB"
          % (label, n, n * 4 / 1e6, n * 4 / 1e6 + core_mb))
    x = torch.zeros(2, 3, 532, 700)
    with torch.no_grad():
        d, lf, rf, idp = model(x, 152, 200)
    expect = [(1, 256, 152, 200), (1, 256, 76, 100), (1, 256, 38, 50)]
    ok = ([tuple(t.shape) for t in d] == expect
          and tuple(lf.shape) == (1, 256, 152, 200)
          and tuple(rf.shape) == (1, 256, 152, 200)
          and tuple(idp.shape) == (1, 1, 152, 200))
    print("    contract: %s   d_features %s  left/right %s  idepth %s"
          % ("OK" if ok else "MISMATCH", [tuple(t.shape) for t in d],
             tuple(lf.shape), tuple(idp.shape)))
    return ok


if __name__ == "__main__":
    print("teacher encoder (DepthAnythingV2-L) is 1463.7 MB for reference\n")
    all_ok = True
    for w in (0.5, 1.0, 1.5):
        all_ok &= _report("mbconv width %.1f" % w, build_student(width=w))
    try:
        all_ok &= _report("mobilevit-s", build_student(backbone="mobilevit"))
    except (FileNotFoundError, ImportError) as exc:
        print("mobilevit-s           unavailable: %s" % exc)
    print("\nall contracts OK" if all_ok else "\nCONTRACT MISMATCH -- do not train")
