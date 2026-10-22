#!/usr/bin/env python3
"""Student encoder: a drop-in replacement for DepthAnythingV2 inside DEFOM-Stereo.

    forward(x, out_h, out_w) -> (d_features, left_feat, right_feat, idepth)

    x            [2B, 3, ih, iw]  left and right stacked on the batch dim
    d_features   3 maps, LEFT only, 256 ch, at (oh, ow), /2, /4
    left_feat    [B, 256, oh, ow]
    right_feat   [B, 256, oh, ow]
    idepth       [B, 1, oh, ow]   LEFT only, raw (DefomEncoder normalises it)

MobileViT-S backbone (mobilevit.py) tapped at /4, /8, /16, then a light FPN
and one head per output. Outputs are resized to (out_h, out_w), as the
teacher's DPT heads do, so any input size works. DEFOM's stereo core is
untouched. Run this file to check the output contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent


def _norm(channels, kind="group"):
    # GroupNorm: the teacher limits batches to 2-4, where BatchNorm made
    # train() and eval() outputs differ by 28-45% (measured).
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    groups = 32
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _conv_bn(inp, oup, stride=1, kernel=3, norm="group"):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, kernel // 2, bias=False),
        _norm(oup, norm),
        nn.SiLU(inplace=True),
    )


class MobileViTBackbone(nn.Module):
    """MobileViT-S, tapped at /4 (64 ch), /8 (96), /16 (128) and /16 (160).

    Layer names match the DINOv3->MobileViT distillation, so its
    student_mobilevit_*.pth checkpoints warm-start `self.net`. The input is
    resized up to a multiple of 32: the 2x2 patch unfold on the /16 map needs
    even sides.
    """

    MULTIPLE = 32

    def __init__(self, init_from=""):
        super().__init__()
        from mobilevit import mobilevit_s

        self.net = mobilevit_s()
        self.out_channels = (64, 96, 128, 160)
        if init_from:
            self.load_pretrained(init_from)

    def load_pretrained(self, path):
        """Warm start from a DINOv3-distilled MobileViT .pth (not the TorchScript .pt).
        Its conv2 (160 -> 1024 projection) is expected to be unused here."""
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(blob, dict):
            raise TypeError("%s is not a state dict (a TorchScript .pt?) -- use the .pth" % path)
        state = next((blob[k] for k in ("model_state_dict", "model", "student", "state_dict")
                      if isinstance(blob.get(k), dict)), blob)
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items() if isinstance(v, torch.Tensor)}
        own = self.net.state_dict()
        loaded = sum(1 for k, v in state.items() if k in own and own[k].shape == v.shape)
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        print("[MobileViTBackbone] warm start from %s: %d/%d tensors (missing %d, unexpected %d)"
              % (path, loaded, len(own), len(missing), len(unexpected)))
        # A wrong key or variant would otherwise load nothing and still "succeed".
        if loaded < 0.5 * len(own):
            raise RuntimeError("Only %d of %d tensors matched: not a MobileViT-S checkpoint" % (loaded, len(own)))

    def forward(self, x):
        h, w = x.shape[-2:]
        ph = (self.MULTIPLE - h % self.MULTIPLE) % self.MULTIPLE
        pw = (self.MULTIPLE - w % self.MULTIPLE) % self.MULTIPLE
        if ph or pw:
            x = F.interpolate(x, (h + ph, w + pw), mode="bilinear", align_corners=True)
        net = self.net
        y = net.mv2[0](net.conv1(x))
        f1 = net.mv2[3](net.mv2[2](net.mv2[1](y)))   # /4
        f2 = net.mvit[0](net.mv2[4](f1))             # /8
        f3 = net.mvit[1](net.mv2[5](f2))             # /16
        f4 = net.mvit[2](net.mv2[6](f3))             # /16 (mv2[6] is stride 1)
        return [f1, f2, f3, f4]


class DefomStudentEncoder(nn.Module):
    """MobileViT-S + FPN, emitting DepthAnythingV2's output tuple."""

    def __init__(self, features=256, fpn_dim=128, depth_dim=64, mobilevit_init_from="",
                 norm="group", freeze_bn=True):
        super().__init__()
        self.out_dim = features
        self.freeze_bn = freeze_bn
        self.backbone = MobileViTBackbone(init_from=mobilevit_init_from)
        c1, c2, c3, c4 = self.backbone.out_channels

        self.lat4 = nn.Conv2d(c4, fpn_dim, 1)
        self.lat3 = nn.Conv2d(c3, fpn_dim, 1)
        self.lat2 = nn.Conv2d(c2, fpn_dim, 1)
        self.lat1 = nn.Conv2d(c1, fpn_dim, 1)
        self.smooth3 = _conv_bn(fpn_dim, fpn_dim, norm=norm)
        self.smooth2 = _conv_bn(fpn_dim, fpn_dim, norm=norm)
        self.smooth1 = _conv_bn(fpn_dim, fpn_dim, norm=norm)

        # One head per output: the teacher's three d_features come from
        # different ViT depths, and the matching feature is shared by both eyes.
        self.rn1 = nn.Conv2d(fpn_dim, features, 1)
        self.rn2 = nn.Conv2d(fpn_dim, features, 1)
        self.rn3 = nn.Conv2d(fpn_dim, features, 1)
        self.fuse = nn.Sequential(_conv_bn(fpn_dim, fpn_dim, norm=norm), nn.Conv2d(fpn_dim, features, 1))
        # Predicts a standardised value (may be negative); clamped in forward().
        self.depth = nn.Sequential(
            _conv_bn(fpn_dim, depth_dim, norm=norm),
            nn.Conv2d(depth_dim, depth_dim // 2, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(depth_dim // 2, 1, 1),
        )

        # Output calibration: heads predict standardised values, mapped back to
        # the teacher's range here. The teacher's outputs span five orders of
        # magnitude (matching features ~1e7); without this the large terms get
        # ~1e-4 gradients and never train. Set by set_output_scales(), saved
        # with the checkpoint.
        for name in ("rn1", "rn2", "rn3", "fuse", "depth"):
            self.register_buffer("%s_shift" % name, torch.zeros(1))
            self.register_buffer("%s_scale" % name, torch.ones(1))

    def train(self, mode=True):
        """With freeze_bn, the backbone's BatchNorm stays in eval mode so training
        optimises the same function that is deployed (unfrozen: 19-22% gap).
        Its statistics are re-estimated before training (distill_defom.py)."""
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def _denorm(self, x, name):
        return x * getattr(self, "%s_scale" % name) + getattr(self, "%s_shift" % name)

    @torch.no_grad()
    def set_output_scales(self, stats):
        """stats: {'rn1': (mean, std), ...} measured from the teacher."""
        for name, (mean, std) in stats.items():
            getattr(self, "%s_shift" % name).fill_(float(mean))
            getattr(self, "%s_scale" % name).fill_(max(float(std), 1e-6))

    @staticmethod
    def _up_add(top, lateral):
        return lateral + F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x, out_h, out_w):
        half = x.shape[0] // 2
        f1, f2, f3, f4 = self.backbone(x)

        p4 = self.lat4(f4)
        p3 = self.smooth3(self._up_add(p4, self.lat3(f3)))
        p2 = self.smooth2(self._up_add(p3, self.lat2(f2)))
        p1 = self.smooth1(self._up_add(p2, self.lat1(f1)))

        def to(t, h, w):
            return F.interpolate(t, (h, w), mode="bilinear", align_corners=True)

        d_features = [
            self._denorm(to(self.rn1(p1), out_h, out_w), "rn1")[:half],
            self._denorm(to(self.rn2(p2), out_h // 2, out_w // 2), "rn2")[:half],
            self._denorm(to(self.rn3(p3), out_h // 4, out_w // 4), "rn3")[:half],
        ]
        fused = self._denorm(to(self.fuse(p1), out_h, out_w), "fuse")
        idepth = self._denorm(to(self.depth(p1), out_h, out_w), "depth")[:half].clamp_min(0.0)
        return d_features, fused[:half], fused[half:], idepth


def build_student(features=256, fpn_dim=128, depth_dim=64, mobilevit_init_from="", norm="group",
                  freeze_bn=True, backbone="mobilevit", mobilevit_variant="s", **_legacy):
    """Builds the student from a checkpoint's student_cfg. Older checkpoints
    also carry `width` and `mobilevit_scripts_dir`; they are ignored."""
    if backbone != "mobilevit" or mobilevit_variant != "s":
        raise ValueError("only the MobileViT-S student is supported (got %s/%s)" % (backbone, mobilevit_variant))
    return DefomStudentEncoder(features=features, fpn_dim=fpn_dim, depth_dim=depth_dim,
                               mobilevit_init_from=mobilevit_init_from, norm=norm, freeze_bn=freeze_bn)


def resolve_student_checkpoint(path, prefer_best=True):
    """A file is used as-is. A run folder gives its student_best.pth, else (or
    with prefer_best=False, for resuming) its newest student_epoch####.pth.
    Relative paths missing from the cwd resolve against the repo root."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    if p.is_file():
        return p
    if prefer_best and (p / "student_best.pth").is_file():
        return p / "student_best.pth"
    candidates = sorted(p.glob("student_epoch*.pth"), key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)))
    if not candidates:
        raise FileNotFoundError("No student_best.pth or student_epoch*.pth under %s" % p)
    return candidates[-1]


def load_student(path, device="cpu"):
    """Returns (student in eval mode, checkpoint dict, resolved path). Every
    weight comes from the checkpoint, so its warm-start path is not needed."""
    path = resolve_student_checkpoint(path)
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    student = build_student(features=ck["out_dim"], **dict(ck["student_cfg"], mobilevit_init_from=""))
    student.load_state_dict(ck["student"])
    return student.to(device).eval(), ck, path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_student().eval()
    n = count_parameters(model)
    print("mobilevit-s student: %d params, %.1f MB (deployed ~%.1f MB with the 66.8 MB stereo core)"
          % (n, n * 4 / 1e6, n * 4 / 1e6 + 66.8))
    with torch.no_grad():
        d, lf, rf, idp = model(torch.zeros(2, 3, 532, 700), 152, 200)
    shapes = [tuple(t.shape) for t in d] + [tuple(lf.shape), tuple(rf.shape), tuple(idp.shape)]
    expect = [(1, 256, 152, 200), (1, 256, 76, 100), (1, 256, 38, 50),
              (1, 256, 152, 200), (1, 256, 152, 200), (1, 1, 152, 200)]
    print("output shapes: %s" % shapes)
    print("all contracts OK" if shapes == expect else "CONTRACT MISMATCH -- do not train")
