#!/usr/bin/env python3
"""Slim DEFOM: a faster, configurable DEFOM-Stereo for embedded GPUs.

Same algorithm as core/defom_stereo.py -- monocular depth prior from the
student encoder, correlation lookup, scale then disparity GRU updates, convex
upsampling -- with every expensive part made configurable:

    stem         strides of the image branch (fnet/cnet): DEFOM (1,1,2,2) runs
                 its first layers at full resolution; (2,1,2,1) starts at 1/2
    enc_widths   fnet/cnet channel widths (DEFOM: 64, 96, 128)
    student_scale  student encoder input size relative to DEFOM's (1.0)
    encoder      "student" (the distilled MobileViT, default) or a Depth Anything
                 v2 ViT ("vits"), as in DEFOM's own releases; vit_scale shrinks
                 the ViT's input (its outputs keep their size)
    hidden       GRU hidden size (DEFOM: 128), n_gru_layers 2 or 3 (DEFOM: 3)
    motion_out / head   update-block inner widths (DEFOM: 128 / 256)

Configured with DEFOM's sizes and loaded with DEFOM's weights it reproduces
DEFOMStereo exactly (run this file to check). Parameter names match DEFOM's.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "core", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.corr import CorrBlock1D
from core.extractor import ConvBlock, ResidualBlock
from core.update import BasicMotionEncoder, ConvGRU, DispHead, pool2x, interp
from core.utils.utils import coords_grid, get_danv2_io_size

# "defom": DEFOM's own sizes (the equivalence check). The two deployed models (Orin NX, TensorRT fp16b):
PRESETS = {
    "defom": dict(stem=(1, 1, 2, 2), enc_widths=(64, 96, 128), student_scale=1.0,
                  hidden=128, n_gru_layers=3, motion_out=128, head=256),
    # slim v9: MobileViT student at 0.75x input, stride-2 slim fnet/cnet, GRU 64x2.
    # 800x600, 12/4 iterations: 172 ms (5.8 Hz)
    "v9": dict(stem=(2, 1, 2, 1), enc_widths=(32, 48, 64), student_scale=0.75,
               hidden=64, n_gru_layers=2, motion_out=64, head=96),
    # DEFOM's ViT-S release (defomstereo_vits_rvc.pth) exactly
    "vits": dict(encoder="vits", stem=(1, 1, 2, 2), enc_widths=(64, 96, 128), vit_scale=1.0,
                 hidden=128, n_gru_layers=3, motion_out=128, head=256),
    # the ViT-S release with fnet/cnet from 1/2 resolution; every tensor keeps the release's
    # shape, so the release's weights load unchanged (--make). 512x384, 10/4: 165 ms (6.05 Hz)
    "vits_s2": dict(encoder="vits", stem=(2, 1, 2, 1), enc_widths=(64, 96, 128), vit_scale=1.0,
                    hidden=128, n_gru_layers=3, motion_out=128, head=256),
}


def _layer(cin, cout, norm, stride):
    return nn.Sequential(ResidualBlock(cin, cout, norm, stride=stride), ResidualBlock(cout, cout, norm, stride=1))


class FeatureNet(nn.Module):
    """fnet: matching features of both images at 1/4, plus the depth features."""

    def __init__(self, d_dim, output_dim, widths, stem):
        super().__init__()
        w1, w2, w3 = widths
        s0, s1, s2, s3 = stem
        self.conv1 = nn.Conv2d(3, w1, 7, stride=s0, padding=3)
        self.norm1 = nn.InstanceNorm2d(w1)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = _layer(w1, w1, "instance", s1)
        self.layer2 = _layer(w1, w2, "instance", s2)
        self.layer3 = _layer(w2, w3, "instance", s3)
        self.convd = ConvBlock(d_dim, w3, "instance")
        self.conv2 = nn.Conv2d(w3, output_dim, 1)

    def forward(self, x, dfeats):
        n = x[0].shape[0]
        x, dfeats = torch.cat(x, 0), torch.cat(dfeats, 0)
        x = self.layer3(self.layer2(self.layer1(self.relu1(self.norm1(self.conv1(x))))))
        return self.conv2(x + self.convd(dfeats)).split(n, 0)


class ContextNet(nn.Module):
    """cnet: GRU initial states and context at 1/4, 1/8, 1/16."""

    def __init__(self, d_dim, hidden_dims, widths, stem, norm="batch"):
        super().__init__()
        w1, w2, w3 = widths
        s0, s1, s2, s3 = stem
        self.conv1 = nn.Conv2d(3, w1, 7, stride=s0, padding=3)
        self.norm1 = nn.BatchNorm2d(w1) if norm == "batch" else nn.InstanceNorm2d(w1)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = _layer(w1, w1, norm, s1)
        self.layer2 = _layer(w1, w2, norm, s2)
        self.layer3 = _layer(w2, w3, norm, s3)
        self.layer4 = _layer(w3, w3, norm, 2)
        self.layer5 = _layer(w3, w3, norm, 2)
        h = hidden_dims
        self.conv08 = ConvBlock(d_dim, w3, norm)
        self.outputs08 = nn.ModuleList([nn.Sequential(ResidualBlock(w3, w3, norm, 1), nn.Conv2d(w3, h[2], 3, padding=1))
                                        for _ in range(2)])
        self.conv16 = ConvBlock(d_dim, w3, norm)
        self.outputs16 = nn.ModuleList([nn.Sequential(ResidualBlock(w3, w3, norm, 1), nn.Conv2d(w3, h[1], 3, padding=1))
                                        for _ in range(2)])
        self.conv32 = ConvBlock(d_dim, w3, norm)
        self.outputs32 = nn.ModuleList([nn.Conv2d(w3, h[0], 3, padding=1) for _ in range(2)])

    def forward(self, x, d_feats):
        x = self.layer3(self.layer2(self.layer1(self.relu1(self.norm1(self.conv1(x))))))
        o08 = [f(x + self.conv08(d_feats[0])) for f in self.outputs08]
        y = self.layer4(x)
        o16 = [f(y + self.conv16(d_feats[1])) for f in self.outputs16]
        z = self.layer5(y)
        o32 = [f(z + self.conv32(d_feats[2])) for f in self.outputs32]
        return (o08, o16, o32)


class UpdateBlock(nn.Module):
    """DEFOM's (Scale)BasicMultiUpdateBlock with configurable inner widths."""

    def __init__(self, cor_planes, hidden_dims, n_gru_layers, motion_out, head, factor, scale):
        super().__init__()
        self.n_gru_layers, self.scale = n_gru_layers, scale
        self.encoder = BasicMotionEncoder(cor_planes, out_planes=motion_out)
        self.gru08 = ConvGRU(hidden_dims[2], motion_out + hidden_dims[1] * (n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=head, output_dim=1)
        self.mask = nn.Sequential(nn.Conv2d(hidden_dims[2], head, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(head, factor * factor * 9, 1, padding=0))

    def forward(self, net, inp, corr, disp):
        if self.n_gru_layers == 3:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if self.n_gru_layers >= 2:
            if self.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        motion = self.encoder(disp, corr)
        if self.n_gru_layers > 1:
            net[0] = self.gru08(net[0], *(inp[0]), motion, interp(net[1], net[0]))
        else:
            net[0] = self.gru08(net[0], *(inp[0]), motion)
        x = self.disp_head(net[0])
        out = F.relu6(torch.exp(.25 * x)) if self.scale else x
        return net, .25 * self.mask(net[0]), out


class StudentEncoder(nn.Module):
    """DefomEncoder around the distilled student, with a configurable input size."""

    def __init__(self, student, student_scale, idepth_scale):
        super().__init__()
        self.depth_anything = student
        student.input_scale = float(student_scale)     # the student shrinks its own input
        self.idepth_scale = idepth_scale
        self.out_dim = student.out_dim

    def forward(self, x, io_sizes):
        ih, iw, oh, ow = io_sizes
        x = F.interpolate(torch.cat(x, dim=0), (ih, iw), mode="bilinear", align_corners=True)
        features, left_feat, right_feat, idepth = self.depth_anything(x, oh, ow)
        bs = idepth.shape[0]
        max_idepth = torch.max(idepth.view(bs, -1), dim=1)[0].detach().view(bs, 1, 1, 1) + 1e-8
        idepth = idepth / max_idepth * self.idepth_scale * ow + 0.01
        return features, left_feat, right_feat, idepth


class ViTEncoder(nn.Module):
    """DEFOM's DefomEncoder (Depth Anything v2 ViT), with its input shrunk by
    vit_scale. Parameter names match DEFOM's, so its releases load as they are."""

    CONFIGS = {"vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
               "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
               "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])}

    def __init__(self, encoder, vit_scale, idepth_scale):
        super().__init__()
        from depth_anything_v2.dpt import DepthAnythingV2
        self.depth_anything = DepthAnythingV2(**self.CONFIGS[encoder])
        self.vit_scale, self.idepth_scale = float(vit_scale), idepth_scale
        self.out_dim = self.CONFIGS[encoder]["features"]

    def forward(self, x, io_sizes):
        ih, iw, oh, ow = io_sizes
        if self.vit_scale != 1.0:          # multiples of the 14-pixel patch
            ih, iw = (max(14, int(round(v * self.vit_scale / 14.0)) * 14) for v in (ih, iw))
        x = F.interpolate(torch.cat(x, dim=0), (ih, iw), mode="bilinear", align_corners=True)
        features, left_feat, right_feat, idepth = self.depth_anything(x, oh, ow)
        bs = idepth.shape[0]
        max_idepth = torch.max(idepth.view(bs, -1), dim=1)[0].detach().view(bs, 1, 1, 1) + 1e-8
        idepth = idepth / max_idepth * self.idepth_scale * ow + 0.01
        return features, left_feat, right_feat, idepth


class SlimDEFOM(nn.Module):
    def __init__(self, arch, slim, student=None):
        """arch: DEFOM arch dict (corr, scale_list, n_downsample, ...);
        slim: a PRESETS entry (or the same keys); student: a DefomStudentEncoder
        (not used when slim["encoder"] is a ViT)."""
        super().__init__()
        self.arch = argparse.Namespace(**arch)
        self.arch.mixed_precision = False
        self.slim = dict(slim)
        a, s = self.arch, self.slim
        hidden = [s["hidden"]] * 3
        self.register_buffer("mean", torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer("std", torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)
        if s.get("encoder", "student") == "student":
            self.defomencoder = StudentEncoder(student, s["student_scale"], a.idepth_scale)
        else:
            self.defomencoder = ViTEncoder(s["encoder"], s.get("vit_scale", 1.0), a.idepth_scale)
        d = self.defomencoder.out_dim
        self.fnet = FeatureNet(d, 256, s["enc_widths"], s["stem"])
        self.cnet = ContextNet(d, hidden, s["enc_widths"], s["stem"], norm=a.context_norm)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(hidden[i], hidden[i] * 3, 3, padding=1)
                                                for i in range(s["n_gru_layers"])])
        factor = 2 ** a.n_downsample
        self.update_block = UpdateBlock(a.corr_levels * (2 * a.corr_radius + 1), hidden, s["n_gru_layers"],
                                        s["motion_out"], s["head"], factor, scale=False)
        self.scale_update_block = UpdateBlock(len(a.scale_list) * (2 * a.scale_corr_radius + 1), hidden,
                                              s["n_gru_layers"], s["motion_out"], s["head"], factor, scale=True)

    @property
    def args(self):
        """DEFOMStereo's name for the arch settings (used by the exporter)."""
        return self.arch

    amp_dtype = torch.float16       # autocast precision (training may use torch.bfloat16)

    @property
    def mixed_precision(self):
        """Networks under autocast, correlation lookup and disparity maths in
        fp32 -- exactly DEFOM's mixed-precision regions. Off for export."""
        return bool(self.arch.mixed_precision)

    @mixed_precision.setter
    def mixed_precision(self, value):
        self.arch.mixed_precision = bool(value)

    def upsample_flow(self, flow, mask):
        N, D, H, W = flow.shape
        factor = 2 ** self.arch.n_downsample
        mask = torch.softmax(mask.view(N, 1, 9, factor, factor, H, W), dim=2)
        up = F.unfold(factor * flow, [3, 3], padding=1).view(N, D, 9, 1, 1, H, W)
        up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, scale_iters=4, test_mode=False):
        """Disparity of image1 (left). test_mode: the final full-resolution
        disparity; otherwise the full-resolution disparity after every iteration."""
        return self.refine(self.encode(image1, image2), iters, scale_iters, test_mode)

    def encode(self, image1, image2):
        """Everything before the refinement iterations: a dict with the matching
        features (fmap1, fmap2, 1/4), the GRU states (net) and context (inp) per
        level, and the monocular initial disparity (disp, 1/4)."""
        a = self.arch
        image1 = ((image1 - self.mean) / self.std).contiguous()
        image2 = ((image2 - self.mean) / self.std).contiguous()
        h, w = image1.shape[-2:]
        amp = torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.mixed_precision)
        with amp:
            d_features, dfeat1, dfeat2, disp = self.defomencoder([image1, image2],
                                                                 get_danv2_io_size(h, w, a.n_downsample))
            cnet_list = self.cnet(image1, d_features)
            fmap1, fmap2 = self.fnet([image1, image2], [dfeat1, dfeat2])
            net = [torch.tanh(x[0]) for x in cnet_list]
            inp = [torch.relu(x[1]) for x in cnet_list]
            inp = [list(conv(i).split(conv.out_channels // 3, dim=1)) for i, conv in zip(inp, self.context_zqr_convs)]
        inp += [None] * (3 - len(inp))
        return {"fmap1": fmap1, "fmap2": fmap2, "net": net, "inp": inp, "disp": disp}

    def refine(self, enc, iters=12, scale_iters=4, test_mode=False, states=None):
        """The refinement iterations on encode()'s output (see forward). A list
        passed as states receives the finest GRU state (1/4) after every iteration."""
        a = self.arch
        fmap1, fmap2, net, inp, disp = enc["fmap1"], enc["fmap2"], list(enc["net"]), enc["inp"], enc["disp"]
        coords = coords_grid(net[0].shape[0], net[0].shape[2], net[0].shape[3])[:, :1].to(fmap1.device)
        fmap1, fmap2, disp = fmap1.float(), fmap2.float(), disp.float()
        corr_fn = CorrBlock1D(fmap1, fmap2, coords, radius=a.corr_radius, num_levels=a.corr_levels,
                              scale_list=a.scale_list, scale_corr_radius=a.scale_corr_radius)
        limit = 2 ** (a.corr_levels - 1) * a.corr_radius
        predictions = []
        for itr in range(iters):
            disp = disp.detach()
            if itr < scale_iters:
                corr = corr_fn(disp, scaling=True)
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.mixed_precision):
                    net, up_mask, scale = self.scale_update_block(net, inp, corr, disp)
                disp = scale.float() * disp
            else:
                corr = corr_fn(disp, scaling=False)
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.mixed_precision):
                    net, up_mask, delta = self.update_block(net, inp, corr, disp)
                disp = disp + torch.clip(delta.float(), min=-limit, max=limit)
            if states is not None:
                states.append(net[0])
            if test_mode and itr < iters - 1:
                continue
            predictions.append(self.upsample_flow(disp, up_mask.float()))
        return predictions[-1] if test_mode else predictions


def build_slim(arch, slim, student):
    return SlimDEFOM(arch, slim, copy.deepcopy(student))


def load_defom_weights(model, path):
    """Copy every parameter of an original DEFOM-Stereo checkpoint whose name and
    shape match into model. Returns the names model keeps freshly initialised."""
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    state = state.get("model", state)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    own = model.state_dict()
    same = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(same, strict=False)
    return sorted(k for k in own if k not in same)


def load_slim(path, device="cpu", prefer_best=True):
    """Returns (SlimDEFOM in eval mode, checkpoint dict, resolved path). A run
    folder gives its student_best.pth (prefer_best=False: its newest epoch)."""
    from defom_student import build_student, resolve_student_checkpoint
    path = resolve_student_checkpoint(path, prefer_best=prefer_best)
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    if "slim" not in ck:
        raise ValueError("%s is not a slim DEFOM checkpoint" % path)
    student = None
    if ck["slim"].get("encoder", "student") == "student":
        student = build_student(features=ck["out_dim"], **ck["student_cfg"])
    model = SlimDEFOM(ck["config"]["arch"], ck["slim"], student)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck, path


def _check_equivalence():
    """SlimDEFOM with DEFOM's sizes and weights must equal DEFOMStereo exactly."""
    from common import DEFOM_ARCH
    from core.defom_stereo import DEFOMStereo
    from defom_student import build_student
    torch.manual_seed(0)
    student = build_student(features=256).eval()      # any encoder: both models get the same one
    arch = dict(DEFOM_ARCH)
    teacher_core = REPO / "checkpoints/defomstereo_vitl_middlebury.pth"
    state = torch.load(str(teacher_core), map_location="cpu", weights_only=True)
    state = state["model"] if "model" in state else state
    ref = DEFOMStereo(argparse.Namespace(**arch))
    ref.load_state_dict(state)
    del ref.defomencoder.depth_anything
    ref.defomencoder.depth_anything = student
    ref.args.mixed_precision = False
    slim = build_slim(arch, PRESETS["defom"], student)
    own = {k: v for k, v in state.items() if not k.startswith("defomencoder.")}
    missing, unexpected = slim.load_state_dict(own, strict=False)
    missing = [k for k in missing if not k.startswith("defomencoder.")]
    print("DEFOM weights into SlimDEFOM('defom'): missing %s, unexpected %s" % (missing, unexpected))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref, slim = ref.to(dev).eval(), slim.to(dev).eval()
    torch.manual_seed(0)
    l, r = torch.rand(1, 3, 480, 640, device=dev) * 255, torch.rand(1, 3, 480, 640, device=dev) * 255
    with torch.no_grad():
        a = ref(l, r, iters=12, scale_iters=4, test_mode=True)
        b = slim(l, r, iters=12, scale_iters=4, test_mode=True)
    print("SlimDEFOM('defom') vs DEFOMStereo: max |difference| %.3e px" % (a - b).abs().max().item())
    vits = REPO / "checkpoints/defomstereo_vits_rvc.pth"
    if vits.is_file():
        varch = dict(arch, dinov2_encoder="vits")
        ref_s = DEFOMStereo(argparse.Namespace(**varch))
        st = torch.load(str(vits), map_location="cpu", weights_only=True)
        ref_s.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in st.items()})
        ref_s.args.mixed_precision = False
        fast = SlimDEFOM(varch, PRESETS["vits"])
        fresh = load_defom_weights(fast, vits)
        ref_s, fast = ref_s.to(dev).eval(), fast.to(dev).eval()
        with torch.no_grad():
            d = (ref_s(l, r, iters=12, scale_iters=4, test_mode=True)
                 - fast(l, r, iters=12, scale_iters=4, test_mode=True)).abs().max().item()
        print("SlimDEFOM('vits') vs DEFOMStereo ViT-S release: %d fresh tensors, max |difference| %.3e px"
              % (len(fresh), d))
        diff_vits = d
    else:
        print("(%s not found: ViT-S check skipped)" % vits)
        diff_vits = 0.0
    for name in ("v9",):
        m = build_slim(arch, PRESETS[name], student).to(dev).eval()
        with torch.no_grad():
            out = m(l, r, iters=12, scale_iters=4, test_mode=True)
        core = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("defomencoder")) / 1e6
        print("preset %s: output %s, stereo core %.2f M parameters" % (name, tuple(out.shape), core))
    return max(float((a - b).abs().max()), diff_vits)


def make_release_checkpoint(release, preset, out_dir):
    """A slim-DEFOM checkpoint (loadable by load_slim, export_model.py, evaluate.py) that is an
    original DEFOM release with unchanged weights in a ViT preset whose tensors all keep the
    release's shapes -- e.g. vits_s2: fnet/cnet from 1/2 resolution, no training needed."""
    from common import DEFOM_ARCH
    encoder = PRESETS[preset].get("encoder")
    if not encoder:
        raise SystemExit("preset %s has no ViT encoder" % preset)
    arch = dict(DEFOM_ARCH, dinov2_encoder=encoder, mixed_precision=False)
    model = SlimDEFOM(arch, PRESETS[preset])
    fresh = load_defom_weights(model, release)
    if fresh:
        raise SystemExit("preset %s does not fit %s unchanged (%d new tensors, e.g. %s)" % (preset, release, len(fresh), fresh[0]))
    out = Path(out_dir)
    if (out / "student_best.pth").exists():
        raise SystemExit("%s already exists: pass another --out" % (out / "student_best.pth"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "slim": dict(PRESETS[preset]), "preset": preset, "epoch": "release",
                "out_dim": model.defomencoder.out_dim, "student_cfg": {"backbone": "dinov2_" + encoder},
                "note": "%s, unchanged weights, preset %s; no training" % (Path(release).name, preset),
                "config": {"arch": arch, "training": {"iters": 12, "scale_iters": 4},
                           "paths": {"teacher_ckpt": str(release)}}}, str(out / "student_best.pth"))
    print("wrote %s" % (out / "student_best.pth"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Without arguments: check SlimDEFOM against DEFOMStereo.")
    ap.add_argument("--make", metavar="PRESET", help="write an untrained checkpoint of a release in this preset (vits_s2)")
    ap.add_argument("--release", default=str(REPO / "checkpoints/defomstereo_vits_rvc.pth"))
    ap.add_argument("--out", default=str(REPO / "distillation/3_checkpoints/2_fast_vits_model"))
    a = ap.parse_args()
    if a.make:
        make_release_checkpoint(a.release, a.make, a.out)
    else:
        diff = _check_equivalence()
        print("EQUIVALENT" if diff == 0.0 else "NOT EQUIVALENT")
