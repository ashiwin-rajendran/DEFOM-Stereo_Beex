#!/usr/bin/env python3
"""Compare DEFOM models on BeeX recordings (DWE or Voyis, rosbag or live), on the PC.

    rosbag play --loop --topics /ikan/explore3d/depth_image -- <dive>.beex
    python3 compare_beex_bags.py --show --input /ikan/explore3d/depth_image                 # DWE
    python3 compare_beex_bags.py --show --input /ikan/voyis/stereo_concatenated/compressed  # Voyis

Messages are decoded with the definition the bag/publisher sends, so no
bx_msgs workspace needs to be sourced (and a different bx_msgs version is fine).

--input, the stereo pair (must be rectified):
    bx_msgs/DepthImage           DWE (sensor_type 116): 800x600 left|right + refined_pro depth
    sensor_msgs/CompressedImage  side-by-side left|right (Voyis stereo_concatenated: 5632x2816;
                                 not stereo_concatenated_raw, which is not rectified)
    sensor_msgs/Image            side-by-side left|right
    bx_msgs/StereoImage          left_data / right_data JPEGs
--ref, the camera's own depth when --input has none, matched to each frame by time:
    Voyis: /ikan/voyis/stereo/data (RGBD stream, float EXR metres). Default for Voyis inputs.
Camera: --camera, else DepthImage.sensor_type (116 DWE, 115 Voyis), else the topic name.

Models (listed, numbered, at start):
  ours       every exported <name>.json + .onnx under distillation/5_output_models/<model>/; it runs
             from the checkpoint in distillation/3_checkpoints/<model>/ (same folder name), with the
             export's iterations, at the frame's aspect (or the export's size if smaller)
  baselines  every original release checkpoints/defomstereo_*.pth (ViT-L = the teacher), at 32/8
Switch: keys 1-9 / n / p in the window (--show), or ~select by number or part of the name.

Publishes (node defom_compare): ~compare (left | camera depth | model depth), ~depth (32FC1 m),
~depth/color, ~points, ~camera_info, ~ref/depth/color, ~ref/points.
Live controls: ~select, ~range "MIN MAX" (metres; model depth outside -> invalid), ~texture T
(model depth where the image has no texture -> invalid); "data: 'off'" clears them. Write values as
"data: '...'" (rostopic reads a bare 6 as a number, off as false). See distillation/README.md, section 7.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def import_cv2():
    """DWE's depth is EXR. ROS's system OpenCV reads it; the pip build may not."""
    system = "/usr/lib/python3/dist-packages"
    if os.path.isdir(system) and any(p.startswith("cv2") for p in os.listdir(system)):
        sys.path.insert(0, system)
        try:
            import cv2
        finally:
            sys.path.remove(system)
    else:
        import cv2
    return cv2


cv2 = import_cv2()

import collections  # noqa: E402

import genpy.dynamic  # noqa: E402
import rospy  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import Header, String  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = str(HERE.parent.parent)                          # DEFOM-Stereo_Beex (this file: distillation/7_ros_tools/)

# Rectified intrinsics at each camera's calibration size; scaled to the frame size.
SENSORS = {115: "voyis", 116: "dwe"}                     # bx_msgs/DepthImage sensor_type
REF_NAMES = {"dwe": "DWE refined_pro depth", "voyis": "Voyis depth"}
CAMERAS = {
    # e3d1008.dwecal at its 800x600 stream; baseline from the explore3D drawing
    "dwe": dict(width=800, fx=479.266, fy=478.810, cx=397.983, cy=322.841, baseline=0.1, frame_id="explore3d"),
    # Voyis 180114006 (AgisoftParams), 2816x2816 per eye
    "voyis": dict(width=2816, fx=1884.6483, fy=1884.6483, cx=1466.4675, cy=1384.1693, baseline=0.1700411,
                  frame_id="voyis"),
}


# ---------------------------------------------------------------- models
class TorchModel:
    """The checkpoint an export came from, with its iterations, on the GPU, at any input size."""

    def __init__(self, meta, folder, repo):
        from export_model import build_model
        ckpt = source_checkpoint(meta, folder, repo)
        model, ck, _ = build_model(str(ckpt))
        if str(ck.get("epoch")) != str(meta.get("source_epoch")):
            rospy.logwarn("%s: checkpoint epoch %s, export says %s", meta["name"], ck.get("epoch"),
                          meta.get("source_epoch"))
        self.setup(model, meta["iters"], meta["scale_iters"])

    def setup(self, model, iters, scale_iters):
        import torch
        from export_model import DeployWrapper        # pads to a multiple of 32, crops back
        self.torch, self.wrap, self.model = torch, DeployWrapper, model.cuda().eval()
        self.iters, self.scale_iters = iters, scale_iters
        self.wrappers = {}

    def __call__(self, left, right):
        h, w = left.shape[:2]
        if (h, w) not in self.wrappers:
            self.wrappers[(h, w)] = self.wrap(self.model, self.iters, self.scale_iters, h, w).eval()
        t = lambda x: self.torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float().cuda()
        with self.torch.no_grad():
            d = self.wrappers[(h, w)](t(left), t(right))
        d = d[-1] if isinstance(d, (list, tuple)) else d
        return d[0, 0].float().cpu().numpy()             # .cpu() waits for the GPU: the time is complete


def source_checkpoint(meta, folder, repo):
    """The checkpoint an export came from: the path it recorded, else the checkpoint folder named like
    its output folder (5_output_models/1_slim_model <-> 3_checkpoints/1_slim_model), else the recorded run."""
    ckpts = Path(repo) / "distillation" / "3_checkpoints"
    name = meta.get("source_checkpoint", "student_best.pth")
    tried = [Path(meta.get("source_path") or "/nonexistent"), ckpts / Path(folder).name / name,
             ckpts / meta.get("source_run", "") / name]
    for path in tried:
        if path.is_file():
            return path
    raise SystemExit("checkpoint for %s not found; tried %s" % (meta["name"], ", ".join(map(str, tried))))


class PretrainedModel(TorchModel):
    """An original DEFOM-Stereo release (defomstereo_<vits|vitb|vitl|vitg>_<data>.pth), as released:
    DEFOM's architecture with the depth encoder named in the file."""

    def __init__(self, path, iters, scale_iters):
        import torch
        from export_model import build_original
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        model, _, _ = build_original(Path(path), state.get("model", state))
        self.setup(model, iters, scale_iters)


class OnnxModel:
    """The exported .onnx file itself (fixed size)."""

    def __init__(self, meta, folder):
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(folder / meta["onnx"]), providers=providers)
        rospy.loginfo("%s: onnxruntime on %s", meta["name"], providers[0])

    def __call__(self, left, right):
        t = lambda x: np.ascontiguousarray(x.transpose(2, 0, 1))[None].astype(np.float32)
        return self.session.run(["disparity"], {"left": t(left), "right": t(right)})[0][0, 0]


def find_models(models_dir, baselines_dir, iters):
    """Our exports (every <name>.json with its ONNX beside it), then the original DEFOM releases
    (defomstereo_<vits|vitb|vitl|vitg>_<data>.pth) in baselines_dir."""
    found = []
    jsons = [Path(d) / f for d, _, files in os.walk(models_dir, followlinks=True) for f in files if f.endswith(".json")]
    for js in sorted(jsons):                     # (os.walk: model folders may be symlinks)
        try:
            meta = json.loads(js.read_text())
        except ValueError:
            continue
        if meta.get("format") == "onnx" and "source_run" in meta and (js.parent / meta.get("onnx", "")).is_file():
            found.append({"name": meta["name"], "meta": meta, "folder": js.parent,
                          "size": (meta["width"], meta["height"])})
    for c in sorted(Path(baselines_dir).glob("defomstereo_*.pth")) if baselines_dir else []:
        found.append({"name": "BASE_%s_it%ds%d" % (c.stem.replace("defomstereo_", ""), *iters),
                      "pretrained": c, "size": None})          # any size: runs at --long_side
    return found


# ---------------------------------------------------------------- input
def jpeg(data, reduce=1):
    # (OpenCV 4.2's imdecode ignores the IMREAD_REDUCED_* flags: decode, then shrink.)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode the image data")
    if reduce > 1:
        img = cv2.resize(img, (img.shape[1] // reduce, img.shape[0] // reduce), interpolation=cv2.INTER_AREA)
    return img


def split(stereo):
    half = stereo.shape[1] // 2
    return stereo[:, :half], stereo[:, half:]


def raw_image(msg):
    """sensor_msgs/Image -> BGR."""
    ch = {"bgr8": 3, "rgb8": 3, "mono8": 1, "bgra8": 4, "rgba8": 4}.get(msg.encoding)
    if ch is None:
        raise ValueError("unsupported image encoding %s" % msg.encoding)
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    conv = {"rgb8": cv2.COLOR_RGB2BGR, "mono8": cv2.COLOR_GRAY2BGR, "bgra8": cv2.COLOR_BGRA2BGR,
            "rgba8": cv2.COLOR_RGBA2BGR}.get(msg.encoding)
    return cv2.cvtColor(a, conv) if conv is not None else a


class AnyTopic:
    """Subscribes to any topic and decodes it with the definition the publisher
    (or rosbag) sends, so a different bx_msgs version in this workspace does not
    matter (ROS would otherwise silently refuse to connect)."""

    def __init__(self, topic, callback):
        self.topic, self.callback, self.cls, self.type = topic, callback, None, None
        rospy.Subscriber(topic, rospy.AnyMsg, self._receive, queue_size=2, buff_size=2 ** 27)

    def _receive(self, raw):
        if self.cls is None:
            header = raw._connection_header
            self.type = header["type"]
            self.cls = genpy.dynamic.generate_dynamic(self.type, header["message_definition"])[self.type]
            rospy.loginfo("%s: %s", self.topic, self.type)
        msg = self.cls()
        msg.deserialize(raw._buff)
        self.callback(msg)


def depth_image(data):
    """Encoded depth -> metres (NaN = invalid). Float EXR in metres (DWE, Voyis RGBD)
    or 16-bit PNG in millimetres."""
    d = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d[..., 0] if d.ndim == 3 else d
    m = d.astype(np.float32) / 1000.0 if d.dtype == np.uint16 else d.astype(np.float32)
    m[~(m > 0.01)] = np.nan                           # Voyis: 0 or less = invalid
    return m


def msg_time(msg):
    """Capture/publish time of a message, if it carries one."""
    if hasattr(msg, "unix_time_ms") and msg.unix_time_ms > 0:
        return rospy.Time.from_sec(msg.unix_time_ms / 1000.0)
    if hasattr(msg, "header") and not msg.header.stamp.is_zero():
        return msg.header.stamp
    return None


def decode(msg, reduce):
    """Any supported message -> left, right (BGR), reference depth (metres or None), stamp, frame_id, sensor_type."""
    ref, frame_id, stamp, sensor = None, "", msg_time(msg), None
    if hasattr(msg, "depth_data") and not hasattr(msg, "right_rgb_data"):
        raise ValueError("this topic has no right image (RGBD only): use it as --ref, and a stereo topic "
                         "(e.g. /ikan/voyis/stereo_concatenated/compressed) as --input")
    if hasattr(msg, "depth_data"):                      # bx_msgs/DepthImage (DWE 116, Voyis 115, ...)
        sensor = msg.sensor_type
        stereo = jpeg(msg.rgb_data, reduce)
        right = jpeg(msg.right_rgb_data, reduce) if len(msg.right_rgb_data) else None
        if right is not None and stereo.shape[1] == 2 * right.shape[1]:
            left = stereo[:, :right.shape[1]]               # rgb_data is left | right (DWE)
        elif right is None:
            left, right = split(stereo)
        else:
            left = stereo                                   # rgb_data is the left image
        if len(msg.depth_data):
            ref = depth_image(msg.depth_data)
    elif hasattr(msg, "left_data"):                     # bx_msgs/StereoImage
        left, right = jpeg(msg.left_data, reduce), jpeg(msg.right_data, reduce)
    elif hasattr(msg, "format"):                        # sensor_msgs/CompressedImage
        left, right = split(jpeg(msg.data, reduce))
        frame_id = msg.header.frame_id
    elif hasattr(msg, "encoding"):                      # sensor_msgs/Image
        left, right = split(raw_image(msg))
        if reduce > 1:
            left, right = (cv2.resize(x, None, fx=1.0 / reduce, fy=1.0 / reduce, interpolation=cv2.INTER_AREA)
                           for x in (left, right))
        frame_id = msg.header.frame_id
    else:
        raise ValueError("unsupported message %s" % type(msg).__name__)
    if right is None or right.shape != left.shape:
        raise ValueError("left and right images differ in size")
    if stamp is None:
        stamp = rospy.Time.now()
    return np.ascontiguousarray(left), np.ascontiguousarray(right), ref, stamp, frame_id, sensor


# ---------------------------------------------------------------- output helpers
def colour(depth, lo, hi):
    """Near = red, far = blue (like DWE's decoder); invalid = black."""
    valid = np.isfinite(depth) & (depth > 0)
    u8 = np.clip(255.0 * (1.0 - (np.where(valid, depth, hi) - lo) / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    out[~valid] = 0
    return out


def texture_mask(bgr, threshold):
    """True where the image has texture: mean gradient in a local window, worked out at 512 px long side
    (so the threshold means the same for DWE and Voyis), small specks (particles in the water) removed."""
    H, W = bgr.shape[:2]
    s = 512.0 / max(H, W)
    g = cv2.cvtColor(cv2.resize(bgr, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1)
    grad = cv2.blur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (11, 11))
    keep = cv2.morphologyEx((grad > threshold).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return cv2.resize(keep, (W, H), interpolation=cv2.INTER_NEAREST) > 0


def range_text(r):
    lo, hi = r
    return "" if not (lo > 0 or hi > 0) else "%s..%s m" % ("%.2g" % lo if lo > 0 else "0", "%.2g" % hi if hi > 0 else "inf")


def label(img, text, y=34):
    # Shrink the font until the text fits the panel (square Voyis panels are narrower).
    scale = 0.8
    while scale > 0.4 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > img.shape[1] - 20:
        scale -= 0.05
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def image_msg(header, arr, encoding):
    m = Image(header=header, height=arr.shape[0], width=arr.shape[1], encoding=encoding, is_bigendian=0)
    m.step = arr.strides[0]
    m.data = np.ascontiguousarray(arr).tobytes()
    return m


def cloud_msg(header, depth, bgr, K, stride, lo, hi):
    fx, fy, cx, cy = K
    z = depth[::stride, ::stride]
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride].astype(np.float32)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(z) & (z > 0)
        if lo > 0:
            ok &= z >= lo
        if hi > 0:
            ok &= z <= hi
    pts = np.empty(int(ok.sum()), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    zz = z[ok]
    pts["x"], pts["y"], pts["z"] = (u[ok] - cx) / fx * zz, (v[ok] - cy) / fy * zz, zz
    c = bgr[::stride, ::stride][ok].astype(np.uint32)
    pts["rgb"] = ((c[:, 2] << 16) | (c[:, 1] << 8) | c[:, 0]).view(np.float32)
    m = PointCloud2(header=header, height=1, width=len(pts), is_bigendian=False, is_dense=True,
                    point_step=16, row_step=16 * len(pts), data=pts.tobytes())
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate(("x", "y", "z", "rgb"))]
    return m


# ---------------------------------------------------------------- node
class Compare:
    """DWE / Voyis input. The engine (models, controls, inference, outputs, loop) is shared with
    compare_public_bags.py, which only replaces the input."""

    NODE = "defom_compare"

    def __init__(self, args):
        self.setup(args)
        # Camera from --camera, else DepthImage.sensor_type (per frame), else the topic name.
        self.set_camera(args.camera if args.camera != "auto" else ("voyis" if "voyis" in args.input.lower() else "dwe"))
        # Reference depth from a separate topic (Voyis RGBD: /ikan/voyis/stereo/data), matched by time.
        self.ref_topic = args.ref
        if self.ref_topic == "auto":
            self.ref_topic = "/ikan/voyis/stereo/data" if self.camera == "voyis" else ""
        if self.ref_topic.lower() == "none" or self.ref_topic == args.input:
            self.ref_topic = ""
        self.refs = collections.deque(maxlen=20)        # (time, depth in metres)
        if "raw" in args.input:
            rospy.logwarn("'%s' looks unrectified: stereo needs rectified pairs", args.input)
        self.advertise(("ref/depth/color", Image), ("ref/points", PointCloud2))
        AnyTopic(args.input, self.submit)
        if self.ref_topic:
            AnyTopic(self.ref_topic, self.on_ref)
        rospy.loginfo("waiting for %s%s (rosbag playing?)", args.input,
                      " and reference depth on %s" % self.ref_topic if self.ref_topic else "")

    # ---- shared engine
    def setup(self, args):
        """Models, live controls (~select, ~range, ~texture) and the newest-frame hand-off."""
        self.args = args
        self.models = find_models(args.models_dir, args.baselines_dir, args.base_iters)
        if not self.models:
            raise SystemExit("no models: no exports (<name>.json + .onnx) under %s, no defomstereo_*.pth in %s"
                             % (args.models_dir, args.baselines_dir))
        sys.path[:0] = [args.repo, args.repo + "/core", args.repo + "/distillation"]
        self.loaded, self.active, self.current = collections.OrderedDict(), 0, None   # current: this frame's model
        self.active = self.pick(args.model) if args.model else 0
        self.range, self.texture = tuple(args.range), args.texture
        self.lock = threading.Lock()
        self.latest, self.seq, self.done_seq, self.skipped, self.stats = None, 0, 0, 0, []
        rospy.loginfo("models (%s backend):", args.backend)
        for i, m in enumerate(self.models):
            rospy.loginfo("  %d: %-42s %s%s", i + 1, m["name"], "baseline (original DEFOM)" if m.get("pretrained")
                          else "ours (exported)", "   <- active" if i == self.active else "")
        for topic, callback in (("select", self.on_select), ("range", self.on_range), ("texture", self.on_texture)):
            rospy.Subscriber("~" + topic, String, callback, queue_size=1)

    def advertise(self, *extra):
        topics = (("compare", Image), ("depth", Image), ("depth/color", Image), ("points", PointCloud2),
                  ("camera_info", CameraInfo)) + extra
        self.pub = {name: rospy.Publisher("~" + name, typ, queue_size=1) for name, typ in topics}

    def wanted(self, name):
        return self.pub[name].get_num_connections() > 0

    def pick(self, key):
        key = str(key).strip()
        if key.isdigit() and 1 <= int(key) <= len(self.models):
            return int(key) - 1
        for i, m in enumerate(self.models):
            if key.lower() in m["name"].lower():
                return i
        rospy.logwarn("no model matches '%s'", key)
        return self.active

    def on_select(self, msg):
        self.select(self.pick(msg.data))

    def on_range(self, msg):
        """"MIN MAX" in metres (0 = no limit), or "off"."""
        text = msg.data.replace(",", " ").split()
        try:
            value = (0.0, 0.0) if text == ["off"] else tuple(float(v) for v in text)
            assert len(value) == 2
        except (ValueError, AssertionError):
            rospy.logwarn("range: expected 'MIN MAX' in metres or 'off', got '%s'", msg.data)
            return
        self.range = value
        rospy.loginfo("model depth range: %s", range_text(self.range) or "no limit")

    def on_texture(self, msg):
        try:
            self.texture = 0.0 if msg.data.strip() == "off" else float(msg.data)
        except ValueError:
            rospy.logwarn("texture: expected a number (0 = off) or 'off', got '%s'", msg.data)
            return
        rospy.loginfo("texture filter: %s", self.texture or "off")

    def select(self, index):
        self.active = index % len(self.models)
        rospy.loginfo("active model: %d %s", self.active + 1, self.models[self.active]["name"])

    def model(self, index):
        """The loaded model (at most --max_loaded stay on the GPU, least recently used unloaded first).
        A model that cannot be loaded raises ValueError: the frame is skipped, the node keeps running."""
        if index in self.loaded:
            self.loaded.move_to_end(index)
            return self.loaded[index]
        m = self.models[index]
        while len(self.loaded) >= max(1, self.args.max_loaded):        # free GPU memory (ViT-L is 1.5 GB)
            old, _ = self.loaded.popitem(last=False)
            rospy.loginfo("unloading %s", self.models[old]["name"])
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        rospy.loginfo("loading %s ...", m["name"])
        try:
            if m.get("pretrained"):
                net = PretrainedModel(m["pretrained"], *self.args.base_iters)
            elif self.args.backend == "onnx":
                net = OnnxModel(m["meta"], m["folder"])
            else:
                net = TorchModel(m["meta"], m["folder"], self.args.repo)
        except (Exception, SystemExit) as e:
            raise ValueError("cannot load %s: %s" % (m["name"], e))
        self.loaded[index] = net
        return net

    def submit(self, item):
        """Hand the newest frame to the loop; one still waiting is dropped (counted as skipped)."""
        with self.lock:
            if self.latest is not None and self.seq > self.done_seq:
                self.skipped += 1
            self.latest = item
            self.seq += 1

    def infer_size(self, m, W, H, fixed):
        """(crop box in the frame, model input size). Torch runs at the frame's aspect, at --long_side or the
        export's own long side if smaller (e.g. a 512x384 export); ONNX at its export size."""
        if m["size"] is None:                     # a baseline: any size
            s = self.args.long_side / max(W, H)
            return (0, 0, W, H), (max(32, int(round(W * s))), max(32, int(round(H * s))))
        ew, eh = m["size"]
        if not fixed:
            s = min(self.args.long_side, max(ew, eh)) / max(W, H)
            return (0, 0, W, H), (max(32, int(round(W * s))), max(32, int(round(H * s))))
        if abs(W / H - ew / eh) < 0.01:
            return (0, 0, W, H), (ew, eh)
        if W / H > ew / eh:                       # centre-crop to the export aspect
            cw = int(round(H * ew / eh))
            return ((W - cw) // 2, 0, cw, H), (ew, eh)
        ch = int(round(W * eh / ew))
        return (0, (H - ch) // 2, W, ch), (ew, eh)

    def estimate(self, left, right, fx, baseline):
        """The active model on a rectified pair -> (depth in metres at the frame size, NaN = invalid; ms;
        model input size). --range and --texture are applied here, so every output sees them."""
        H, W = left.shape[:2]
        index = self.active                       # read once: ~select may change it meanwhile
        m = self.models[index]
        try:
            net = self.model(index)
        except ValueError:
            if self.current is not None:          # back to the last model that worked
                self.active = self.current
            raise
        self.current = index
        (x0, y0, cw, ch), (w, h) = self.infer_size(m, W, H, isinstance(net, OnnxModel))
        prep = lambda im: cv2.cvtColor(cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
                                       if (w, h) != (cw, ch) else im[y0:y0 + ch, x0:x0 + cw], cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        disp_small = net(prep(left), prep(right))
        ms = 1000 * (time.perf_counter() - t0)
        disp = np.full((H, W), np.nan, np.float32)
        disp[y0:y0 + ch, x0:x0 + cw] = (cv2.resize(disp_small, (cw, ch), interpolation=cv2.INTER_LINEAR) * (cw / w)
                                        if (w, h) != (cw, ch) else disp_small)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disp > 0.5, fx * baseline / disp, np.nan).astype(np.float32)
            lo, hi = self.range                           # outside = invalid
            if lo > 0:
                depth[depth < lo] = np.nan
            if hi > 0:
                depth[depth > hi] = np.nan
        if self.texture > 0:                              # no texture = invalid
            depth[~texture_mask(left, self.texture)] = np.nan
        return depth, ms, (w, h)

    def publish(self, header, left, depth, K):
        """~depth, ~depth/color, ~points (only with subscribers), ~camera_info. Returns the coloured depth."""
        a = self.args
        mc = colour(depth, a.vis_min, a.vis_max) if (a.show or self.wanted("compare") or self.wanted("depth/color")) else None
        if self.wanted("depth"):
            self.pub["depth"].publish(image_msg(header, depth, "32FC1"))
        if mc is not None and self.wanted("depth/color"):
            self.pub["depth/color"].publish(image_msg(header, mc, "bgr8"))
        if self.wanted("points"):
            self.pub["points"].publish(cloud_msg(header, depth, left, K, a.stride, a.min_depth, a.max_depth))
        H, W = depth.shape
        fx, fy, cx, cy = K
        info = CameraInfo(header=header, width=W, height=H, distortion_model="plumb_bob")
        info.D = [0.0] * 5
        info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        self.pub["camera_info"].publish(info)
        return mc

    def compose(self, header, tiles, mc, ms, note, size):
        """[left, ...] + the model's depth, 600 px high, published on ~compare; None if nobody looks."""
        a = self.args
        if mc is None or not (a.show or self.wanted("compare")):
            return None
        dh = 600
        fit = lambda im: cv2.resize(im, (int(round(im.shape[1] * dh / im.shape[0])), dh), interpolation=cv2.INTER_AREA)
        model_tile = label(label(fit(mc), "%d: %s @%dx%d" % (self.current + 1, self.models[self.current]["name"], *size)),
                           "%.0f ms  %s" % (ms, note), 68)
        panel = np.hstack([label(fit(img), text) for img, text in tiles] + [model_tile])
        label(panel, "colour: %.1f m (red) .. %.1f m (blue)%s%s" % (
            a.vis_min, a.vis_max, "   model depth %s" % range_text(self.range) if any(self.range) else "",
            "   texture > %g" % self.texture if self.texture > 0 else ""), dh - 14)
        if self.wanted("compare"):
            self.pub["compare"].publish(image_msg(header, panel, "bgr8"))
        return panel

    def spin(self):
        t_report, panel = time.time(), None
        if self.args.show:
            cv2.namedWindow(self.NODE, cv2.WINDOW_NORMAL)
        while not rospy.is_shutdown():
            with self.lock:
                item, seq = (self.latest, self.seq) if self.seq > self.done_seq else (None, self.seq)
                self.done_seq = seq
            if item is not None:
                try:
                    panel = self.process(item)
                except ValueError as e:
                    rospy.logwarn_throttle(5, "skipping frame: %s", e)
            if self.args.show:
                if panel is not None:
                    cv2.imshow(self.NODE, panel)
                key = cv2.waitKey(1 if item is not None else 20) & 0xFF
                if key in (ord("q"), 27):
                    break
                if ord("1") <= key <= ord("9") and key - ord("1") < len(self.models):
                    self.select(key - ord("1"))
                elif key == ord("n"):
                    self.select(self.active + 1)
                elif key == ord("p"):
                    self.select(self.active - 1)
            elif item is None:
                time.sleep(0.01)
            if time.time() - t_report >= self.args.report_s:
                if self.stats:
                    ms = [s[0] for s in self.stats]
                    rospy.loginfo("%s @%dx%d | %.1f frames/s | %.0f ms per frame | skipped %d | %s",
                                  self.models[self.current]["name"], self.stats[-1][2][0], self.stats[-1][2][1],
                                  len(ms) / (time.time() - t_report), np.median(ms), self.skipped, self.stats[-1][1])
                self.stats, t_report = [], time.time()
        cv2.destroyAllWindows()

    # ---- DWE / Voyis input
    def set_camera(self, name):
        self.camera = name
        self.cam = dict(CAMERAS[name])
        for k in ("fx", "fy", "cx", "cy", "baseline"):
            if getattr(self.args, k) is not None:
                self.cam[k] = getattr(self.args, k)
        # Decode big JPEGs at reduced size when the model runs smaller anyway (Voyis 2816 -> 1408).
        self.reduce = 1
        while self.reduce < 8 and self.cam["width"] / (2 * self.reduce) >= self.args.long_side * 1.25:
            self.reduce *= 2
        rospy.loginfo("camera %s: fx %.3f at %d wide, baseline %.4f m%s", name, self.cam["fx"], self.cam["width"],
                      self.cam["baseline"], ", decoding at 1/%d size" % self.reduce if self.reduce > 1 else "")

    def on_ref(self, msg):
        if not hasattr(msg, "depth_data"):
            rospy.logwarn_throttle(10, "%s has no depth_data: no reference depth", self.ref_topic)
            return
        d = depth_image(msg.depth_data)
        if d is not None:
            with self.lock:
                self.refs.append((msg_time(msg) or rospy.Time.now(), d))

    def matching_ref(self, stamp):
        """The reference depth closest in time to the stereo frame, if within --ref_max_dt."""
        with self.lock:
            best = min(self.refs, key=lambda r: abs((r[0] - stamp).to_sec()), default=None)
        if best is None or abs((best[0] - stamp).to_sec()) > self.args.ref_max_dt:
            return None, None
        return best[1], (best[0] - stamp).to_sec()

    def process(self, msg):
        a = self.args
        left, right, ref, stamp, frame_id, sensor = decode(msg, self.reduce)
        if a.camera == "auto" and SENSORS.get(sensor, self.camera) != self.camera:
            self.set_camera(SENSORS[sensor])               # DepthImage says which camera it is
            left, right, ref, stamp, frame_id, sensor = decode(msg, self.reduce)
        ref_name = REF_NAMES.get(self.camera, "reference")
        if ref is None and self.ref_topic:
            ref, ref_dt = self.matching_ref(stamp)
            if ref_dt is not None:
                ref_name += " (%+.2f s)" % ref_dt
        H, W = left.shape[:2]
        if ref is not None and ref.shape != (H, W):     # e.g. Voyis RGBD 1024x1024: the left camera scaled
            ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_NEAREST)
        s = W / self.cam["width"]
        K = (self.cam["fx"] * s, self.cam["fy"] * s, self.cam["cx"] * s, self.cam["cy"] * s)
        depth, ms, size = self.estimate(left, right, K[0], self.cam["baseline"])

        diff = ""
        if ref is not None:
            with np.errstate(invalid="ignore"):
                both = np.isfinite(depth) & np.isfinite(ref) & (ref > 0) & (ref < a.vis_max)
            if both.any():
                diff = "differs from %s by %.1f%% (median)" % (
                    self.camera.upper(), 100 * np.median(np.abs(depth[both] - ref[both]) / ref[both]))
        self.stats.append((ms, diff, size))

        header = Header(stamp=stamp, frame_id=a.frame_id or frame_id or self.cam["frame_id"])
        mc = self.publish(header, left, depth, K)
        shown = mc is not None and (a.show or self.wanted("compare"))
        rc = colour(ref, a.vis_min, a.vis_max) if ref is not None and (shown or self.wanted("ref/depth/color")) else None
        if rc is not None and self.wanted("ref/depth/color"):
            self.pub["ref/depth/color"].publish(image_msg(header, rc, "bgr8"))
        if ref is not None and self.wanted("ref/points"):
            self.pub["ref/points"].publish(cloud_msg(header, ref, left, K, a.stride, a.min_depth, a.max_depth))
        tiles = [(left, "left (%s)" % self.camera),
                 (rc if rc is not None else np.zeros_like(left),
                  ref_name if ref is not None else "no reference depth in this topic")] if shown else []
        return self.compose(header, tiles, mc, ms, diff, size)


def add_common_args(ap):
    """Arguments shared with compare_public_bags.py: models, display, filters, outputs."""
    ap.add_argument("--models_dir", default=str(Path(REPO) / "distillation/5_output_models"),
                    help="exported models: <model>/<name>.json + .onnx")
    ap.add_argument("--baselines_dir", default=str(Path(REPO) / "checkpoints"),
                    help="original DEFOM releases defomstereo_*.pth ('' = none)")
    ap.add_argument("--repo", default=REPO, help="DEFOM training repo (checkpoints, export_model.py)")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--long_side", type=int, default=800, help="torch: model input long side (at most the export's)")
    ap.add_argument("--model", default="", help="start with this model: index (1..) or part of the name")
    ap.add_argument("--base_iters", default="32/8", help="baselines' iterations/scale iterations (DEFOM's default)")
    ap.add_argument("--max_loaded", type=int, default=3, help="models kept on the GPU at once (least recent unloaded)")
    ap.add_argument("--show", action="store_true", help="OpenCV window; keys 1-9 / n / p switch model, q quits")
    ap.add_argument("--vis_min", type=float, default=0.0, help="colour scale, metres (red)")
    ap.add_argument("--vis_max", type=float, default=3.0, help="colour scale, metres (blue)")
    ap.add_argument("--range", type=float, nargs=2, default=[0.0, 0.0], metavar=("MIN", "MAX"),
                    help="model depth kept only in MIN..MAX metres, every model (0 = no limit); also ~range at run time")
    ap.add_argument("--texture", type=float, default=0.0,
                    help="drop model depth where the left image has no texture (open water), every model; "
                         "0 = off, ~6 for murky Voyis; also ~texture at run time")
    ap.add_argument("--min_depth", type=float, default=0.0, help="point clouds: metres, 0 = no limit")
    ap.add_argument("--max_depth", type=float, default=0.0, help="point clouds: metres, 0 = no limit")
    ap.add_argument("--stride", type=int, default=2, help="point clouds: every Nth pixel")
    ap.add_argument("--report_s", type=float, default=5.0)


def parse_args(ap):
    args = ap.parse_args(rospy.myargv(argv=sys.argv)[1:])
    try:
        args.base_iters = tuple(int(v) for v in args.base_iters.split("/"))
        assert len(args.base_iters) == 2
    except (ValueError, AssertionError):
        raise SystemExit("--base_iters must look like 32/8")
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="/ikan/voyis/stereo_concatenated/compressed",
                    help="DWE: /ikan/explore3d/depth_image; Voyis: /ikan/voyis/stereo_concatenated/compressed")
    ap.add_argument("--ref", default="auto",
                    help="topic with the camera's own depth when --input has none (auto: Voyis "
                         "/ikan/voyis/stereo/data; 'none' to disable). Matched to frames by time")
    ap.add_argument("--ref_max_dt", type=float, default=0.5, help="largest time gap to accept a reference depth, s")
    ap.add_argument("--camera", choices=["auto", "dwe", "voyis"], default="auto",
                    help="auto: from DepthImage.sensor_type, else the topic name")
    ap.add_argument("--frame_id", default="", help="default: the message's, else explore3d / voyis")
    for k in ("fx", "fy", "cx", "cy"):
        ap.add_argument("--" + k, type=float, default=None, help="override, at the camera's calibration size")
    ap.add_argument("--baseline", type=float, default=None, help="override, metres")
    add_common_args(ap)
    args = parse_args(ap)
    rospy.init_node(Compare.NODE)
    Compare(args).spin()


if __name__ == "__main__":
    main()
