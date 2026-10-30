// defom_svc: DEFOM stereo depth on the SVC (Orin NX), streamed to the AIB over UDP.
//
// Captures both DWE cameras (raw 1600x1200 MJPEG), rectifies them with the
// .dwecal maps (or not, if the raw stream turns out to be rectified already),
// runs a DEFOM TensorRT engine at its own size (slim v9 800x600, ViT-S s2 512x384;
// any 4:3 engine), upsamples the disparity to 1600x1200 and sends float32 depth
// + intrinsics + the rectified left image (JPEG) to the AIB. No ROS, nothing to
// install. See distillation/README.md, section 6.
//
// Build on the SVC (JetPack 6), one line:
//   g++ -O3 -std=c++17 -o defom_svc defom_svc.cpp -I/usr/include/opencv4 -I/usr/local/cuda/include
//       -L/usr/local/cuda/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d
//       -lnvinfer -lnvinfer_plugin -lcudart -lpthread

#include <arpa/inet.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// ---------------------------------------------------------------- wire format
// Every frame is one blob: FrameHeader | depth (float32, metres, NaN = invalid)
// | left image (JPEG). The blob is cut into UDP datagrams, each starting with a
// PacketHeader. All fields little-endian. defom_receiver.py mirrors these.
#pragma pack(push, 1)
struct PacketHeader {                    // 32 bytes
  uint32_t magic = 0x504D4644;           // "DFMP"
  uint16_t version = 1;
  uint16_t header_bytes = 32;
  uint32_t frame_id = 0;
  uint32_t chunk_index = 0;
  uint32_t chunk_count = 0;
  uint32_t offset = 0;                   // of this chunk in the blob
  uint32_t frame_bytes = 0;              // whole blob
  uint32_t reserved = 0;
};
struct FrameHeader {                     // 128 bytes
  uint32_t magic = 0x464D4644;           // "DFMF"
  uint16_t version = 1;
  uint16_t header_bytes = 128;
  uint32_t frame_id = 0;
  uint32_t width = 0, height = 0;        // depth and left image (pixel-aligned)
  float fx = 0, fy = 0, cx = 0, cy = 0;  // rectified, at width x height
  float baseline_m = 0;
  uint32_t depth_format = 1;             // 1 = float32 metres, row-major, NaN = invalid
  uint32_t depth_offset = 0, depth_bytes = 0;
  uint32_t image_format = 1;             // 1 = JPEG, BGR, rectified left
  uint32_t image_offset = 0, image_bytes = 0;
  uint32_t rectified = 0;                // 1 = .dwecal maps applied, 0 = raw used as is
  uint32_t infer_width = 0, infer_height = 0;
  uint32_t age_us = 0;                   // left capture -> first datagram sent (SVC clock)
  int32_t lr_skew_us = 0;                // right capture - left capture
  float prep_ms = 0, infer_ms = 0, post_ms = 0;
  uint64_t capture_mono_ns = 0;          // SVC CLOCK_MONOTONIC
  uint64_t sent_wall_ns = 0;             // SVC CLOCK_REALTIME (valid only if its clock is set)
  uint8_t pad[16] = {};
};
#pragma pack(pop)
static_assert(sizeof(PacketHeader) == 32, "PacketHeader layout");
static_assert(sizeof(FrameHeader) == 128, "FrameHeader layout");

// ---------------------------------------------------------------- utilities
static std::atomic<bool> g_stop{false};
static const auto g_t0 = std::chrono::steady_clock::now();

static void info(const char* fmt, ...) {
  const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count();
  std::fprintf(stdout, "[%8.1f] ", t);
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stdout, fmt, ap);
  va_end(ap);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

[[noreturn]] static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "ERROR: ");
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
  std::exit(1);
}

static int64_t mono_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

static int64_t wall_ns() {
  timespec ts{};
  clock_gettime(CLOCK_REALTIME, &ts);
  return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

static void sleep_ms(double ms) {
  if (ms > 0) std::this_thread::sleep_for(std::chrono::microseconds(int64_t(ms * 1000)));
}

static void cuda_ok(cudaError_t e, const char* what) {
  if (e != cudaSuccess) die("%s: %s", what, cudaGetErrorString(e));
}

// ---------------------------------------------------------------- options
struct Options {
  std::string engine, dwecal;
  std::string rectify = "auto";          // auto | on | off
  std::string left_dev = "/dev/video0", right_dev = "/dev/video2";
  int cam_fps = 60;
  double max_skew_ms = 12.0;
  bool swap_lr = false;
  std::string input_dir;                 // test mode: side-by-side images instead of cameras
  std::string aib = "192.168.4.1";
  int port = 5610;
  double send_mbps = 500.0;
  int packet_bytes = 8192;
  double baseline_m = 0.1;
  double max_hz = 0.0;
  double edge_px = 1.5;
  int jpeg_quality = 90;
  double report_s = 5.0;
  bool check_only = false;
  int check_pairs = 10;
  std::string save_dir;                  // save the first frames' inputs/outputs for checking
  int save_frames = 3;
};

static void usage() {
  std::printf(
      "defom_svc --engine MODEL_fp16b.engine --dwecal e3d1008.dwecal [options]\n\n"
      "  --rectify auto|on|off   auto (default): measure at start-up whether the raw stream is\n"
      "                          already rectified; rectify with the .dwecal maps only if not\n"
      "  --check_rectification   only run that measurement, print the verdict and exit\n"
      "  --check_pairs N         pairs used for the measurement (10)\n"
      "  --left/--right DEV      cameras (/dev/video0, /dev/video2)   --swap_lr\n"
      "  --cam_fps N             camera rate (60); pairs are matched by capture time\n"
      "  --max_skew_ms X         largest left/right capture-time gap accepted (12)\n"
      "  --input_dir DIR         test without cameras: side-by-side raw images (3200x1200)\n"
      "  --aib IP --port N       receiver (192.168.4.1, 5610)\n"
      "  --send_mbps X           UDP pacing (500)      --packet_bytes N  datagram size (8192)\n"
      "  --baseline_m X          stereo baseline (0.1)\n"
      "  --max_hz X              cap the output rate (0 = as fast as possible)\n"
      "  --edge_px X             depth-edge threshold for upsampling, model pixels (1.5)\n"
      "  --jpeg_quality N        left image (90)       --report_s X  status line period (5)\n"
      "  --save_dir DIR          save the first --save_frames frames (left, disparity, depth), and with\n"
      "                          --check_rectification the raw/rectified pairs, for offline checks\n");
}

static Options parse_args(int argc, char** argv) {
  Options o;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) die("%s needs a value", argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--engine") o.engine = need(i);
    else if (a == "--dwecal") o.dwecal = need(i);
    else if (a == "--rectify") o.rectify = need(i);
    else if (a == "--check_rectification") o.check_only = true;
    else if (a == "--check_pairs") o.check_pairs = std::stoi(need(i));
    else if (a == "--left") o.left_dev = need(i);
    else if (a == "--right") o.right_dev = need(i);
    else if (a == "--swap_lr") o.swap_lr = true;
    else if (a == "--cam_fps") o.cam_fps = std::stoi(need(i));
    else if (a == "--max_skew_ms") o.max_skew_ms = std::stod(need(i));
    else if (a == "--input_dir") o.input_dir = need(i);
    else if (a == "--aib") o.aib = need(i);
    else if (a == "--port") o.port = std::stoi(need(i));
    else if (a == "--send_mbps") o.send_mbps = std::stod(need(i));
    else if (a == "--packet_bytes") o.packet_bytes = std::stoi(need(i));
    else if (a == "--baseline_m") o.baseline_m = std::stod(need(i));
    else if (a == "--max_hz") o.max_hz = std::stod(need(i));
    else if (a == "--edge_px") o.edge_px = std::stod(need(i));
    else if (a == "--jpeg_quality") o.jpeg_quality = std::stoi(need(i));
    else if (a == "--report_s") o.report_s = std::stod(need(i));
    else if (a == "--save_dir") o.save_dir = need(i);
    else if (a == "--save_frames") o.save_frames = std::stoi(need(i));
    else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
    else die("unknown option %s (--help)", a.c_str());
  }
  if (o.dwecal.empty()) die("--dwecal is required (the rectified intrinsics come from it)");
  if (o.engine.empty() && !o.check_only) die("--engine is required");
  if (o.rectify != "auto" && o.rectify != "on" && o.rectify != "off") die("--rectify auto|on|off");
  if (o.packet_bytes < 512 || o.packet_bytes > 65000) die("--packet_bytes 512..65000");
  return o;
}

// ---------------------------------------------------------------- calibration
// DWECal v1: "DWECAL", version (3 B), n_cameras (1 B), resolution code (4 B),
// fx fy cx cy (float32), then the left and right maps: H x W x (x, y) float32,
// the raw pixel each rectified pixel samples (cv::remap's format).
struct Calibration {
  int width = 0, height = 0;
  float fx = 0, fy = 0, cx = 0, cy = 0;
  cv::Mat map1[2], map2[2];              // fixed-point maps for fast remap, [left, right]
};

static Calibration load_dwecal(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot read %s", path.c_str());
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (blob.size() < 30 || std::memcmp(blob.data(), "DWECAL", 6) != 0) die("%s is not a DWECal file", path.c_str());
  Calibration c;
  float k[4];
  std::memcpy(k, blob.data() + 14, sizeof(k));
  c.fx = k[0], c.fy = k[1], c.cx = k[2], c.cy = k[3];
  const size_t floats = (blob.size() - 30) / 4;
  const int sizes[][2] = {{1600, 1200}, {800, 600}, {1280, 720}, {640, 480}};
  for (const auto& s : sizes)
    if (floats == size_t(2) * s[0] * s[1] * 2) c.width = s[0], c.height = s[1];
  if (!c.width) die("unrecognised map size in %s", path.c_str());
  for (int eye = 0; eye < 2; ++eye) {
    // Copy out of the file buffer: the maps start at byte 30, not float-aligned.
    cv::Mat xy(c.height, c.width, CV_32FC2);
    std::memcpy(xy.data, blob.data() + 30 + size_t(eye) * c.width * c.height * 2 * sizeof(float), xy.total() * 8);
    // How far the map moves pixels (0 everywhere = identity = nothing to rectify).
    std::vector<float> shift;
    for (int y = 0; y < c.height; y += 8)
      for (int x = 0; x < c.width; x += 8) {
        const cv::Vec2f m = xy.at<cv::Vec2f>(y, x);
        shift.push_back(std::hypot(m[0] - x, m[1] - y));
      }
    std::nth_element(shift.begin(), shift.begin() + shift.size() / 2, shift.end());
    const float median = shift[shift.size() / 2];
    info("  %s map: moves pixels by a median %.1f px (max %.1f)", eye ? "right" : "left", median,
         *std::max_element(shift.begin(), shift.end()));
    cv::convertMaps(xy, cv::Mat(), c.map1[eye], c.map2[eye], CV_16SC2);
  }
  return c;
}

// ---------------------------------------------------------------- capture
struct RawPair {
  cv::Mat left, right;                   // BGR, camera resolution
  int64_t t_left = 0, t_right = 0;       // capture time, CLOCK_MONOTONIC ns
  uint32_t seq = 0;
};

class PairSource {
 public:
  virtual ~PairSource() = default;
  // The freshest pair not handed out before. False on stop or failure.
  virtual bool next(RawPair& out) = 0;
};

// One UVC camera via V4L2 (mmap, MJPEG). A thread keeps the last few
// compressed frames with their kernel timestamps; nothing is decoded here.
class V4l2Camera {
 public:
  struct Frame {
    std::vector<uint8_t> jpeg;
    int64_t t_ns = 0;
    uint32_t seq = 0;
  };

  V4l2Camera(const std::string& dev, int w, int h, int fps) : dev_(dev) {
    fd_ = ::open(dev.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) die("cannot open %s: %s", dev.c_str(), std::strerror(errno));
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = w;
    fmt.fmt.pix.height = h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(VIDIOC_S_FMT, &fmt) < 0)
      die("%s: cannot set MJPEG %dx%d (%s) -- is DWE's stream or another program using it?",
          dev.c_str(), w, h, std::strerror(errno));
    if (int(fmt.fmt.pix.width) != w || int(fmt.fmt.pix.height) != h || fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG)
      die("%s gives %ux%u, not MJPEG %dx%d", dev.c_str(), fmt.fmt.pix.width, fmt.fmt.pix.height, w, h);
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    xioctl(VIDIOC_S_PARM, &parm);        // best effort

    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(VIDIOC_REQBUFS, &req) < 0 || req.count < 2) die("%s: REQBUFS failed", dev.c_str());
    for (unsigned i = 0; i < req.count; ++i) {
      v4l2_buffer b{};
      b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      b.memory = V4L2_MEMORY_MMAP;
      b.index = i;
      if (xioctl(VIDIOC_QUERYBUF, &b) < 0) die("%s: QUERYBUF failed", dev.c_str());
      void* p = mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, b.m.offset);
      if (p == MAP_FAILED) die("%s: mmap failed", dev.c_str());
      buffers_.push_back({p, b.length});
      if (xioctl(VIDIOC_QBUF, &b) < 0) die("%s: QBUF failed", dev.c_str());
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(VIDIOC_STREAMON, &type) < 0)
      die("%s: STREAMON failed (%s) -- is DWE's stream or another program using it?", dev.c_str(), std::strerror(errno));
    thread_ = std::thread([this] { loop(); });
  }

  ~V4l2Camera() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(VIDIOC_STREAMOFF, &type);
    for (auto& b : buffers_) munmap(b.first, b.second);
    if (fd_ >= 0) ::close(fd_);
  }

  // The last few frames, oldest first.
  std::vector<Frame> recent() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<Frame>(ring_.begin(), ring_.end());
  }
  bool failed() const { return failed_; }
  const std::string& name() const { return dev_; }

 private:
  int xioctl(unsigned long req, void* arg) {
    int r;
    do r = ioctl(fd_, req, arg); while (r < 0 && errno == EINTR);
    return r;
  }

  void loop() {
    int64_t last_frame = mono_ns();
    while (running_ && !g_stop) {
      pollfd p{fd_, POLLIN, 0};
      const int r = poll(&p, 1, 200);
      if (r <= 0) {
        if (mono_ns() - last_frame > 3000000000LL) {
          info("%s: no frames for 3 s", dev_.c_str());
          last_frame = mono_ns();
        }
        continue;
      }
      v4l2_buffer b{};
      b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      b.memory = V4L2_MEMORY_MMAP;
      if (xioctl(VIDIOC_DQBUF, &b) < 0) {
        if (errno == EAGAIN) continue;
        info("%s: DQBUF failed (%s) -- camera disconnected?", dev_.c_str(), std::strerror(errno));
        failed_ = true;
        return;
      }
      last_frame = mono_ns();
      // UVC timestamps are CLOCK_MONOTONIC (start of frame); fall back to now.
      int64_t t = int64_t(b.timestamp.tv_sec) * 1000000000LL + int64_t(b.timestamp.tv_usec) * 1000LL;
      if ((b.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) != V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC || t == 0) t = last_frame;
      if (!(b.flags & V4L2_BUF_FLAG_ERROR) && b.bytesused > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        Frame f;
        if (ring_.size() >= 4) {          // reuse the oldest frame's memory
          f = std::move(ring_.front());
          ring_.pop_front();
        }
        const uint8_t* src = static_cast<const uint8_t*>(buffers_[b.index].first);
        f.jpeg.assign(src, src + b.bytesused);
        f.t_ns = t;
        f.seq = b.sequence;
        ring_.push_back(std::move(f));
      }
      xioctl(VIDIOC_QBUF, &b);
    }
  }

  std::string dev_;
  int fd_ = -1;
  std::vector<std::pair<void*, size_t>> buffers_;
  std::deque<Frame> ring_;
  std::mutex mutex_;
  std::thread thread_;
  std::atomic<bool> running_{true}, failed_{false};
};

class CameraPairs : public PairSource {
 public:
  CameraPairs(const Options& o, int w, int h)
      : left_(o.swap_lr ? o.right_dev : o.left_dev, w, h, o.cam_fps),
        right_(o.swap_lr ? o.left_dev : o.right_dev, w, h, o.cam_fps),
        max_skew_ns_(int64_t(o.max_skew_ms * 1e6)), width_(w), height_(h) {}

  bool next(RawPair& out) override {
    const int64_t deadline = mono_ns() + 2000000000LL;
    while (!g_stop && !left_.failed() && !right_.failed()) {
      const auto L = left_.recent(), R = right_.recent();
      // Newest left frame not used yet whose closest right frame is within max skew.
      for (auto l = L.rbegin(); l != L.rend(); ++l) {
        if (have_last_ && l->t_ns <= last_left_t_) break;
        const V4l2Camera::Frame* best = nullptr;
        for (const auto& r : R)
          if (!best || std::llabs(r.t_ns - l->t_ns) < std::llabs(best->t_ns - l->t_ns)) best = &r;
        if (!best || std::llabs(best->t_ns - l->t_ns) > max_skew_ns_) continue;
        // Decode both in parallel (~15 ms each at 1600x1200).
        auto fr = std::async(std::launch::async, [&] { return cv::imdecode(best->jpeg, cv::IMREAD_COLOR); });
        out.left = cv::imdecode(l->jpeg, cv::IMREAD_COLOR);
        out.right = fr.get();
        last_left_t_ = l->t_ns;
        have_last_ = true;
        if (out.left.empty() || out.right.empty() || out.left.cols != width_ || out.right.cols != width_ ||
            out.left.rows != height_ || out.right.rows != height_) {
          ++corrupt_;
          break;                         // corrupt JPEG: try the next frames
        }
        out.t_left = l->t_ns;
        out.t_right = best->t_ns;
        out.seq = l->seq;
        return true;
      }
      if (mono_ns() > deadline) {
        info("no left/right pair within %.0f ms of each other for 2 s (cameras running? --max_skew_ms)",
            max_skew_ns_ / 1e6);
        return false;
      }
      sleep_ms(2);
    }
    return false;
  }
  long corrupt() const { return corrupt_; }

 private:
  V4l2Camera left_, right_;
  int64_t max_skew_ns_;
  int width_, height_;
  int64_t last_left_t_ = 0;
  bool have_last_ = false;
  long corrupt_ = 0;
};

// Test input: side-by-side raw frames (left | right) from a folder, looped.
class FilePairs : public PairSource {
 public:
  FilePairs(const Options& o, int w, int h) : swap_(o.swap_lr), width_(w), height_(h) {
    std::vector<cv::String> files;
    for (const char* ext : {"*.png", "*.jpg", "*.jpeg"}) {
      std::vector<cv::String> f;
      cv::glob(o.input_dir + "/" + ext, f, false);
      files.insert(files.end(), f.begin(), f.end());
    }
    std::sort(files.begin(), files.end());
    for (const auto& f : files) files_.push_back(f);
    if (files_.empty()) die("no images in %s", o.input_dir.c_str());
    info("test input: %zu side-by-side images from %s (looped)", files_.size(), o.input_dir.c_str());
  }

  bool next(RawPair& out) override {
    for (size_t tries = 0; tries < files_.size() && !g_stop; ++tries) {
      const cv::Mat im = cv::imread(files_[index_++ % files_.size()], cv::IMREAD_COLOR);
      if (im.rows != height_ || im.cols != 2 * width_) continue;
      out.left = im(cv::Rect(swap_ ? width_ : 0, 0, width_, height_)).clone();
      out.right = im(cv::Rect(swap_ ? 0 : width_, 0, width_, height_)).clone();
      out.t_left = out.t_right = mono_ns();
      out.seq = uint32_t(index_);
      return true;
    }
    if (!g_stop) info("no %dx%d side-by-side images in the input folder", 2 * width_, height_);
    return false;
  }

 private:
  std::vector<std::string> files_;
  size_t index_ = 0;
  bool swap_;
  int width_, height_;
};

// ---------------------------------------------------------------- rectification check
// Matches features between the eyes and measures their vertical offset: zero
// everywhere (and no trend across the image) for a rectified pair.
struct RowAlignment {
  int matches = 0;
  double dy_median = 0, dy_p90 = 0, trend_x = 0, trend_y = 0, disparity_median = 0;
  bool enough() const { return matches >= 50; }
  bool rectified() const {
    return enough() && dy_median < 0.5 && std::fabs(trend_x) < 1.0 && std::fabs(trend_y) < 1.0;
  }
};

static void match_rows(const cv::Mat& left, const cv::Mat& right, std::vector<cv::Vec4d>& pts) {
  cv::Mat gl, gr, dl, dr;
  cv::cvtColor(left, gl, cv::COLOR_BGR2GRAY);
  cv::cvtColor(right, gr, cv::COLOR_BGR2GRAY);
  std::vector<cv::KeyPoint> kl, kr;
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 4)
  auto detector = cv::SIFT::create(4000);
  const int norm = cv::NORM_L2;
#else
  auto detector = cv::ORB::create(4000);
  const int norm = cv::NORM_HAMMING;
#endif
  detector->detectAndCompute(gl, cv::noArray(), kl, dl);
  detector->detectAndCompute(gr, cv::noArray(), kr, dr);
  if (kl.size() < 10 || kr.size() < 10) return;
  std::vector<std::vector<cv::DMatch>> knn;
  cv::BFMatcher(norm).knnMatch(dl, dr, knn, 2);
  for (const auto& m : knn) {
    if (m.size() < 2 || m[0].distance >= 0.7f * m[1].distance) continue;
    const cv::Point2f a = kl[m[0].queryIdx].pt, b = kr[m[0].trainIdx].pt;
    const double dx = a.x - b.x, dy = a.y - b.y;
    if (std::fabs(dy) < 40 && dx > -5) pts.push_back({a.x, a.y, dx, dy});
  }
}

static RowAlignment summarise(const std::vector<cv::Vec4d>& pts, int w, int h) {
  RowAlignment s;
  s.matches = int(pts.size());
  if (!s.enough()) return s;
  std::vector<double> ady, disp;
  cv::Mat A(s.matches, 3, CV_64F), y(s.matches, 1, CV_64F), coef;
  for (int i = 0; i < s.matches; ++i) {
    ady.push_back(std::fabs(pts[i][3]));
    disp.push_back(pts[i][2]);
    A.at<double>(i, 0) = 1.0;
    A.at<double>(i, 1) = pts[i][0] / w;
    A.at<double>(i, 2) = pts[i][1] / h;
    y.at<double>(i, 0) = pts[i][3];
  }
  auto pct = [](std::vector<double> v, double q) {
    std::nth_element(v.begin(), v.begin() + size_t(q * (v.size() - 1)), v.end());
    return v[size_t(q * (v.size() - 1))];
  };
  s.dy_median = pct(ady, 0.5);
  s.dy_p90 = pct(ady, 0.9);
  s.disparity_median = pct(disp, 0.5);
  cv::solve(A, y, coef, cv::DECOMP_SVD);   // dy = a + b*x/W + c*y/H
  s.trend_x = coef.at<double>(1);
  s.trend_y = coef.at<double>(2);
  return s;
}

static void print_alignment(const char* label, const RowAlignment& s) {
  if (!s.enough()) {
    info("  %-18s %4d matches -- not enough (point the cameras at a textured scene)", label, s.matches);
    return;
  }
  info("  %-18s %4d matches | vertical offset median %.2f px, p90 %.2f px | trend across width %+.2f px, "
      "height %+.2f px | disparity median %.1f px -> %s",
      label, s.matches, s.dy_median, s.dy_p90, s.trend_x, s.trend_y, s.disparity_median,
      s.rectified() ? "RECTIFIED" : "NOT rectified");
}

// Returns whether to rectify. "auto": only if the raw stream is not already rectified.
static bool decide_rectification(PairSource& src, const Calibration& cal, const Options& o) {
  info("rectification check: %d pairs, raw vs after the .dwecal maps", o.check_pairs);
  std::vector<cv::Vec4d> raw_pts, rect_pts;
  for (int k = 0; k < o.check_pairs && !g_stop; ++k) {
    RawPair p;
    if (!src.next(p)) break;
    cv::Mat rl, rr;
    cv::remap(p.left, rl, cal.map1[0], cal.map2[0], cv::INTER_LINEAR);
    cv::remap(p.right, rr, cal.map1[1], cal.map2[1], cv::INTER_LINEAR);
    match_rows(p.left, p.right, raw_pts);
    match_rows(rl, rr, rect_pts);
    if (k == 0) {
      cv::Mat d;
      cv::absdiff(p.left, rl, d);
      info("  pair 0: rectified left differs from raw by %.1f grey levels on average", cv::mean(d)[0]);
    }
    if (!o.save_dir.empty()) {                // pairs for offline analysis
      cv::Mat raw, rect;
      cv::hconcat(p.left, p.right, raw);
      cv::hconcat(rl, rr, rect);
      cv::imwrite(o.save_dir + "/raw_" + std::to_string(k) + ".png", raw);
      cv::imwrite(o.save_dir + "/rectified_" + std::to_string(k) + ".png", rect);
    }
    if (o.input_dir.empty()) sleep_ms(200);   // spread over a couple of seconds of motion
  }
  const RowAlignment raw = summarise(raw_pts, cal.width, cal.height);
  const RowAlignment rect = summarise(rect_pts, cal.width, cal.height);
  print_alignment("raw stream", raw);
  print_alignment("after .dwecal maps", rect);
  if (raw.enough() && raw.disparity_median < 0)
    info("  WARNING: negative disparity -- left/right look swapped (--swap_lr)");
  if (rect.enough() && !rect.rectified())
    info("  note: the .dwecal is a water calibration; in air it is not expected to reach 0.5 px");

  bool on;
  if (o.rectify == "on") on = true;
  else if (o.rectify == "off") on = false;
  else if (!raw.enough()) {
    on = true;
    info("  too few matches to judge the raw stream: rectifying (the calibration says it is needed)");
  } else on = !raw.rectified();
  info("VERDICT raw stream: %s -> rectification %s%s", raw.enough() ? (raw.rectified() ? "RECTIFIED" : "NOT rectified")
                                                                  : "unknown",
      on ? "ON (.dwecal maps)" : "OFF (raw used as is)", o.rectify == "auto" ? "" : " (forced by --rectify)");
  return on;
}

// ---------------------------------------------------------------- TensorRT
class TrtLogger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
  }
};

class Engine {
 public:
  explicit Engine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot read %s", path.c_str());
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // Create the CUDA context first: TensorRT compares the engine's GPU with the
    // current one when deserialising (else: "engine plan file across different models").
    cuda_ok(cudaSetDevice(0), "cudaSetDevice");
    cuda_ok(cudaFree(nullptr), "CUDA init");
    // The ONNX parser maps InstanceNorm to a plugin: register plugins first.
    initLibNvInferPlugins(&logger_, "");
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) die("cannot load %s (built by another TensorRT version or GPU?)", path.c_str());
    context_.reset(engine_->createExecutionContext());
    cuda_ok(cudaStreamCreate(&stream_), "cudaStreamCreate");
    for (const char* name : {"left", "right", "disparity"}) {
      const auto d = engine_->getTensorShape(name);
      if (d.nbDims != 4 || engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
        die("engine tensor '%s' missing or not float32 NCHW", name);
      size_t n = 1;
      for (int i = 0; i < d.nbDims; ++i) n *= size_t(d.d[i]);
      void* ptr = nullptr;
      cuda_ok(cudaMalloc(&ptr, n * sizeof(float)), "cudaMalloc");
      buffers_[name] = ptr;
      context_->setTensorAddress(name, ptr);
      if (std::string(name) == "left") height_ = int(d.d[2]), width_ = int(d.d[3]);
    }
    cuda_ok(cudaMallocHost(reinterpret_cast<void**>(&host_out_), size_t(width_) * height_ * sizeof(float)), "cudaMallocHost");
  }
  ~Engine() {
    for (auto& b : buffers_) cudaFree(b.second);
    if (host_out_) cudaFreeHost(host_out_);
    if (stream_) cudaStreamDestroy(stream_);
  }
  int width() const { return width_; }
  int height() const { return height_; }

  // left/right: float32 CHW RGB 0-255 at width x height. Returns disparity (owns its data).
  cv::Mat run(const std::vector<float>& left, const std::vector<float>& right) {
    const size_t bytes = left.size() * sizeof(float);
    cuda_ok(cudaMemcpyAsync(buffers_["left"], left.data(), bytes, cudaMemcpyHostToDevice, stream_), "H2D");
    cuda_ok(cudaMemcpyAsync(buffers_["right"], right.data(), bytes, cudaMemcpyHostToDevice, stream_), "H2D");
    if (!context_->enqueueV3(stream_)) die("TensorRT enqueue failed");
    cuda_ok(cudaMemcpyAsync(host_out_, buffers_["disparity"], size_t(width_) * height_ * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_), "D2H");
    cuda_ok(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    return cv::Mat(height_, width_, CV_32F, host_out_).clone();
  }

 private:
  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::map<std::string, void*> buffers_;
  float* host_out_ = nullptr;
  cudaStream_t stream_ = nullptr;
  int width_ = 0, height_ = 0;
};

// ---------------------------------------------------------------- pipeline stages
// One-item hand-over between threads; a newer item replaces an unconsumed one.
template <typename T>
class Slot {
 public:
  void put(T item) {
    {
      std::lock_guard<std::mutex> lock(m_);
      if (item_) ++dropped_;
      item_ = std::move(item);
    }
    cv_.notify_all();
  }
  bool take(T& out) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait_for(lock, std::chrono::milliseconds(200), [&] { return item_.has_value(); });
    if (!item_) return false;
    out = std::move(*item_);
    item_.reset();
    cv_.notify_all();
    return true;
  }
  void wait_empty() {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait_for(lock, std::chrono::milliseconds(200), [&] { return !item_.has_value(); });
  }
  bool empty() {
    std::lock_guard<std::mutex> lock(m_);
    return !item_.has_value();
  }
  long dropped() const { return dropped_; }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::optional<T> item_;
  long dropped_ = 0;
};

struct Prepared {
  std::vector<float> left, right;        // model input, CHW RGB 0-255
  cv::Mat left_full;                     // rectified left, BGR, full resolution
  int64_t t_left = 0, t_right = 0;
  float prep_ms = 0;
};

struct Inferred {
  cv::Mat disparity;                     // model resolution
  cv::Mat left_full;
  int64_t t_left = 0, t_right = 0;
  float prep_ms = 0, infer_ms = 0;
};

static void to_chw_rgb(const cv::Mat& bgr, std::vector<float>& out) {
  const int n = bgr.rows * bgr.cols;
  out.resize(size_t(3) * n);
  float* r = out.data();
  float* g = r + n;
  float* b = g + n;
  for (int y = 0; y < bgr.rows; ++y) {
    const uint8_t* p = bgr.ptr<uint8_t>(y);
    const int o = y * bgr.cols;
    for (int x = 0; x < bgr.cols; ++x) {
      b[o + x] = p[3 * x];
      g[o + x] = p[3 * x + 1];
      r[o + x] = p[3 * x + 2];
    }
  }
}

// Model disparity (model pixels) -> depth in metres at full resolution.
// Bilinear inside surfaces; at depth edges (local disparity range above
// edge_px) nearest-neighbour, so no "flying" points between foreground and
// background.
static cv::Mat disparity_to_depth(const cv::Mat& disp, int W, int H, double fx_baseline, double edge_px) {
  const double sx = double(W) / disp.cols;
  cv::Mat lin, nn, dmax, dmin, range_low, range;
  cv::resize(disp, lin, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
  cv::resize(disp, nn, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
  const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(disp, dmax, k);
  cv::erode(disp, dmin, k);
  range_low = dmax - dmin;
  cv::resize(range_low, range, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
  cv::Mat depth(H, W, CV_32F);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float fb = float(fx_baseline), scale = float(sx), edge = float(edge_px);
  for (int y = 0; y < H; ++y) {
    const float *l = lin.ptr<float>(y), *n = nn.ptr<float>(y), *r = range.ptr<float>(y);
    float* d = depth.ptr<float>(y);
    for (int x = 0; x < W; ++x) {
      const float m = (r[x] > edge ? n[x] : l[x]) * scale;   // full-resolution pixels
      d[x] = m > 0.5f ? fb / m : nan;
    }
  }
  return depth;
}

class UdpSender {
 public:
  UdpSender(const std::string& ip, int port, double mbps, int packet_bytes)
      : bytes_per_ns_(mbps * 1e6 / 8.0 / 1e9), payload_(packet_bytes - int(sizeof(PacketHeader))) {
    fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) die("socket: %s", std::strerror(errno));
    int buf = 4 << 20;
    setsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(uint16_t(port));
    if (inet_pton(AF_INET, ip.c_str(), &a.sin_addr) != 1) die("bad --aib address %s", ip.c_str());
    if (connect(fd_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0) die("connect %s: %s", ip.c_str(), std::strerror(errno));
  }
  ~UdpSender() { if (fd_ >= 0) ::close(fd_); }

  // Sends one blob, paced to --send_mbps. Returns false if datagrams failed.
  bool send(uint32_t frame_id, const std::vector<uint8_t>& blob) {
    const uint32_t count = uint32_t((blob.size() + payload_ - 1) / payload_);
    std::vector<uint8_t> packet(sizeof(PacketHeader) + payload_);
    const int64_t start = mono_ns();
    size_t sent = 0;
    bool ok = true;
    for (uint32_t i = 0; i < count; ++i) {
      PacketHeader h;
      h.frame_id = frame_id;
      h.chunk_index = i;
      h.chunk_count = count;
      h.offset = uint32_t(size_t(i) * payload_);
      h.frame_bytes = uint32_t(blob.size());
      const size_t n = std::min(size_t(payload_), blob.size() - h.offset);
      std::memcpy(packet.data(), &h, sizeof(h));
      std::memcpy(packet.data() + sizeof(h), blob.data() + h.offset, n);
      if (::send(fd_, packet.data(), sizeof(h) + n, 0) < 0 && errno != ECONNREFUSED) ok = false;
      sent += sizeof(h) + n;
      const int64_t due = start + int64_t(sent / bytes_per_ns_);
      const int64_t now = mono_ns();
      if (due - now > 200000) sleep_ms((due - now) / 1e6);
    }
    return ok;
  }

 private:
  int fd_ = -1;
  double bytes_per_ns_;
  int payload_;
};

// ---------------------------------------------------------------- main
struct Stats {
  std::mutex m;
  long frames = 0, send_errors = 0;
  double prep = 0, infer = 0, post = 0, send = 0, age = 0, skew = 0;
  void add(double p, double i, double po, double s, double a, double k) {
    std::lock_guard<std::mutex> lock(m);
    ++frames, prep += p, infer += i, post += po, send += s, age += a, skew += std::fabs(k);
  }
};

int main(int argc, char** argv) {
  const Options o = parse_args(argc, argv);
  std::signal(SIGINT, [](int) { g_stop = true; });
  std::signal(SIGTERM, [](int) { g_stop = true; });
  std::signal(SIGPIPE, SIG_IGN);

  const Calibration cal = load_dwecal(o.dwecal);
  info("dwecal %s: %dx%d, rectified fx %.3f fy %.3f cx %.3f cy %.3f, baseline %.4f m", o.dwecal.c_str(), cal.width,
      cal.height, cal.fx, cal.fy, cal.cx, cal.cy, o.baseline_m);

  std::unique_ptr<PairSource> source;
  if (!o.input_dir.empty()) source = std::make_unique<FilePairs>(o, cal.width, cal.height);
  else {
    source = std::make_unique<CameraPairs>(o, cal.width, cal.height);
    info("cameras: left %s, right %s, MJPEG %dx%d @ %d fps", o.swap_lr ? o.right_dev.c_str() : o.left_dev.c_str(),
        o.swap_lr ? o.left_dev.c_str() : o.right_dev.c_str(), cal.width, cal.height, o.cam_fps);
    sleep_ms(500);                       // let both cameras start streaming
  }

  const bool rectify = (o.rectify == "auto" || o.check_only) ? decide_rectification(*source, cal, o) : o.rectify == "on";
  if (o.check_only) return 0;
  if (o.rectify != "auto") info("rectification %s (--rectify %s)", rectify ? "ON" : "OFF", o.rectify.c_str());

  Engine engine(o.engine);
  const int W = cal.width, H = cal.height, w = engine.width(), h = engine.height();
  if (std::fabs(double(W) / w - double(H) / h) > 1e-6) die("engine %dx%d has another aspect ratio than %dx%d", w, h, W, H);
  // Disparity is computed at the model size; depth uses full-resolution fx with
  // the disparity scaled up by W / w (inside disparity_to_depth).
  const double fx_baseline = cal.fx * o.baseline_m;
  info("engine %s: %dx%d -> depth %dx%d; sending to %s:%d (%.0f Mbit/s, %d B datagrams)", o.engine.c_str(), w, h, W, H,
      o.aib.c_str(), o.port, o.send_mbps, o.packet_bytes);
  {
    std::vector<float> z(size_t(3) * w * h, 0.f);
    engine.run(z, z);                    // warm-up
  }

  Slot<Prepared> to_infer;
  Slot<Inferred> to_send;
  Stats stats;
  std::atomic<int64_t> infer_free_at{0};               // when the engine is expected to be free
  std::atomic<double> prep_ema{40.0}, infer_ema{170.0};
  std::atomic<bool> failed{false};

  // Stage 1: pick the freshest pair, timed so it is ready just as the engine frees up.
  std::thread prep_thread([&] {
    while (!g_stop) {
      to_infer.wait_empty();
      if (!to_infer.empty()) continue;
      const double lead_ms = prep_ema + 5.0;
      const double wait = (infer_free_at.load() - mono_ns()) / 1e6 - lead_ms;
      if (wait > 0) sleep_ms(std::min(wait, 100.0));
      if ((infer_free_at.load() - mono_ns()) / 1e6 > lead_ms + 1.0) continue;
      const int64_t t0 = mono_ns();      // includes waiting for the pair and decoding it
      RawPair p;
      if (!source->next(p)) {
        if (!g_stop) failed = true;
        g_stop = true;
        break;
      }
      Prepared out;
      cv::Mat rect[2];
      const cv::Mat* raw[2] = {&p.left, &p.right};
      auto work = [&](int eye) {
        if (rectify) cv::remap(*raw[eye], rect[eye], cal.map1[eye], cal.map2[eye], cv::INTER_LINEAR);
        else rect[eye] = *raw[eye];
        cv::Mat small;
        cv::resize(rect[eye], small, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        to_chw_rgb(small, eye == 0 ? out.left : out.right);
      };
      auto right_job = std::async(std::launch::async, work, 1);
      work(0);
      right_job.get();
      out.left_full = rect[0];
      out.t_left = p.t_left;
      out.t_right = p.t_right;
      out.prep_ms = float((mono_ns() - t0) / 1e6);
      prep_ema = 0.8 * prep_ema + 0.2 * out.prep_ms;
      to_infer.put(std::move(out));
    }
  });

  // Stage 2: inference.
  std::thread infer_thread([&] {
    int64_t last_start = 0;
    while (!g_stop) {
      Prepared in;
      if (!to_infer.take(in)) continue;
      if (o.max_hz > 0) {
        const double gap = (last_start + 1e9 / o.max_hz - mono_ns()) / 1e6;
        if (gap > 0) sleep_ms(gap);
      }
      const int64_t t0 = mono_ns();
      last_start = t0;
      infer_free_at = std::max(t0 + int64_t(infer_ema * 1e6), o.max_hz > 0 ? t0 + int64_t(1e9 / o.max_hz) : 0);
      Inferred out;
      out.disparity = engine.run(in.left, in.right);
      out.infer_ms = float((mono_ns() - t0) / 1e6);
      infer_ema = 0.8 * infer_ema + 0.2 * out.infer_ms;
      // With --max_hz the next inference starts later; prepare for then, not now.
      infer_free_at = std::max(mono_ns(), o.max_hz > 0 ? last_start + int64_t(1e9 / o.max_hz) : 0);
      out.left_full = std::move(in.left_full);
      out.t_left = in.t_left;
      out.t_right = in.t_right;
      out.prep_ms = in.prep_ms;
      to_send.put(std::move(out));
    }
  });

  // Stage 3: depth at full resolution, JPEG, send.
  std::thread send_thread([&] {
    UdpSender udp(o.aib, o.port, o.send_mbps, o.packet_bytes);
    uint32_t frame_id = 0;
    std::vector<uint8_t> blob;
    while (!g_stop) {
      Inferred in;
      if (!to_send.take(in)) continue;
      const int64_t t0 = mono_ns();
      std::vector<uint8_t> jpeg;
      auto jpeg_job = std::async(std::launch::async, [&] {
        cv::imencode(".jpg", in.left_full, jpeg, {cv::IMWRITE_JPEG_QUALITY, o.jpeg_quality});
      });
      const cv::Mat depth = disparity_to_depth(in.disparity, W, H, fx_baseline, o.edge_px);
      jpeg_job.get();

      FrameHeader fh;
      fh.frame_id = frame_id;
      fh.width = W, fh.height = H;
      fh.fx = cal.fx, fh.fy = cal.fy, fh.cx = cal.cx, fh.cy = cal.cy;
      fh.baseline_m = float(o.baseline_m);
      fh.depth_offset = sizeof(FrameHeader);
      fh.depth_bytes = uint32_t(size_t(W) * H * sizeof(float));
      fh.image_offset = fh.depth_offset + fh.depth_bytes;
      fh.image_bytes = uint32_t(jpeg.size());
      fh.rectified = rectify ? 1 : 0;
      fh.infer_width = w, fh.infer_height = h;
      fh.lr_skew_us = int32_t((in.t_right - in.t_left) / 1000);
      fh.prep_ms = in.prep_ms, fh.infer_ms = in.infer_ms;
      fh.post_ms = float((mono_ns() - t0) / 1e6);
      fh.capture_mono_ns = uint64_t(in.t_left);
      const int64_t send_start = mono_ns();
      fh.age_us = uint32_t((send_start - in.t_left) / 1000);
      fh.sent_wall_ns = uint64_t(wall_ns());
      blob.resize(size_t(fh.image_offset) + jpeg.size());
      std::memcpy(blob.data(), &fh, sizeof(fh));
      std::memcpy(blob.data() + fh.depth_offset, depth.data, fh.depth_bytes);
      std::memcpy(blob.data() + fh.image_offset, jpeg.data(), jpeg.size());
      if (!udp.send(frame_id, blob)) {
        std::lock_guard<std::mutex> lock(stats.m);
        ++stats.send_errors;
      }
      const double send_ms = (mono_ns() - send_start) / 1e6;
      stats.add(in.prep_ms, in.infer_ms, fh.post_ms, send_ms, fh.age_us / 1000.0, fh.lr_skew_us / 1000.0);

      if (!o.save_dir.empty() && int(frame_id) < o.save_frames) {
        // left: rectified PNG; disparity (model size) and depth (full size): raw float32
        const std::string base = o.save_dir + "/frame" + std::to_string(frame_id);
        cv::imwrite(base + "_left.png", in.left_full);
        std::ofstream(base + "_disparity_" + std::to_string(w) + "x" + std::to_string(h) + ".f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(in.disparity.data), std::streamsize(in.disparity.total() * 4));
        std::ofstream(base + "_depth_" + std::to_string(W) + "x" + std::to_string(H) + ".f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(depth.data), std::streamsize(depth.total() * 4));
      }
      ++frame_id;
    }
  });

  // Status line every report_s.
  long last_frames = 0;
  int64_t last_t = mono_ns();
  while (!g_stop) {
    sleep_ms(100);
    if ((mono_ns() - last_t) / 1e9 < o.report_s) continue;
    std::lock_guard<std::mutex> lock(stats.m);
    const long n = stats.frames - last_frames;
    const double dt = (mono_ns() - last_t) / 1e9;
    if (n > 0)
      info("%.2f Hz | prep %.0f ms, infer %.0f ms, depth+jpeg %.0f ms, send %.0f ms | capture->sent %.0f ms | "
          "L/R skew %.1f ms | dropped %ld | send errors %ld",
          n / dt, stats.prep / n, stats.infer / n, stats.post / n, stats.send / n, stats.age / n, stats.skew / n,
          to_infer.dropped() + to_send.dropped(), stats.send_errors);
    else info("no frames in the last %.0f s", dt);
    stats.prep = stats.infer = stats.post = stats.send = stats.age = stats.skew = 0;
    last_frames = stats.frames;
    last_t = mono_ns();
  }
  prep_thread.join();
  infer_thread.join();
  send_thread.join();
  source.reset();
  info("stopped after %ld frames", stats.frames);
  return failed ? 1 : 0;
}
