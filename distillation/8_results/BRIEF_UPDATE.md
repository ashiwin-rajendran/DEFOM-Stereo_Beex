# DEFOM stereo depth on the Orin: brief update (2026-10-30)

**Goal:** live depth from the DWE stereo cameras, computed on the vehicle's SVC (Jetson Orin
NX) at 5-6 Hz and published on ROS at 1600x1200, for inspection within 1.5 m.

**Status: achieved.** Two models now run at ~6 Hz on the Orin, measured live on the vehicle.

| model | what it is | resolution | Orin NX | accuracy on dives | accuracy on new scenes (tank, lab) |
|---|---|---|---|---|---|
| **slim v9** | our small model, distilled from DEFOM ViT-L on our dive footage | 800x600 | 5.8 Hz | **69%** | 27% |
| **ViT-S s2** | DEFOM's own small ViT-S model, made faster without retraining | 512x384 | 6.05 Hz (5.8 Hz live) | 63% | **74%** |

Accuracy = % of pixels within 3 px of the large DEFOM ViT-L model (1.5 GB, far too slow for
the Orin), which serves as the reference.

**How it runs.**
- **SVC:** the Orin captures both cameras, rectifies them with DWE's calibration and runs the
  model (TensorRT). It upsamples the depth to 1600x1200 and sends it with the left image over
  UDP.
- **AIB:** receives it and publishes depth, a point cloud and the image on ROS.
- **Latency:** ~0.47 s from capture to the AIB.

**What we learned.**
- **Which model to use:** slim v9 is the most accurate on dive sites like the ones it was
  trained on, at full 800x600 detail. ViT-S s2 holds up on unfamiliar scenes (tank, lab).
  ViT-S s2 is the safer default for new sites; slim v9 is the choice where the footage
  resembles its training dives.
- **Why slim v9 fails on new scenes:** it learned stereo matching from our ~11k dive frames
  alone. DEFOM's own ViT-S was trained on ~17 public datasets. Every attempt to make ViT-S
  faster by retraining it on our dives lost that generality. What worked was keeping its
  weights and making it cheaper: half-resolution feature networks, a smaller input, fewer
  iterations.
- **How to test:** testing only on held-out frames from the training dives hid this; new
  scenes must be part of every test.

**Limitations.**
- **Thin objects:** ViT-S s2's 512x384 input under-resolves thin objects (a 5 mm cable at 1 m
  is ~1.5 px wide); 640x480 at 3.4 Hz is the step up.
- **No ground truth yet:** accuracy is measured against the large model, not against true
  distances.

**Next steps.**
1. Add public stereo datasets to training, so the fast models generalise beyond our dives.
   Both models have ready fine-tuning configs.
2. Test in the tank at 0.5-1.5 m on known targets (a flat plate, 3/5/10 mm cables) for an
   absolute accuracy check and a thin-object check.
3. Optional: send depth at the model's resolution and upsample on the AIB, to cut ~0.1 s of
   latency and most of the network load.

Details: [RESULTS.md](RESULTS.md) (every model tested, with figures) and
[../README.md](../README.md) (how to train, evaluate and deploy).
