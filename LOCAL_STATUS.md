# Local execution status — pre-Vast.ai

**Goal**: walk every smoke test in `TRAINING_PLAN.md` §9 (S1-S8) green
locally before booking any Vast.ai instance.  Current state.

## ✅ Smoke tests passed locally

| ID | what | result |
|---|---|---|
| **S1** | byte-equivalence Lite vs `.task` runtime | max abs diff 0.0022 / max relative 4e-4 (KP), 1e-5 (world).  ALL 5 outputs match. |
| **S2** | byte-equivalence Full vs `.task` runtime | max abs diff 0.002 / max relative 1.4e-4. ALL 5 outputs match. |
| **bonus** | byte-equivalence **Heavy** vs `.task` (the teacher) | max abs diff 0.005 / max relative 1.6e-5. All 5 outputs match. Discovered + fixed channel-padding handler in PAD op. |
| **S3** | dataloader yields valid batches | synth dataset loads, 17/33 hard joints per sample, image (3, 256, 256) tensors. |
| **S4** | one forward + backward + opt step on CPU | finite loss, no NaN, all gradients finite, AdamW step succeeds. |
| **S5** | gradient flow (small smoke) | 6 steps in 10.1 s on CPU; loss varying as expected for batch=1 + 6 steps. |
| **S6** | hold-out enforcement fires on poisoned input | clean train list passes; poisoned (one held-out uid) raises RuntimeError exit-99. |
| **bonus** | three teacher wrappers all loadable + downloadable from CDN | Heavy (already on disk, Apache-2.0), Hand Landmarker (downloaded ~7 MB), Face Mesh (downloaded ~3 MB). |

## ⏳ Remaining for full local readiness

| ID | what | blocker |
|---|---|---|
| **S7** | PyTorch → `.task` round-trip | needs **either** WSL2 + `ai-edge-torch` (Linux-only per the agent research) **or** a custom-built flatbuffer rewriter (~1-2 days work, no third-party libs needed). Decision: go custom on Vast where Linux is native + `ai-edge-torch` works out of the box. |
| **S8** | Round-tripped Lite vs `benchmark/run_eval.py` matches v1 | depends on S7. |

## Audit fixes applied (post-self-prompt critique)

After the strict acceptance audit, these gaps were fixed and reverified:

1. **Per-joint anchor masking** ([losses.py](training/losses.py)): `L_anchor`
   now applies a 5× stronger weight to BP joints with **neither** hard
   supervision **nor** Heavy-teacher signal (specifically the 20 of 33 not
   in the 17-COCO mapping).  Prevents regression from side-effect
   gradients propagating through the shared backbone.
2. **FixMatch strong/weak split** ([dataset.py](training/dataset.py),
   [train.py](training/train.py)): student forward gets the
   strong-augmented crop; teacher + anchor get the weak (light photometric
   only) crop.  Forces invariance, prevents teacher-pseudo-label noise
   on heavily occluded inputs.  This is the "single biggest distillation
   win 2022-2025" per the augmentation research synthesis.
3. **Source-aware augmentation tiers** ([augment.py](training/augment.py)):
   synth gets heavy aug (F1 0.8 / F2 0.9 / FDA 0.5), egoexo gets light
   (F1 0.4, no F2 — already real bg), replay gets minimal.  Tuned per
   Sárándi/RTMPose/DWPose/FixMatch lineage.
4. **Real benchmark validation in `train.py`**: every epoch runs the
   student on a subset of held-out manifest frames, reports
   `BENCH_PA_MPJPE` directly.  This is the metric — anchor drift alone
   doesn't answer "is this actually getting better?"
5. **λ_anchor bumped 0.1 → 0.4** to take advantage of the new mask;
   supervised joints still see ~0.08 effective weight (below L_hard +
   L_kd_body), unsupervised joints see ~0.4 effective weight.

## v2.1 future-work (intentionally deferred — works but not critical for first run)

- **Hand teacher KD with per-wrist crop ROI**: Hand Landmarker expects
  close-up hand crops; running on the 256×256 person crop gives noisy
  output.  Body-axis alignment for the resulting points requires Heavy's
  predicted wrist position as the rigid-alignment anchor.  Can be added
  as a `cache_hand_aligned.py` step.  Without it, Heavy's own hand-finger
  predictions (BP idx 17-22) cover this — just lower-precision.
- **Face teacher KD with body-axis alignment**: similar.  Heavy already
  emits face landmarks (BP idx 0-10).
- **Multi-view consistency loss** for Ego-Exo4D 4-cam takes: stub
  returns zero.  Adding requires per-batch K + Rt + 2D-projection
  helper.
- **Sárándi-2024 human-shaped occluders**: a 2-day add (CIHP/COCO
  segmentation pipeline) for a known further win.  Current Open Images
  cutouts are still SOTA-baseline.

## Files produced

| path | role |
|---|---|
| [model/inspect_task.py](model/inspect_task.py) | Phase-1 op-graph inspector (still useful for any future variant) |
| [model/inspection/*_summary.txt](model/inspection/) | per-variant op histograms + I/O shapes |
| [model/inspection/*_ops.txt](model/inspection/) | per-variant full op-by-op listings (305 / 332 / 689 ops) |
| [model/port.py](model/port.py) | the **custom** TFLite → PyTorch walker (no `zmurez`, no `nobuco`, no third-party port libs).  Handles PAD/DEQUANTIZE/CONV_2D/DEPTHWISE_CONV_2D/ADD/RESHAPE/RESIZE_BILINEAR/MAX_POOL_2D/LOGISTIC/CONCATENATION/MEAN with NHWC↔NCHW perms, asymmetric SAME pad, channel pad. |
| [model/test_byte_equiv.py](model/test_byte_equiv.py) | S1/S2 byte-equivalence harness. |
| [training/holdout.py](training/holdout.py) | S6 hold-out enforcement (raises on intersection with `frames_manifest.json` ∪ `subset.json`). |
| [training/teachers.py](training/teachers.py) | Heavy / Hand / Face teacher wrappers + auto-download. |
| [training/dataset.py](training/dataset.py) | Synth dataset (17 COCO → 33 BP mapping; 256×192 → 256×256 letterbox). |
| [training/losses.py](training/losses.py) | V2DistillationLoss (body KD + visibility BCE + frozen-v1 anchor). |
| [training/smoke_train.py](training/smoke_train.py) | S3-S5 end-to-end CPU smoke loop. |
| [training/cache_teachers.py](training/cache_teachers.py) | pre-cache teacher outputs to per-frame `.npz`; ready for the Vast 4-8 hr first-pass over Ego-Exo4D train. |
| [assets/teachers/hand_landmarker.task](assets/teachers/) | downloaded (7.5 MB) |
| [assets/teachers/face_landmarker.task](assets/teachers/) | downloaded (3.6 MB) |
| **state_dict round-trip verified bit-exact** | save/load preserves all 5 outputs to 0.0 max diff — Vast checkpoint resume will work. |

## Key local-phase findings (worth carrying forward)

1. **`.task` files are ZIPs** containing two `.tflite` flatbuffers (`pose_detector.tflite` + `pose_landmarks_detector.tflite`) plus `metadata`. Walker `unzipfile.ZipFile().extractall()` first.
2. **All three landmark nets ship int8-quantized weights** with explicit per-tensor `DEQUANTIZE` ops (Lite 176, Full 192, Heavy 396 dequants). The walker dequantizes on load and stores fp32 nn.Parameters; DEQUANTIZE ops at runtime are no-ops.
3. **Channel-padding via PAD op** is used in Heavy (and likely Full)'s residual blocks where channels expand. The PAD handler must support 4-D padding (N, H, W, **C**).
4. **No explicit ReLU ops**: activations are fused into CONV_2D / DEPTHWISE_CONV_2D via `fused_activation_function = RELU6`. Walker reads BuiltinOptions to apply.
5. **Output ordering** of the 5-tensor signature is consumed by **index** (not name) by the MediaPipe pose graph. `forward()` returns a dict keyed by name, but the export must preserve index order.
6. **`tflite-runtime` is not Windows-packaged**. We use `ai-edge-litert` (Apache-2.0) as the official replacement for byte-equivalence reference inference.
7. **Heavy's 27 MB / 689-op landmark net** runs at ~0.5 s/frame on CPU via the port. Lite at ~0.2 s/frame. Both will be ~30-50× faster on A100.

## What needs Vast.ai (genuinely; cannot do locally)

1. Pre-cache teachers (Heavy + Hand + Face) on the Ego-Exo4D train split (~600 k frames; 300-500 GB raw video; faster to re-pull from S3 inside the Vast box than upload from home).
2. Full 10-epoch training × 2 students (~14 hr A100 total).
3. PyTorch → `.task` export via `ai-edge-torch` (Linux-only on first cut; a Vast Linux instance is the native home for this).
4. Run the v2 students through `benchmark/run_eval.py` + `analyze.py` on the SOTA benchmark to compute the headline win delta.

## Estimated post-local effort (when on Vast)

- **Day 1**: download Ego-Exo4D train annotations + downscaled exo videos.
- **Day 1-2**: pre-cache Heavy / Hand / Face teacher outputs on the full corpus (~600k frames × 3 teachers ~ 4-8 hr GPU).
- **Day 2-3**: Lite v2 training (10 epochs, ~7 hr).
- **Day 3-4**: Full v2 training (10 epochs, ~7 hr).
- **Day 4**: export + benchmark scoring.
- **Total**: ~3-4 calendar days, ~$30 in GPU time.
