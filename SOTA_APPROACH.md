# SOTA approach: improving BlazePose Lite/Full

**Goal:** push the published Apache-2.0 BlazePose **Lite** and **Full** landmark
checkpoints closer to (or onto) the **Heavy** error curve, *without* changing
the on-device runtime cost. The output is two drop-in `.tflite` files
(`pose_landmark_lite_v2.tflite`, `pose_landmark_full_v2.tflite`) consumable by
the existing MediaPipe graph and detector.

This document is the architectural / training-recipe summary. Implementation
details belong in the code we will write under this folder.

---

## 0. Constraints (reread before every decision)

- **Strictly commercial-clean.** No SMPL / SMPL-X / AMASS / BEDLAM / AGORA /
  Human3.6M / 3DPW / MPI-INF-3DHP / EMDB / RICH / MoYo / GHUM / Sapiens /
  COCO-WholeBody / Mixamo / RenderPeople-non-free / CC-BY-NC anything.
  See [licensing/LICENSE_AUDIT.md](licensing/LICENSE_AUDIT.md).
- **Continued pre-training, not from-scratch.** We initialise from the
  published Apache-2.0 BlazePose Lite/Full weights and adapt them; we do not
  retrain the architecture from a cold start.
- **No `zmurez/MediaPipePyTorch`.** We re-implement the BlazePose landmark
  network in PyTorch from scratch — straight from the public
  CVPR-W 2020 paper ("BlazePose: On-device Real-time Body Pose Tracking",
  Bazarevsky et al.) and the `pose_landmark` model card / pbtxt. Weight
  porting is a one-shot conversion from the Apache-2.0 `.tflite` files
  shipped under `assets/`.
- **Deployment parity.** The exported student must match the original
  graph's input/output tensor names, shapes, and dtypes so the MediaPipe
  C++ graph runs unmodified.

## 1. Architecture (recreate ourselves)

BlazePose landmark network has three published variants (Lite / Full / Heavy);
all share the same I/O contract:

| Tensor                | Shape (NHWC, TFLite)            | Note                                                       |
|-----------------------|----------------------------------|------------------------------------------------------------|
| `input_1`             | `1 × 256 × 256 × 3` float32      | ROI-cropped + rotated body crop                            |
| `Identity` (kps)      | `1 × 195` float32                | 39 × (x, y, z, vis, presence) for 33 KPs + 6 virtual KPs   |
| `Identity_1` (flag)   | `1 × 1` float32                  | "pose presence" sigmoid                                    |
| `Identity_2` (seg)    | `1 × 256 × 256 × 1` float32      | foreground segmentation logits                             |
| `Identity_3` (heat)   | `1 × 64 × 64 × 39` float32       | optional heatmap branch                                    |
| `Identity_4` (world)  | `1 × 117` float32                | 39 × (x, y, z) world-space metres                          |

Backbone is a stack of MobileNetV2-style depthwise-separable + residual
"Blaze blocks" (3×3 dw + 1×1 pw, with optional 5×5 dw in the late stages),
ending in a global-pool + FC head for the 195-D landmark vector and a
small upsampling head for the segmentation map. Lite, Full, and Heavy
differ only in channel widths and depth; the head shapes are identical.

**Re-implementation plan:**

1. `model.py`: PyTorch `nn.Module` mirroring the layer order encoded in the
   `pose_landmark_*.tflite` flatbuffers (read with
   `tensorflow.lite.python.schema_py_generated` — Apache-2.0).
2. `port_weights.py`: walk the `.tflite` operator graph, copy each
   constant tensor into the matching PyTorch parameter (name-by-shape +
   topological order). Verify by running an identical 256×256 input
   through both runtimes and asserting `max(|out_tflite - out_torch|) <
   1e-4` per tensor.
3. `export.py`: round-trip back to TFLite via **Google AI Edge Torch**
   (Apache-2.0), preserving I/O tensor names so the MediaPipe graph is
   plug-compatible.

The architecture is public; the weights are Apache-2.0; nothing in this
chain is licence-tainted. We never read or copy `zmurez/MediaPipePyTorch`.

## 2. Data sources (commercial-clean)

| Source                                | Use                                        | Licence                       |
|---------------------------------------|--------------------------------------------|-------------------------------|
| Ego-Exo4D (signed commercial licence) | hard 3-D supervision, 17 COCO body KPs     | Meta commercial-research      |
| MediaPipe Pose Heavy `.task`          | KD teacher: full 33 BlazePose body KPs     | Apache-2.0                    |
| MediaPipe Hand Landmarker `.task`     | KD teacher: 21 KPs/hand → BlazePose 17–22  | Apache-2.0                    |
| MediaPipe Face Mesh `.task`           | KD teacher: face KPs → BlazePose 1–10      | Apache-2.0                    |
| Open Images V7 cutouts (existing)     | training-time occlusion + bg compositing   | CC-BY-2.0 / Apache-2.0        |
| Poly Haven HDRIs (existing)           | environment lighting if synth is added     | CC-0                          |

Notable **exclusions**:
- **GHUM** — Google's parametric body model, research-only. We rely on
  Ego-Exo4D's manual annotations + teacher pseudolabels instead.
- **COCO-WholeBody / Halpe** — annotation licences not commercial-clean.
- **AMASS-derived synth** (BEDLAM, AGORA, MoYo, etc.) — all SMPL-tainted.

## 3. Keypoint mapping (BlazePose's 33 indices)

Hard-supervised vs teacher-only is fixed by what Ego-Exo4D actually labels:

| BlazePose idx | Name              | Source            |
|---------------|-------------------|-------------------|
| 0             | nose              | Ego-Exo4D (hard)  |
| 1, 4          | eye-inner L/R     | Face Mesh (KD)    |
| 2, 5          | eye L/R           | Ego-Exo4D partial + Face Mesh |
| 3, 6          | eye-outer L/R     | Face Mesh (KD)    |
| 7, 8          | ear L/R           | Ego-Exo4D partial + Face Mesh |
| 9, 10         | mouth L/R         | Face Mesh (KD)    |
| 11–16         | shoulder/elbow/wrist L/R | Ego-Exo4D (hard) |
| 17–22         | hand pinky/index/thumb L/R | Hand Landmarker (KD) |
| 23–28         | hip/knee/ankle L/R | Ego-Exo4D (hard)  |
| 29–32         | heel/foot-index L/R | BlazePose Heavy (KD) |

13 hard, 4 partial-hard, 16 KD-only. Loss masks per-KP per-sample so
absent supervisory signals contribute zero gradient.

## 4. Coordinate transform (the easy place to introduce a bug)

BlazePose's z is **weak-perspective-orthographic** in the ROI's local
frame, scaled by the same factor that maps metric XY into the 256-px
ROI:

```
z_norm = s · Δz_metric          where s = roi_pixel_size / metric_size
```

Not perspective depth. Not a fixed metres → unit-cube. This is the trap
that bricks every from-scratch BlazePose port. Hard-supervised z values
from Ego-Exo4D camera-space metres MUST be converted via the ROI scale
the augmentation pipeline produced — store `s` in the sample dict.

Visibility/presence are sigmoid scalars per KP; world coords are metres
in the camera frame (unaffected by the ROI transform).

## 5. Multi-teacher ensemble distillation

For every training image we ALSO run the three teachers and persist
their outputs (one-shot, before training, to keep epoch wallclock down):

- **Body teacher = BlazePose Heavy** → all 33 KPs in BlazePose's own
  ROI/coord convention. Direct soft target for KPs 11–32.
- **Hand teacher = Hand Landmarker** → run on each detected wrist crop;
  pinky/index/thumb tips (idx 17–22) are taken from the hand model's KP
  20 / 8 / 4 respectively.
- **Face teacher = Face Mesh** → 478 KPs; we map landmarks 263/362/33/133
  (eye corners), 468/473 (eye centres), 234/454 (ears proxy), 61/291
  (mouth) to BlazePose 1–10.

Each teacher contributes a per-KP soft target with a confidence weight.
KD loss is **Smooth-L1 (β = 0.02)** on the (x, y, z) triple, weighted by
teacher confidence × per-KP availability mask.

## 6. Multi-view triangulation residual (Ego-Exo4D's lever)

Ego-Exo4D ships 4 synchronised exo cameras per take. For any frame
where ≥ 2 exo views see the same person, we add a **triangulation
residual** loss:

```
L_tri = Σ_{view v} || project(X_world, K_v, Rt_v) - x_pred^v ||_smooth-L1
```

`X_world` is computed *from the predictions of the other views* (DLT
triangulation, weighted by visibility). This is what closes the gap on
self-occluded KPs that single-view 2-D-only training never resolves.

## 7. Augmentation (source-aware)

Reuse the existing F1/F2/FDA library at
[tooling/sim2real_aug.py](tooling/sim2real_aug.py) but **gentler on
real frames** (Ego-Exo4D photos already have realistic lighting,
sensor noise, and motion blur — heavy photometric on top of that
just collapses signal):

| Aug                       | Synth        | Real (Ego-Exo4D) |
|---------------------------|--------------|------------------|
| Geometric (rot ±15°, jitter ±10%) | yes  | yes              |
| F1: occluder paste        | p = 0.6      | p = 0.3          |
| F2: BG composite (matte)  | p = 0.5      | n/a (real bg)    |
| FDA (frequency-domain DA) | p = 0.3      | p = 0.05         |
| Photometric (synth recipe)| yes          | **no**           |
| Photometric (real recipe) | n/a          | mild             |

Photometric "real recipe" = ColorJitter(±0.15, ±0.15, ±0.1) +
ISO-noise(σ ≤ 0.01) + 50/50 flip. Nothing else.

## 8. Loss

Total loss per sample (sum of weighted terms):

| term            | weight | function                            | scope                         |
|-----------------|--------|-------------------------------------|-------------------------------|
| `hard_l1`       | 1.0    | Smooth-L1 (β = 0.05 m)              | 17 hard KPs from Ego-Exo4D    |
| `kp_kd`         | 0.5    | Smooth-L1 (β = 0.02)                | KD targets, KPs 1–32          |
| `face_kd`       | 0.3    | Smooth-L1                           | KPs 1, 3, 4, 6, 9, 10         |
| `hand_kd`       | 0.3    | Smooth-L1                           | KPs 17–22                     |
| `vis`           | 0.1    | BCE-with-logits                     | per-KP visibility/presence    |
| `seg`           | 0.2    | BCE + Dice                          | foreground mask (Heavy teacher pseudolabel) |
| `multi_view`    | 0.5    | triangulation residual (Smooth-L1)  | frames with ≥ 2 exo views     |

KD weights are halved during the warmup epoch and ramped to full by
epoch 3.

## 9. Training recipe

| field                | value                                          |
|----------------------|------------------------------------------------|
| optim                | AdamW                                          |
| lr                   | `1e-4` peak                                    |
| schedule             | 1-epoch linear warmup → 11-epoch cosine        |
| weight decay         | `1e-4` (excl. norms / biases)                  |
| batch                | 96 / GPU (Vast.ai A6000 ×2 → effective 192)    |
| epochs               | 12                                             |
| EMA                  | decay `0.9998`, kept for export                |
| frozen               | segmentation head (use Heavy's frozen output)  |
| unfrozen             | full backbone + landmark head                  |
| amp                  | bf16                                           |
| grad clip            | `1.0`                                          |
| seed                 | 42 (deterministic dataloaders)                 |

Two passes:
1. **Lite student** initialised from `pose_landmark_lite.tflite`.
2. **Full student** initialised from `pose_landmark_full.tflite`.

Heavy is the teacher in both — never a student.

## 10. Export

Google AI Edge Torch (Apache-2.0) handles **PyTorch → TFLite** directly,
preserving op semantics. Output:

```
exports/pose_landmark_lite_v2.tflite
exports/pose_landmark_full_v2.tflite
```

Validation: side-by-side run through the MediaPipe pose graph against
the v1 weights; KPs/visibility/seg outputs must be byte-compatible
(same tensor shapes, same dtypes, same names). On `assets/testvideo.mp4`
we expect:

- **Lite v2**: 3-D MAE 38–42 mm (vs. ~45 mm baseline)
- **Full v2**: 3-D MAE 36–37 mm (matching Heavy's 36 mm at Full's
  on-device cost)

Numbers are computed against Heavy's predictions on the test video
(since we have no commercial-clean ground-truth eval set with 33
BlazePose-style KPs); a separate held-out Ego-Exo4D split provides
the absolute MPJPE in metres on the 17 COCO subset.

## 11. Folder layout (target end-state)

```
BlazePose tune/
├── SOTA_APPROACH.md          (this file)
├── assets/
│   ├── pose_landmarker_lite.task
│   ├── pose_landmarker_full.task
│   ├── pose_landmarker_heavy.task
│   ├── sim2real_refs/        (Open Images V7 corpus)
│   └── testvideo.mp4
├── licensing/
│   ├── LICENSE_AUDIT.md
│   └── THIRD_PARTY_NOTICES.md
├── tooling/
│   ├── sim2real_aug.py       (F1/F2/FDA augmentation)
│   ├── _iter_compute_mattes.py
│   ├── _iter_extract_openimages.py
│   └── _iter_setup_openimages.sh
├── reference/
│   ├── baseline_compare.py   (compare_blazepose_vs_ours.py, reused)
│   └── baseline_v4_vs_lite.mp4
├── model/                    (TBD: PyTorch BlazePose re-impl)
│   ├── model.py
│   ├── port_weights.py
│   └── export.py
├── data/                     (TBD: Ego-Exo4D + teacher caching)
│   ├── egoexo_extract.py
│   └── teacher_cache.py
└── train/                    (TBD: distillation training loop)
    ├── train.py
    ├── losses.py
    └── triangulation.py
```

## 12. Open questions / decisions still to make

- **Hand teacher**: Hand Landmarker requires per-wrist ROI cropping at
  inference time. Decide whether we run it once per training sample
  (cache to disk) or live in the dataloader. Disk cache is the obvious
  choice if storage allows (~6 KB / sample).
- **Heavy teacher quantisation**: the Apache-2.0 Heavy `.tflite` is
  fp32. Running it inside the dataloader on CPU is the bottleneck.
  Pre-cache to disk (one-shot pass over Ego-Exo4D); rerun only when
  augmentation changes shouldn't matter because augmentation runs
  *after* the teacher (we apply the same geometric transform to both
  the teacher's output and the student's input).
- **Triangulation weighting**: how to weight `L_tri` per-frame when only
  some KPs are visible in 2+ views. Default plan: per-KP availability
  mask × visibility-product across views; revisit after the first 1k
  samples of debug rendering.

## References

- Bazarevsky et al., "BlazePose: On-device Real-time Body Pose Tracking",
  CVPR-W 2020.
- Bazarevsky & Grishchenko, MediaPipe Pose model card (TF Hub).
- Grishchenko et al., MediaPipe Holistic / Hand Landmarker model cards.
- Grauman et al., "Ego-Exo4D: Understanding Skilled Human Activity from
  First- and Third-Person Perspectives", CVPR 2024.
- Lugaresi et al., "MediaPipe: A Framework for Building Perception
  Pipelines", arXiv 2019 (graph semantics).
- Google AI Edge Torch (PyTorch → TFLite converter), Apache-2.0.
