# Training plan — BlazePose Lite/Full v2 students

End-to-end recipe for continuing-pre-training the Apache-2.0 BlazePose Lite
and Full checkpoints into v2 students that beat their baselines on the
SOTA Ego-Exo4D exo benchmark while preserving close-frontal AR
performance. Drop-in `.tflite` compatibility with the existing MediaPipe
pose graph is required.

**Headline target to beat (from `benchmark/results/RESULTS.md`):**
- Full v1 PA-MPJPE @ ≤ 4 m: **86.0 mm**, intersection-only manual-only N=585
- Heavy v1 PA-MPJPE @ ≤ 4 m: 93.2 mm (the ceiling we want to *exceed*)
- Wilcoxon-significant gap that the v2 student must reproduce with
  the same student model size as v1.

This document is intended to be executable by another engineer (or by me
on Vast.ai) without further questions. It links to existing scripts under
`benchmark/` for byte-stable evaluation.

---

## 0. Hard constraints (re-read before every commit)

1. **Commercial-clean only.** Apache-2.0 / CC-BY / CC0 / public domain.
   Already audited in [`benchmark/results/LICENCE_AUDIT.md`](benchmark/results/LICENCE_AUDIT.md).
   No SMPL / AMASS / BEDLAM / AGORA / Sapiens / COCO-WholeBody / Halpe.
2. **Don't use `zmurez/MediaPipePyTorch`.** Re-implement from the
   `.task` flatbuffers ourselves. Walker template in §2 is original work.
3. **Held-out**: every `take_uid` in `benchmark/frames_manifest.json`
   AND in `benchmark/subset.json`. Cross-check at training time, fail loudly
   if intersection nonempty.
4. **Output**: `pose_landmark_lite_v2.tflite` and `pose_landmark_full_v2.tflite`
   with **byte-identical I/O signatures** to v1 so the existing pose graph
   runs unmodified.

---

## 1. Data inventory (all commercial-clean, all already on the workstation
   except where noted)

| source | size | location | role |
|---|---|---|---|
| **Synth corpus** (Blender/MakeHuman, 17-COCO labels, 256×192) | ~40 GB compressed | HuggingFace `nestorvfx/3DBodyTrackingDatabase` (10 batches) | hard supervision on 17 BP-mappable joints; small (=square pad) preprocess to 256² |
| Synth iter (~3,584 frames, latest revision) | 259 MB | `3D-Body-Tracking-Approach/dataset/output/synth_iter/` | sanity / smoke set |
| **Ego-Exo4D train split** (~3,072 take exo videos at 448p) | ~300-500 GB | re-pull on Vast via `egoexo` CLI; never copy to home machine | hard 3D supervision on 17 COCO joints in world-frame metres + multi-view consistency |
| Ego-Exo4D body GT (`annotations/ego_pose/train/body/...`) | ~1-2 GB | re-pull on Vast | per-frame 17-joint world coords + 2-D placement flags |
| Ego-Exo4D camera_pose (`annotations/ego_pose/train/camera_pose/...`) | ~1-3 GB | re-pull on Vast | static [K\|Rt\|dist] per exo cam |
| **Open Images V7 augmentation corpus** (occluder cutouts + bg crops) | ~270 MB | already at `BlazePose tune/assets/sim2real_refs/` | F1 occluder paste + F2 bg composite during training (not in val) |
| Poly Haven HDRIs (lighting for synth re-render) | ~5 GB | `3D-Body-Tracking-Approach/dataset/assets/hdris/` | optional re-render at 256² |
| 100STYLE / AIST++ / CMU mocap (BVH) | ~5 GB | `3D-Body-Tracking-Approach/dataset/assets/{bvh,bvh_100style,aist_plusplus}/` | optional re-render |
| **MediaPipe `.task` weights** (`pose_landmarker_{lite,full,heavy}.task`) | 6/9/30 MB | `BlazePose tune/assets/` | initialization + Heavy as KD teacher |
| **MediaPipe Hand Landmarker `.task`** | TBD download | TBD | KD teacher for BP indices 17-22 |
| **MediaPipe Face Mesh `.task`** | TBD download | TBD | KD teacher for BP indices 1, 3, 4, 6, 9, 10 |

Hand and Face teacher weights are pulled at setup time (Apache-2.0 CDN URLs).

---

## 2. Architecture & weight-porting (Q1-Q3)

### 2.1 Reading a `.task` flatbuffer

A `.task` is a **ZIP archive**, not a raw flatbuffer. First step:

```python
import zipfile
with zipfile.ZipFile("pose_landmarker_heavy.task") as z:
    z.extractall("heavy_unpacked/")
# Yields: pose_landmarks_detector.tflite, pose_detector.tflite, metadata
```

Read the `.tflite` flatbuffer with the `tflite` PyPI package (pure-schema,
no TF dependency). Parse opcode list and tensor table:

```python
import tflite
buf = open("pose_landmarks_detector.tflite", "rb").read()
m = tflite.Model.GetRootAsModel(buf, 0)
sub = m.Subgraphs(0)
# Topological iteration:
for i in range(sub.OperatorsLength()):
    op   = sub.Operators(i)
    code = m.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    ins  = [sub.Tensors(op.Inputs(j))  for j in range(op.InputsLength())]
    outs = [sub.Tensors(op.Outputs(j)) for j in range(op.OutputsLength())]
    # ... map to PyTorch
```

Pull a constant weight: `m.Buffers(tensor.Buffer()).DataAsNumpy()` then
reshape to `tensor.ShapeAsNumpy()`.

### 2.2 Output tensors (verify per `.task`; current 0.10 release)

| name | shape | meaning |
|---|---|---|
| `Identity`   | `1 × 195`        | 39 landmarks × 5 (x, y, z, visibility, presence). 33 body + 6 auxiliary. |
| `Identity_1` | `1 × 1`          | pose-presence sigmoid |
| `Identity_2` | `1 × 256×256×1`  | foreground segmentation logits |
| `Identity_3` | `1 × 64×64×39`   | optional heatmap branch |
| `Identity_4` | `1 × 117`        | world landmarks (39 × 3 metric metres, body-axis frame, mid-hip origin) |

Output ordering is consumed **by index**, not name, by
`mediapipe/modules/pose_landmark/pose_landmark_model_loader.pbtxt`. Match
`forward()` return-tuple order to v1's signature exactly.

### 2.3 Op-by-op walker (PyTorch instantiation)

```python
class TfliteToTorch:
    OP_HANDLERS = {
        "CONV_2D":           build_conv,            # weight perm: (0,3,1,2)
        "DEPTHWISE_CONV_2D": build_dwconv,          # weight perm: (3,0,1,2), groups=C
        "TRANSPOSE_CONV":    build_transposed,      # (3,0,1,2)
        "ADD":               ResidualAdd,
        "RELU6":             nn.ReLU6,
        "MAX_POOL_2D":       nn.MaxPool2d,
        "MEAN":              nn.AdaptiveAvgPool2d,
        "RESIZE_BILINEAR":   build_resize,
        "FULLY_CONNECTED":   build_fc,
        "RESHAPE":           build_reshape,
        "CONCAT":            build_cat,
        # Unsupported -> raise: indicates a custom op we must hand-implement.
    }
```

**Critical layout perms (the load-bearing detail):**

| TFLite op | TFLite weight layout | PyTorch | `.permute(...)` |
|---|---|---|---|
| `CONV_2D` | `(O, kH, kW, I)` | `(O, I, kH, kW)` | `(0, 3, 1, 2)` |
| `DEPTHWISE_CONV_2D` | `(1, kH, kW, C·M)` | `(C·M, 1, kH, kW)` | `(3, 0, 1, 2)` (groups=C) |
| `TRANSPOSE_CONV` | `(O, kH, kW, I)` | `(I, O, kH, kW)` | `(3, 0, 1, 2)` |
| `FULLY_CONNECTED` | `(out, in)` | `(out, in)` | identity |

Activations: TFLite is NHWC, PyTorch NCHW. Insert one `.permute(0,3,1,2)`
at model entry, one `.permute(0,2,3,1)` per NHWC-shaped output head.
Don't transpose at every op.

**SAME padding**: TFLite SAME is asymmetric for even kernels. Use
`Conv2d(padding=0)` + explicit `nn.ZeroPad2d((pl, pr, pt, pb))` where
`pad_total = max(k - s, 0)`, `pl = pad_total // 2`, `pr = pad_total - pl`.
For 3×3 stride-1 the symmetric `padding=1` is byte-equivalent — skip the ZeroPad.

**BlazeBlock** (paper §2.5): `DepthwiseConv2D 3×3 → Conv2D 1×1 → Add(skip)`,
ReLU6 after the projection. "Double BlazeBlock" appends a second dw+pw stack.
Full / Heavy use 5×5 depthwise in late stages.

### 2.4 Byte-equivalence smoke test

```python
import tflite_runtime.interpreter as tfi
import torch, numpy as np

roi = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)

# Reference: run the landmark tflite directly (NOT through PoseLandmarker.detect()
# which adds detector + ROI + rotation preprocessing we want to bypass).
it = tfi.Interpreter("heavy_unpacked/pose_landmarks_detector.tflite")
it.allocate_tensors()
it.set_tensor(it.get_input_details()[0]['index'],
              (roi/255.).astype(np.float32)[None])
it.invoke()
ref = {d['name']: it.get_tensor(d['index']) for d in it.get_output_details()}

# Ported PyTorch:
torch_model.eval()
x = torch.from_numpy((roi/255.).astype(np.float32))[None].permute(0, 3, 1, 2)
with torch.no_grad():
    out = torch_model(x)

# Tolerances: 1e-4 abs on Lite, 1e-3 on Heavy (compounded fp32 rounding).
for k, v in ref.items():
    np.testing.assert_allclose(out[k].cpu().numpy(), v, atol=1e-4, rtol=1e-3)
```

If this test passes, the port is byte-faithful and we can begin training
without breaking the v1 numerical contract.

### 2.5 PyTorch → TFLite via AI Edge Torch (now `litert-torch`)

`https://github.com/google-ai-edge/litert-torch` (Apache-2.0, **Linux-only — WSL2 on Windows**).

```python
import litert_torch
edge = litert_torch.convert(model.eval(), (torch.randn(1, 3, 256, 256),))
edge.export("pose_landmarks_pt.tflite")
```

The `.task` consumer reads outputs **by index, not name** — match the
return-tuple order. Then repackage as `.task`: replace
`pose_landmarks_detector.tflite` inside the existing `.task` ZIP, keep
`pose_detector.tflite` and `metadata` untouched.

### 2.6 Effort (honest)

- Lite port: **2-3 days**.
- Full port: **4-5 days**.
- Heavy port: **1-2 weeks** (segmentation decoder + heatmap head).
- AI Edge Torch roundtrip + `.task` repackage: **3-5 days**.

**For the v2 students we only need to PORT and TRAIN Lite + Full.** Heavy
is used only as a teacher (run via `mediapipe-tasks`, no port needed).

### 2.7 Fallback if `litert-torch` can't roundtrip BlazeBlock

Manual reverse-port: walk the trained `state_dict`, instantiate a fresh
TFLite flatbuffer from the original `.task`'s op graph, copy weights back
into the `Buffer` table with inverse permutations from §2.3. The op graph
is unchanged so the pose graph keeps working and tensor names are preserved
verbatim. Same byte-level surgery as the import walker, run backwards.

---

## 3. Multi-teacher distillation (Q4-Q5)

### 3.1 Per-joint supervision-source matrix (33 BP joints)

Rule of precedence: **Hard GT > most-specialized teacher > Heavy body teacher**.
When Ego-Exo4D provides hard GT for joint *j*, set teacher mask = 0 for
that joint+sample (Beyer et al. 2022 — "patient and consistent" KD: hard
labels take precedence).

| BP idx | Joint | Hard (Ego-Exo) | Hard (synth) | Body teacher (Heavy) | Hand teacher | Face teacher |
|---|---|---|---|---|---|---|
| 0 | nose | ✅ | ✅ | ✅ | — | — |
| 1, 4 | inner eyes | — | ✅ | ✅ | — | **primary** |
| 2, 5 | eye L/R | (partial) | ✅ | ✅ | — | ✅ |
| 3, 6 | outer eyes | — | ✅ | ✅ | — | **primary** |
| 7, 8 | ears | (partial) | ✅ | ✅ | — | ✅ |
| 9, 10 | mouth L/R | — | ✅ | ✅ | — | **primary** |
| 11, 12 | shoulders | ✅ | ✅ | ✅ | — | — |
| 13, 14 | elbows | ✅ | ✅ | ✅ | — | — |
| 15, 16 | wrists | ✅ | ✅ | ✅ | — | — |
| 17, 18 | pinky tips | — | ✅ | ✅ | **primary** | — |
| 19, 20 | index tips | — | ✅ | ✅ | **primary** | — |
| 21, 22 | thumb tips | — | ✅ | ✅ | **primary** | — |
| 23, 24 | hips | ✅ | ✅ | ✅ | — | — |
| 25, 26 | knees | ✅ | ✅ | ✅ | — | — |
| 27, 28 | ankles | ✅ | ✅ | ✅ | — | — |
| 29-32 | heels / foot-index | — | ✅ | ✅ | — | — |

The synth corpus uses 17 COCO joints — for it, only the 13 mappable
indices (nose, eye L/R, ear L/R, shoulder/elbow/wrist L/R, hip/knee/ankle L/R)
get hard supervision; the other 16 BP indices fall back to teachers.

### 3.2 Teacher-output caching (essential for tractable epoch wall-clock)

Running BlazePose Heavy + Hand Landmarker + Face Mesh **inside the
dataloader is too slow**. Cache once before training:

```bash
python prep/cache_teacher_outputs.py \
    --frames-list frames_train.jsonl \
    --teachers heavy hand face \
    --out /data/teacher_cache/  # ~6 KB per (frame, teacher)
```

Store as `numpy.savez_compressed` or HDF5 keyed by frame_id. Total cache:
~750k frames × 3 teachers × 6 KB ≈ **15 GB**. Fits on instance disk.

Cache invalidation rule: only re-run if augmentation pipeline changes
*before* the teacher inference; we apply augmentations *after* teachers
(teachers are run on the un-augmented frame; geometric augmentation
post-warps both image and teacher labels by the same transform).

### 3.3 Hand- and Face-teacher coordinate alignment

Hand Landmarker outputs in its own wrist-relative frame; Face Mesh in its
own canonical-face frame. To use them as soft targets in BP's mid-hip
body-axis frame:

1. Run Heavy teacher on the same frame → BP world17 in body-axis frame.
2. For Hand Landmarker: extract the wrist 3-D point from Heavy. Use it
   to rigidly translate hand 3-D into body-axis (rotation aligned via
   shoulder–elbow–wrist chain from Heavy).
3. For Face Mesh: align mesh canonical face frame to Heavy's nose+eye
   centers (Procrustes 3-point alignment).

This adds ~5 ms/frame at cache time; verify on a static frame before
running on full corpus.

---

## 4. Loss formulation (Q7)

```
L_total =  λ_hard   · L_hard          # Smooth-L1 (β = 0.05 m) on KP
        +  λ_kd_b   · L_kd_body       # Smooth-L1 (β = 0.02) on KP from Heavy
        +  λ_kd_h   · L_kd_hand       # Smooth-L1 (β = 0.02) on indices 17-22
        +  λ_kd_f   · L_kd_face       # Smooth-L1 (β = 0.02) on indices 1,3,4,6,9,10
        +  λ_anchor · L_anchor_v1     # KL/L1 vs frozen v1 student (anti-drift)
        +  λ_mv     · L_multiview     # Multi-view reprojection on Ego-Exo4D
        +  λ_vis    · L_vis_BCE       # Visibility BCE distillation
```

Suggested weights (Hinton-style, validated by KD literature):
- `λ_hard = 1.0, λ_kd_b = 0.5, λ_kd_h = 0.5, λ_kd_f = 0.3,
   λ_anchor = 0.1, λ_mv = 0.2, λ_vis = 0.1`

KD weights are **halved during epoch 0** (warmup) and ramp to full by
epoch 3 — prevents early collapse onto teacher predictions while the
backbone is still adapting.

**No temperature** for regression heads. Temperature only matters for
softmax KD (FitNet/Hinton); BP's regression head doesn't have it.

### 4.1 Multi-view consistency

For Ego-Exo4D frames where ≥ 2 exo cams see the same person:

```
L_multiview = Σ_cams Σ_j  vis_j · L1(π_cam(p_3d_j), GT_2d_j)
```

`π_cam` projects predicted 3-D into the cam via static `[K|Rt|dist]`.
This is what closes the gap on self-occluded joints that single-view
2-D-only training never resolves. (Iskakov et al. 2019 / Kocabas et al. 2019.)

---

## 5. Anti-regression (Q8)

**Risk**: training on Ego-Exo4D wide-angle exo data shifts the student
away from BP's original close-frontal AR distribution → regresses on the
yoga / fitness / smartphone-frontal use case.

Three anti-regression mechanisms (Li & Hoiem "Learning without Forgetting" lineage):

1. **Anchor distillation against the frozen v1 student** (`L_anchor_v1`,
   weight 0.1). Initial v1 weights are kept in a frozen copy; for every
   training batch we add a Smooth-L1 between current and v1 outputs on a
   subset of close-frontal samples (replay buffer, see #2). This is
   self-distillation for the purpose of regularization — the student
   doesn't over-write what v1 already knew.

2. **In-distribution replay**: 30-50 % of every batch must be close-frontal
   synthetic frames (which match BP's training distribution: tight
   single-person crops at 2-4 m). The other 50-70 % is Ego-Exo4D wide-angle
   exo. Ratio sweep: start at 50/50, validate on a held-out close-frontal
   set every 500 steps.

3. **Layer-wise LR decay (LLRD) + EMA-of-weights eval**: 0.65× LR
   multiplier per stage going down the backbone. Earliest layers move
   slowest. EMA decay 0.9999, evaluated copy used for sanity-check
   metrics so a bad batch doesn't reach gating.

---

## 6. Training schedule & hyperparameters

| param | value | rationale |
|---|---|---|
| optimizer | AdamW | standard for CNN fine-tune |
| LR | **5e-5** peak | safe upper bound for continued pretrain; 1e-4 is the cliff |
| LR schedule | 1k-step linear warmup → cosine to 1e-6 over 10 epochs | Howard & Ruder ULMFiT-style |
| weight decay | 1e-4 (excl. norms / biases) | |
| EMA decay | 0.9999 | standard for 10-epoch schedule |
| batch size (A100 80 GB) | **128** at 256² with bf16 + grad-checkpointing | BlazeBlock-stable |
| mixed precision | bf16 (NOT fp16 — sigmoid + BCE underflow) | A100/A6000/4090 all support |
| epochs | **10** | Beyer "patient KD" + drift-prevention |
| frozen layers | segmentation decoder (no GT masks); BN in eval mode | Goyal et al. 2017 |
| backbone LR multiplier | 0.1× of head LR (LLRD with γ=0.65) | preserves early-layer features |
| seed | 42 |

Dataset mix per epoch (~60-80k samples):
- 30 % synth (close-frontal, hard 17-COCO)
- 50 % Ego-Exo4D train exo (hard 17-COCO + multi-view)
- 20 % "in-distribution replay" — synth frames at close-frontal angles
  with `L_anchor_v1` engaged

---

## 7. Hold-out enforcement (Q6)

At dataloader init, build a `forbidden_uids` set:

```python
forbidden = set()
for path in ["benchmark/frames_manifest.json", "benchmark/subset.json"]:
    forbidden |= set(json.load(open(path)).get("take_uids",
                     list(json.load(open(path)).keys())))

if any(uid in forbidden for uid in train_uids):
    raise RuntimeError("VAL TAKE UIDS LEAKED INTO TRAIN — abort")
```

This runs every train job startup. Failure to assert empty → exit code 99.

---

## 8. Vast.ai operations (Q10-Q12)

### 8.1 Recommended SKU

**1× A100 SXM4 80 GB on-demand, ~$1.50-1.90/hr** (verify Vast console at run time).

Rationale:
- Single GPU eliminates DDP overhead — at ~75k steps/student this matters.
- 80 GB fits batch 128 + KD-cache per sample with headroom.
- ~14 hr for both students → **~$25 budget for the campaign**.
- 2× A6000 ($1.40/hr) only saves money if A100 not available < $2.
- Avoid 4090 (24 GB OOMs at batch 192 with KD); avoid H100 ($2.50/hr is overkill).

### 8.2 `vast_setup.sh` (idempotent)

```bash
#!/usr/bin/env bash
set -euo pipefail
[ -f /workspace/.setup_done ] && exit 0
apt-get update && apt-get install -y ffmpeg awscli rsync tmux
pip install --upgrade pip
pip install torch==2.4.* torchvision \
            mediapipe==0.10.* \
            litert-torch \
            opencv-python-headless \
            tensorboard pyyaml tqdm scipy h5py \
            ego4d
mkdir -p /workspace/{runs,ckpts,logs} /data/{synth,egoexo,oiv7,teacher_cache}
aws configure set aws_access_key_id     "$AWS_KEY"
aws configure set aws_secret_access_key "$AWS_SECRET"
touch /workspace/.setup_done
```

### 8.3 `vast_train.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"

# 1. Stage data (instance disk = fast, ephemeral). Re-pull on every boot.
[ -d /data/egoexo/takes ] || egoexo -o /data/egoexo \
    --parts annotations downscaled_takes/448 \
    --benchmarks egopose --splits train --views exo
[ -f /data/synth/done ] || rsync -a synth-host:/synth/ /data/synth/
[ -d /data/oiv7 ] || rsync -a oiv7-host:/oiv7/ /data/oiv7/

# 2. Pre-cache teacher outputs once per session (reused across both students)
[ -f /data/teacher_cache/done ] || python prep/cache_teacher_outputs.py \
    --teachers heavy hand face \
    --out /data/teacher_cache/

# 3. Hold-out enforcement
python prep/check_holdout.py \
    --train-root /data/egoexo \
    --forbidden benchmark/frames_manifest.json benchmark/subset.json \
  || { echo "HOLD-OUT VIOLATED — abort"; exit 99; }

# 4. Resume logic
LATEST=$(ls -t /workspace/ckpts/*.pt 2>/dev/null | head -1 || echo "")
RESUME="${LATEST:+--resume $LATEST}"

if [ "$MODE" = "smoke" ]; then
  python train.py --variant lite --epochs 1 --max-samples 5000 --bs 96 $RESUME
else
  python train.py --variant lite --epochs 10 --bs 128 --bf16 \
                  --ckpt-every 600s $RESUME
  python train.py --variant full --epochs 10 --bs 96  --bf16 \
                  --ckpt-every 600s
  python export_tflite.py --variant lite --out /workspace/lite_v2.task
  python export_tflite.py --variant full --out /workspace/full_v2.task

  # Final sanity: run the v2 students through the SOTA benchmark
  python ../benchmark/run_eval.py --variant student_v2_lite --task /workspace/lite_v2.task
  python ../benchmark/run_eval.py --variant student_v2_full --task /workspace/full_v2.task
  python ../benchmark/analyze.py  --manual-only \
         --variants lite full heavy student_v2_lite student_v2_full \
         --out /workspace/analysis_student_v2.json
fi
```

### 8.4 Storage strategy

- `/data/*` on instance NVMe (ephemeral): ~400 GB; **re-download on every boot**
  (cheaper than $25/mo persistent storage).
- `/workspace/*` (persistent, slow): only checkpoints + TB logs + final
  `.task` + `analysis_student_v2.json`. ~5 GB.
- `vast scp` to pull artefacts back to local at job end.

### 8.5 On-demand vs spot

**On-demand.** 14 hr is too short for spot to be worth the preemption risk
on first deployment. Once auto-resume is bullet-proof, switch to spot for ~30 % savings.

### 8.6 Pricing

| line | $ |
|---|---|
| smoke run (~1 hr A100) | ~$2 |
| Lite training (~7 hr A100) | ~$12 |
| Full training (~7 hr A100) | ~$12 |
| Persistent storage (1 month) | ~$0.30 |
| Buffer for one re-run | ~$10 |
| **Cap** | **~$50** |

Abort if smoke alone exceeds $5.

---

## 9. Smoke testing (Q13) — ALL local before Vast

| step | what | success criterion |
|---|---|---|
| S1 | port_weights.py byte-equivalence on Lite | max abs diff < 1e-4 vs `.task` |
| S2 | port_weights.py byte-equivalence on Full | max abs diff < 1e-3 |
| S3 | dataloader yields a batch from synth + Ego-Exo4D val *not in held-out manifest* | shapes correct, no leakage |
| S4 | one forward + backward + opt-step on CPU | no NaN, loss finite |
| S5 | teacher-cache loader matches in-line teacher output | difference < 1e-5 |
| S6 | hold-out check fires on poisoned input | exits 99 |
| S7 | export pre-trained Lite back to `.task` via litert-torch | round-trip equivalence < 1e-3 |
| S8 | run S7's Lite `.task` through `benchmark/run_eval.py` | numbers match v1 to ±1 mm |

**Only after all 8 pass** do we provision a Vast instance. Smoke S1-S6
take a few hours total on the workstation.

---

## 10. Validation gating (Q9)

Mid-training (every 500 steps), the gating dashboard shows:

| metric | threshold for green |
|---|---|
| PCK@0.05 on close-frontal yoga held-out clip vs v1 | within −0.5 pp of v1 |
| MPJPE (mm) on Ego-Exo4D held-out (NOT the SOTA benchmark!) | monotone decreasing |
| Visibility AUC on synthetic | ≥ 0.85 |
| `L_anchor_v1` magnitude | < 2× initial value |
| BP wrist (idx 15) ↔ hand-pinky (idx 17) 3-D residual | ~10 cm baseline preserved |
| Per-cam reprojection error stratified by viewpoint | no spikes on side views |

If anything yellow for > 2k steps → stop and lower LR.

**Final acceptance** — v2 student passes if **all** of:
1. Beats Full v1 by ≥ 5 mm PA-MPJPE on the SOTA benchmark.
2. Within 5 mm of Heavy on close-frontal yoga clip.
3. Round-trip exported `.task` byte-equivalent to PyTorch (1e-3).
4. PA-MPJPE 95 % CI excludes Full v1's 86 mm at p < 0.05.

---

## 11. Local vs Vast.ai split

### 11.1 LOCAL FIRST (do these before booking any GPU)

1. Build `model/blazepose_arch.py` + `model/port_weights.py` (Lite + Full only).
2. Pass byte-equivalence tests S1, S2.
3. Build `prep/cache_teacher_outputs.py` (run on the small synth iter sanity set).
4. Write the dataloader, loss, training loop.
5. Smoke S3-S6 on CPU.
6. Build `model/export_tflite.py` and pass S7, S8.

This is **2-3 weeks of focused work** (per the porting effort estimate)
before there's any value in renting a GPU. A premature Vast booking will
waste $.

### 11.2 VAST.AI ONLY (impossible to do locally)

These genuinely need Vast.ai because of dataset size + compute time:

1. **Pre-caching teachers on Ego-Exo4D train (~600k frames)**:
   - Inputs: ~300-500 GB Ego-Exo4D videos.
   - Output: ~15 GB teacher cache.
   - Vast network pull from S3 (multi-Gb/s typical) is faster than home upload.
   - Single A100 hour, ~$2.
2. **Full 10-epoch training × 2 students**:
   - ~14 hr GPU time per the budget.
   - Instance disk handles 400 GB live dataset.
3. **Final round-trip + benchmark scoring**:
   - Run `benchmark/run_eval.py` on the v2 `.task` outputs.
   - This ALSO can be done locally after retrieving the `.task` files,
     but doing it on Vast keeps the validation loop tight.

### 11.3 Risk register (top 5)

1. **Heavy port has a custom op the walker doesn't recognize.** Fallback:
   keep Heavy as TFLite-only teacher (we never train Heavy anyway), only
   port Lite + Full. ✅ already in plan.
2. **`litert-torch` loses BlazeBlock structure on export.** Fallback:
   manual reverse-port (§2.7) — walk state_dict, copy weights back into
   v1 `.task` op-graph template. Ugly but proven. **Mitigation: prove S7
   on Lite before training Full.**
3. **Teacher coords don't align across hand/face/body frames.** Fallback:
   fall back to body-only KD; skip hand+face teachers if alignment turns
   out to be > 5 cm noisy. Document, don't hide.
4. **Anti-regression fails — student regresses on close-frontal yoga.**
   Fallback: increase replay-buffer ratio to 70 % synth, halve
   λ_kd_b, and bump λ_anchor to 0.3.
5. **Vast instance preempted mid-train.** Fallback: 600 s checkpoint
   interval means ≤ 10 min lost; auto-resume from latest is built-in.
   Don't use spot until S7+S8 pass twice cleanly.

---

## 12. Success scoreboard

| target | beats Full v1? | beats Heavy v1? | preserves AR? | byte-stable export? |
|---|---|---|---|---|
| Lite v2 | ≥ 5 mm under 86 mm PA-MPJPE | yes by ≥ 5 mm | within 5 mm of Lite v1 on yoga | ✅ |
| Full v2 | ≥ 5 mm under 86 mm | ≥ 7 mm under 93 mm | within 5 mm of Full v1 on yoga | ✅ |

If both rows pass on the SOTA benchmark with Wilcoxon p < 0.01 (Holm-corrected
across the two simultaneous contrasts), we ship.

---

## 13. References

- Bazarevsky et al. 2020. BlazePose. https://arxiv.org/abs/2006.10204
- BlazeFace (origin of BlazeBlock): https://arxiv.org/abs/1907.05047
- Hinton et al. 2015. KD: https://arxiv.org/abs/1503.02531
- Beyer et al. 2022. Patient KD: https://arxiv.org/abs/2106.05237
- Li & Hoiem 2017. LwF: https://arxiv.org/abs/1606.09282
- Touvron et al. 2021. DeiT: https://arxiv.org/abs/2012.12877
- Iskakov et al. 2019. Learnable Triangulation: https://arxiv.org/abs/1905.05754
- Kocabas et al. 2019. EpipolarPose: https://arxiv.org/abs/1903.02330
- Liu et al. 2022. ConvNeXt fine-tune: https://arxiv.org/abs/2201.03545
- Grishchenko et al. 2022. GHUM Holistic: https://google.github.io/mediapipe/solutions/holistic
- Pose Landmarker docs: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- AI Edge / LiteRT-Torch: https://github.com/google-ai-edge/litert-torch
- MediaPipe pose graph: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_model_loader.pbtxt
