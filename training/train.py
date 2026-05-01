"""Full multi-epoch distillation trainer for BlazePose v2 students.

Wires together: hold-out enforcement, student + anchor + Heavy teacher,
optional cached Hand/Face teachers, multi-source mixed dataloader,
V2DistillationLoss, AdamW + cosine LR + warmup, EMA decay, gradient
clipping, bf16 AMP, periodic checkpoint with auto-resume, TensorBoard.

Usage on Vast:
    python training/train.py \
        --variant lite \
        --synth-root /data/synth \
        --egoexo-root /data/egoexo \
        --teacher-cache /data/teacher_cache \
        --out-root /workspace/ckpts \
        --epochs 10 --batch-size 96 --lr 5e-5 --bf16

Reads synth's labels.jsonl, Ego-Exo4D's manifest_train.jsonl, and (optional)
pre-cached teacher .npz directory.  Refuses to start if any train uid is
in the held-out manifest/subset.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model when wrapped in DDP, else `m` itself."""
    return m.module if hasattr(m, "module") else m

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE))

from port import load_task                                      # noqa: E402
from dataset import SynthDataset, EgoExoTrainDataset, MixedDataset  # noqa: E402
from losses import V2DistillationLoss                            # noqa: E402
from holdout import assert_no_leakage                            # noqa: E402

ASSETS = HERE.parent / "assets"


# ─── EMA ──────────────────────────────────────────────────────────────────

class EMA:
    """Exponential-moving-average shadow weights."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            v.copy_(self.shadow[k])


# ─── LR scheduler ─────────────────────────────────────────────────────────

def warmup_cosine_lr(step: int, total_steps: int, warmup: int,
                     base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


# ─── Validation ───────────────────────────────────────────────────────────

@torch.no_grad()
def quick_val(student, anchor, val_loader, device, dtype) -> dict:
    """Tiny val loop: mean drift vs anchor (anti-regression sanity)."""
    drift = []
    n = 0
    for batch in val_loader:
        if n >= 32:
            break
        img = batch["image"].to(device).to(dtype)
        s_out = student(img)
        a_out = anchor(img)
        s_w = s_out["Identity_4"].view(-1, 39, 3)[:, :33]
        a_w = a_out["Identity_4"].view(-1, 39, 3)[:, :33]
        drift.append(F.l1_loss(s_w, a_w).item())
        n += img.shape[0]
    return {"val_anchor_drift_m": float(sum(drift) / max(len(drift), 1))}


@torch.no_grad()
def benchmark_eval(student, device, dtype, max_takes: int = 26,
                   max_frames_per_cam: int = 3) -> dict:
    """Run the SOTA benchmark on a stable subset of frames.  Computes
    PA-MPJPE-vs-GT — directly the metric we want to track training on
    (lower is better; v1 baselines: Lite 99.4mm / Full 96.0mm / Heavy 101.6mm
    on the full 1,292-frame benchmark, see benchmark/results/RESULTS.md).

    Default max_takes=26 = the entire 26-take benchmark; max_frames_per_cam=3
    keeps per-epoch eval to ~300 frames (~100 sec on a single 5070 Ti).
    Note: this runs the *PyTorch port*, not the exported .task — for the
    canonical apples-to-apples number against v1, use
    benchmark/run_eval.py + analyze.py against the .task files (stage 7).

    Loads frames from `benchmark/frames/` and `benchmark/frames_manifest.json`.
    Returns NaN-filled dict if the benchmark isn't on disk yet — non-fatal."""
    import sys, json, numpy as np, cv2
    sys.path.insert(0, str(HERE.parent / "benchmark"))
    BENCH = HERE.parent / "benchmark"
    manifest_p = BENCH / "frames_manifest.json"
    frames_root = BENCH / "frames"
    anno_root = BENCH / "raw" / "annotations"
    if not (manifest_p.exists() and frames_root.exists() and anno_root.exists()):
        return {"bench_pa_mpjpe_mm": float("nan"),
                "bench_n_frames": 0,
                "bench_note": "no benchmark frames on disk"}
    try:
        from lib.ego_exo_io import load_body_gt, load_camera_pose, is_exo_cam
        from lib.keypoint_map import gt_to_coco17, BP_INDEX_FOR_COCO, HIP_L, HIP_R
        from lib.projection import world_to_cam
        from lib.metrics import pa_mpjpe_per_frame, root_center
        # Need body-axis transform too
        sys.path.insert(0, str(HERE))
        from coords import build_body_frame, cam_to_body
    except Exception as e:
        return {"bench_pa_mpjpe_mm": float("nan"),
                "bench_n_frames": 0,
                "bench_note": f"import failed: {e}"}
    manifest = json.loads(manifest_p.read_text())
    take_uids = sorted(manifest.keys())[:max_takes]
    pa_errs_mm = []
    for uid in take_uids:
        cp = load_camera_pose(anno_root / "ego_pose/val/camera_pose" / f"{uid}.json")
        gt = load_body_gt(anno_root / "ego_pose/val/body/annotation" / f"{uid}.json")
        for cam_name, fidxs in manifest[uid].items():
            cam = cp["cams"].get(cam_name)
            if cam is None or not is_exo_cam(cam_name):
                continue
            for fi in fidxs[:max_frames_per_cam]:
                jpg = frames_root / uid / cam_name / f"{int(fi):06d}.jpg"
                if not jpg.exists():
                    continue
                entry = gt.get(int(fi))
                if entry is None:
                    continue
                kp_w, present = gt_to_coco17(entry["annotation3D"])
                if not (present[HIP_L] and present[HIP_R]):
                    continue
                img_bgr = cv2.imread(str(jpg))
                # Resize to 256x256 for the student
                if img_bgr.shape[:2] != (256, 256):
                    img_bgr = cv2.resize(img_bgr, (256, 256))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                x = torch.from_numpy(img_rgb.astype(np.float32) / 255.0
                                     ).permute(2, 0, 1)[None].to(device).to(dtype)
                s_out = student(x)
                pred33 = s_out["Identity_4"].view(1, 39, 3)[0, :33].cpu().float().numpy()
                # GT in body-axis: project world→cam→body
                gt_cam = world_to_cam(kp_w, cam["Rt"]).astype(np.float32)
                R, origin = build_body_frame(gt_cam, present.astype(np.float32))
                if R is None:
                    continue
                gt_body = cam_to_body(gt_cam, R, origin)
                # Pad GT to 33 indices using BP_INDEX_FOR_COCO mapping
                gt33 = np.zeros((33, 3), dtype=np.float32)
                mask = np.zeros(33, dtype=bool)
                for coco_idx, bp_idx in enumerate(BP_INDEX_FOR_COCO):
                    if present[coco_idx]:
                        gt33[bp_idx] = gt_body[coco_idx]
                        mask[bp_idx] = True
                if mask.sum() < 4:
                    continue
                # Per-frame PA-MPJPE on body-axis
                P = pred33[None]
                G = gt33[None]
                M = mask[None]
                pa = pa_mpjpe_per_frame(P, G, M)[0]
                if not np.isnan(pa):
                    pa_errs_mm.append(pa * 1000)
    if not pa_errs_mm:
        return {"bench_pa_mpjpe_mm": float("nan"),
                "bench_n_frames": 0,
                "bench_note": "no scorable frames"}
    return {"bench_pa_mpjpe_mm": float(np.mean(pa_errs_mm)),
            "bench_n_frames": len(pa_errs_mm),
            "bench_note": "ok"}


@torch.no_grad()
def per_keypoint_breakdown(student, anchor, val_loader, device, dtype) -> dict:
    """Per-BP-index drift vs frozen v1 (1-of-33 bars to TensorBoard).

    Catches "fingers regress while wrists improve" — the failure mode a
    single PA-MPJPE number won't show.  Also reports visibility AUC so
    we'd see visibility-flag drift before it breaks the smoothing graph.
    """
    import numpy as np
    per_joint_drift = []
    vis_pred = []
    vis_anchor = []
    n = 0
    for batch in val_loader:
        if n >= 64:
            break
        img = batch["image"].to(device).to(dtype)
        s_out = student(img)
        a_out = anchor(img)
        s_w = s_out["Identity_4"].view(-1, 39, 3)[:, :33]
        a_w = a_out["Identity_4"].view(-1, 39, 3)[:, :33]
        per_joint_drift.append((s_w - a_w).abs().mean(-1).cpu().float().numpy())
        s_kp = s_out["Identity"].view(-1, 39, 5)
        a_kp = a_out["Identity"].view(-1, 39, 5)
        vis_pred.append(s_kp[:, :33, 3].cpu().float().numpy())
        vis_anchor.append(a_kp[:, :33, 3].cpu().float().numpy())
        n += img.shape[0]
    if not per_joint_drift:
        return {}
    pj = np.concatenate(per_joint_drift, axis=0).mean(axis=0)   # (33,) metres
    vp = np.concatenate(vis_pred,    axis=0)
    va = np.concatenate(vis_anchor,  axis=0)
    out = {f"per_kp_drift_{i}_mm": float(pj[i] * 1000) for i in range(33)}
    out["per_kp_drift_max_mm"] = float(pj.max() * 1000)
    out["per_kp_drift_mean_mm"] = float(pj.mean() * 1000)
    # Visibility flag agreement: AUC-style — fraction of joints where pred
    # and anchor agree (>0.5 vs <0.5).  Drops if student forgets v1's
    # visibility calibration.
    out["vis_agreement_pct"] = float(((vp > 0.5) == (va > 0.5)).mean() * 100)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",        required=True, choices=["lite", "full"])
    ap.add_argument("--synth-root",     type=Path, required=True)
    ap.add_argument("--egoexo-root",    type=Path, default=None,
                    help="Optional; if missing, train on synth-only.")
    ap.add_argument("--anno-root",      type=Path, default=None,
                    help="Path to annotations dir (defaults to <egoexo-root>/annotations).")
    ap.add_argument("--teacher-cache",  type=Path, default=None,
                    help="Pre-cached teacher .npz dir; if missing, run Heavy live.")
    ap.add_argument("--out-root",       type=Path, default=Path("/workspace/ckpts"))
    ap.add_argument("--runs-root",      type=Path, default=Path("/workspace/runs"))
    ap.add_argument("--epochs",         type=int,   default=20,
                    help="Patient-KD literature (Beyer 2022) recommends 3-10x "
                         "longer than supervised training; 20 is a sane default "
                         "for a 12-hr A100 run on ~750k samples.")
    ap.add_argument("--batch-size",     type=int,   default=96)
    ap.add_argument("--lr",             type=float, default=5e-5)
    ap.add_argument("--weight-decay",   type=float, default=1e-4)
    ap.add_argument("--warmup-steps",   type=int,   default=1000)
    ap.add_argument("--ema-decay",      type=float, default=0.9999)
    ap.add_argument("--grad-clip",      type=float, default=1.0)
    ap.add_argument("--num-workers",    type=int,   default=4)
    ap.add_argument("--ckpt-every-min", type=float, default=10.0)
    ap.add_argument("--synth-ratio",    type=float, default=0.4,
                    help="Fraction of each batch from synth (vs egoexo).")
    ap.add_argument("--bf16",           action="store_true")
    ap.add_argument("--device",         type=str,   default=None)
    ap.add_argument("--limit-synth",    type=int,   default=None)
    ap.add_argument("--limit-egoexo",   type=int,   default=None)
    ap.add_argument("--resume",         type=Path,  default=None)
    ap.add_argument("--no-auto-resume", action="store_true", default=False,
                    help="Disable auto-resume from the latest checkpoint in "
                         "out_root.  Use when changing loss config: an old "
                         "ckpt's optimizer state is fitted to old gradients "
                         "and can crash or undo the new config.")
    ap.add_argument("--seed",           type=int,   default=42)
    # Loss-weight knobs (default to V2DistillationLoss signature defaults).
    # Set non-zero to let the user iterate on the smoke without code edits.
    ap.add_argument("--lam-hard",       type=float, default=0.0,
                    help="Body-axis hard supervision weight.  DEFAULT 0 "
                         "because egoexo's body-frame is defined by "
                         "egoexo-annotated hips/shoulders (14-px reprojection "
                         "noise floor) while L_anchor uses v1's own internal "
                         "estimate — different frames, student can't track "
                         "both.  Rely on L_anchor + L_anchor_img + L_multiview "
                         "(image-frame reprojection) instead.")
    ap.add_argument("--lam-kd-b",       type=float, default=0.5)
    ap.add_argument("--lam-mv",         type=float, default=0.5,
                    help="Multi-view reprojection loss.  Reproject student's "
                         "body-axis pred back to image-frame and compare to "
                         "annotated 2D — frame-agnostic image-space GT.")
    ap.add_argument("--lam-anchor",     type=float, default=1.0,
                    help="Body-axis anchor (Identity_4) weight.  Smoke showed "
                         "0.4 was too weak at effective batch 128 — student "
                         "drifted from v1 and lower-body PA-MPJPE blew up.")
    ap.add_argument("--lam-anchor-img", type=float, default=5.0,
                    help="Image-frame v1 anchor (Identity) weight.  This is "
                         "the load-bearing regularizer because Identity is "
                         "what MediaPipe's downstream pose graph consumes. "
                         "Bumped 2.0→5.0 after smoke 2 showed 2.0 wasn't "
                         "enough to overcome Heavy-KD pulling student away "
                         "from v1.")
    ap.add_argument("--use-heavy-kd",   action="store_true",
                    default=False,
                    help="Distill body-axis from Heavy teacher.  DEFAULT OFF "
                         "because Heavy is *worse* than v1 Full on Ego-Exo4D "
                         "exo views (-5.6 mm; see benchmark/results/RESULTS.md). "
                         "Distilling from Heavy on this benchmark drags the "
                         "student toward Heavy's distribution-shift artefacts. "
                         "Set USE_HEAVY_KD=1 to opt back in.")
    ap.add_argument("--freeze-backbone", action="store_true", default=False,
                    help="Freeze every CONV/DWCONV weight; only train heads "
                         "(the small layers that produce Identity/Identity_4). "
                         "Most conservative possible fine-tune — makes it "
                         "very hard for the student to drift from v1 features. "
                         "Use as a sanity check that the loss is wired right; "
                         "or as the actual training mode if anchor-only KD "
                         "isn't enough.")
    ap.add_argument("--disable-synth-hard", action="store_true", default=False,
                    help="Mask synth's hard labels entirely (set bp33_present "
                         "= 0 for all synth samples).  Synth labels are MPFB2 "
                         "rig joints with 1-4 cm offset from BlazePose visual "
                         "landmarks.  Smoke 4 confirmed L_hard pulled student "
                         "toward MPFB while L_anchor pulled toward visual — "
                         "the fight regressed benchmark.  With this flag, "
                         "synth becomes a pure image-diversity source (anchor "
                         "+ image-frame anchor only); ego-exo provides the "
                         "only L_hard signal (its labels are visual landmarks).")
    # LR scaling rule for DDP.  At effective batch 4×BATCH, the constant 5e-5
    # was an overshoot in our smoke run (anchor loss rose 3.5×).  Sqrt scaling
    # is the gentle middle ground; "none" preserves single-GPU behavior.
    ap.add_argument("--lr-scale-rule",  choices=["none", "sqrt", "linear"],
                    default="sqrt",
                    help="Adjust --lr by f(world_size).  sqrt: lr×sqrt(N); "
                         "linear: lr×N; none: unchanged.")
    ap.add_argument("--variant-lr-scale", type=float, default=None,
                    help="Multiplier on --lr for this variant.  Smoke showed "
                         "Full regressed 5× more than Lite at the same LR; "
                         "set to 0.5 for variant=full to compensate.")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(args.seed)

    # ── Distributed setup (torchrun-launched) ────────────────────────────
    # When LOCAL_RANK is set in env, we're under torchrun → init NCCL,
    # pin to per-rank GPU.  Otherwise fall back to single-GPU mode.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1 and local_rank >= 0
    if is_dist:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        rank = 0
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    is_main = (rank == 0)

    dtype  = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    # ── LR scaling for DDP effective batch + per-variant compensation ──
    # Effective batch = world_size × BATCH.  Sqrt scaling is the standard
    # patient-KD rule (vs linear which is too aggressive for small-LR KD).
    base_lr = args.lr
    if args.lr_scale_rule == "sqrt" and world_size > 1:
        args.lr = base_lr * (world_size ** 0.5)
    elif args.lr_scale_rule == "linear" and world_size > 1:
        args.lr = base_lr * world_size
    if args.variant_lr_scale is not None:
        args.lr = args.lr * args.variant_lr_scale
    if is_main and args.lr != base_lr:
        print(f"[train] LR scaled: {base_lr:.2e} → {args.lr:.2e}  "
              f"(rule={args.lr_scale_rule}, world_size={world_size}, "
              f"variant_scale={args.variant_lr_scale})")

    if is_main:
        args.out_root.mkdir(parents=True, exist_ok=True)
        args.runs_root.mkdir(parents=True, exist_ok=True)
        run_dir = args.runs_root / f"v2_{args.variant}_{int(time.time())}"
        run_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=run_dir)
        print(f"[train] device={device}  dtype={dtype}  variant={args.variant}  "
              f"world_size={world_size}  rank={rank}")
        print(f"[train] run_dir={run_dir}")
    else:
        writer = None

    # ── Models ──────────────────────────────────────────────────────────
    student_task = ASSETS / f"pose_landmarker_{args.variant}.task"
    if is_main:
        print(f"[train] loading student from {student_task}")
    student, _ = load_task(student_task)
    student.train().to(device).to(dtype)

    if is_main:
        print(f"[train] loading frozen anchor (v1 weights)")
    anchor, _ = load_task(student_task)
    for p in anchor.parameters():
        p.requires_grad_(False)
    anchor.eval().to(device).to(dtype)

    # Heavy teacher: only live if we don't have a cache for *every* sample
    teacher = None
    if args.teacher_cache is None or not args.teacher_cache.exists():
        if is_main:
            print(f"[train] loading Heavy teacher live (slow CPU/GPU; "
                  f"prefer pre-cached teachers via prep/cache_teachers.py)")
        teacher, _ = load_task(ASSETS / "pose_landmarker_heavy.task")
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval().to(device).to(dtype)

    # ── Datasets ────────────────────────────────────────────────────────
    # Synth's clean.zip ships both splits.  Train on the train split, and
    # use the val split as a held-out AR-replay tracker (close-frontal /
    # 2-4 m / clean labels — proxy for BP v1's training distribution).
    synth_ds = SynthDataset(
        labels_jsonl=args.synth_root / "labels.jsonl",
        images_root=args.synth_root,
        teacher_cache_dir=args.teacher_cache,
        augment=True,
        limit=args.limit_synth,
        split="train",
        disable_hard=args.disable_synth_hard)
    synth_val_ds = SynthDataset(
        labels_jsonl=args.synth_root / "labels.jsonl",
        images_root=args.synth_root,
        teacher_cache_dir=args.teacher_cache,
        augment=False,           # val: no aug
        limit=512,               # cap val to 512 frames for fast per-epoch eval
        split="val")
    # Fallback for tiny iter-set smoke runs that don't carry a val split:
    # use a small slice of the train side as a degenerate "val" so the
    # validation hooks still produce numbers.
    if len(synth_val_ds) == 0:
        if is_main:
            print("[train] no val records — using first 32 of train as smoke val")
        synth_val_ds = SynthDataset(
            labels_jsonl=args.synth_root / "labels.jsonl",
            images_root=args.synth_root,
            teacher_cache_dir=args.teacher_cache,
            augment=False, limit=32, split="train")

    train_ds: torch.utils.data.Dataset
    if args.egoexo_root and (args.egoexo_root / "manifest_train.jsonl").exists():
        anno_root = args.anno_root or (args.egoexo_root / "annotations")
        ego_ds = EgoExoTrainDataset(
            manifest_jsonl=args.egoexo_root / "manifest_train.jsonl",
            frames_root=args.egoexo_root / "frames",
            anno_root=anno_root,
            teacher_cache_dir=args.teacher_cache,
            augment=True,
            limit=args.limit_egoexo)
        # Hold-out: collect uids from manifest + assert
        manifest_uids = sorted({r["take_uid"] for r in ego_ds.records})
        forbidden_m = HERE.parent / "benchmark" / "frames_manifest.json"
        forbidden_s = HERE.parent / "benchmark" / "subset.json"
        assert_no_leakage(manifest_uids, forbidden_m, forbidden_s)
        train_ds = MixedDataset(synth_ds, ego_ds, synth_ratio=args.synth_ratio)
    else:
        if is_main:
            print(f"[train] no Ego-Exo4D manifest — synth-only training")
        train_ds = synth_ds

    # Validation loader (rank 0 only — runs on the unwrapped module)
    val_loader = DataLoader(synth_val_ds, batch_size=min(args.batch_size, 8),
                            shuffle=False, num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0) if is_main else None

    # Train loader: DistributedSampler shards data across ranks when DDP-launched
    if is_dist:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=args.num_workers,
                                  persistent_workers=args.num_workers > 0,
                                  drop_last=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  persistent_workers=args.num_workers > 0,
                                  drop_last=True)

    # ── Backbone freeze (optional, conservative-fine-tune mode) ─────────
    if args.freeze_backbone:
        n_frozen = n_train = 0
        # The port stores conv/dwconv params under keys "conv_*_w/b" and
        # "dwconv_*_w/b".  Heads (the small final layers producing Identity
        # outputs) get the same naming scheme but are conventionally the LAST
        # few CONV ops.  Simplest robust rule: freeze every param whose name
        # starts with "conv_" or "dwconv_" *except* the last 4 op indices.
        # The model is small enough (~70 conv ops in Lite) that this is a
        # negligible head footprint vs the backbone.
        # Find the highest op index per kind, retain trainable on top-4.
        param_names = list(_unwrap(student).state_dict().keys())
        op_indices = sorted({int(n.split("_")[1]) for n in param_names
                             if n.startswith(("conv_", "dwconv_"))})
        head_threshold = op_indices[-4] if len(op_indices) >= 4 else -1
        for name, p in _unwrap(student).named_parameters():
            if name.startswith(("conv_", "dwconv_")):
                idx = int(name.split("_")[1])
                if idx < head_threshold:
                    p.requires_grad_(False)
                    n_frozen += 1
                else:
                    n_train += 1
            else:
                n_train += 1
        if is_main:
            print(f"[train] freeze_backbone: {n_frozen} params frozen, "
                  f"{n_train} trainable (head threshold op_idx={head_threshold})")

    # ── DDP wrap (after model is on device, before optimizer is built so the
    #              optimizer sees the wrapped params) ────────────────────────
    if is_dist:
        # find_unused_parameters=True: the student computes Identity_2/3 (seg
        # + heatmap) which the loss doesn't consume — those head params don't
        # receive gradients each step, so DDP must allow it.
        student = DDP(student, device_ids=[local_rank],
                      find_unused_parameters=True)

    # ── Optim + sched + EMA ─────────────────────────────────────────────
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # Loss weights from CLI (defaults match what the post-mortem of the
    # smoke regression suggested: stronger anchor on both body-axis AND
    # image-frame, slightly weaker hard so synth's MPFB-rig joint defs
    # don't dominate v1's visual-landmark conventions).
    loss_fn = V2DistillationLoss(
        lam_hard       = args.lam_hard,
        lam_kd_b       = args.lam_kd_b,
        lam_anchor     = args.lam_anchor,
        lam_anchor_img = args.lam_anchor_img,
        lam_mv         = args.lam_mv,
    )
    if is_main:
        print(f"[train] loss weights: hard={args.lam_hard}  kd_b={args.lam_kd_b}  "
              f"anchor={args.lam_anchor}  anchor_img={args.lam_anchor_img}  "
              f"mv={args.lam_mv}")
    # EMA shadow weights live only on rank 0 (DDP keeps replicas in sync, so
    # rank 0's copy is canonical).  Saves 3× memory on a 4-GPU box.
    ema = EMA(_unwrap(student), decay=args.ema_decay) if is_main else None
    total_steps = len(train_loader) * args.epochs

    # ── Resume ──────────────────────────────────────────────────────────
    # Every rank loads the same checkpoint (state_dict is identical across
    # ranks anyway, since DDP all-reduces gradients).  Strip "module." prefix
    # if present so the saved-without-DDP and saved-with-DDP forms interop.
    def _load_student_state(d):
        sd = {k.removeprefix("module."): v for k, v in d.items()}
        _unwrap(student).load_state_dict(sd)

    start_step = 0
    resume_ckpt = None
    if args.resume and args.resume.exists():
        resume_ckpt = args.resume
    elif not args.no_auto_resume:
        latest = sorted(args.out_root.glob(f"{args.variant}_step*.pt"))
        if latest:
            resume_ckpt = latest[-1]
    if resume_ckpt is not None:
        if is_main:
            print(f"[train] resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        _load_student_state(ckpt["student"])
        opt.load_state_dict(ckpt["opt"])
        if is_main and ema is not None:
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        start_step = ckpt.get("step", 0)
        if is_main:
            print(f"[train] resumed at step {start_step}")

    # ── Train loop ──────────────────────────────────────────────────────
    step = start_step
    t_last_ckpt = time.time()
    best_delta_mm = float("inf")    # tracks the best epoch's Δ vs v1 (rank 0 only)
    best_state_saved = False
    if is_main:
        print(f"[train] starting at step {step}/{total_steps}  "
              f"({len(train_loader)} steps/epoch × {args.epochs} epochs)")
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1
            # LR sched
            lr = warmup_cosine_lr(step, total_steps, args.warmup_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            # FixMatch-style: student gets the strong-aug crop, teacher / anchor
            # get the weak-aug crop.  Same labels for both because the augs
            # don't apply geometric transforms (only photometric + occluder paste).
            img       = batch["image"].to(device).to(dtype)        # strong
            img_weak  = batch.get("image_weak", batch["image"]).to(device).to(dtype)
            hard = {
                "bp33_xyz_body": batch["bp33_xyz_body"].to(device).to(dtype),
                "bp33_present":  batch["bp33_present"].to(device).to(dtype),
            }

            # Body teacher: gated by --use-heavy-kd because Heavy is *worse*
            # than v1 Full on this benchmark (-5.6 mm; see RESULTS.md).
            # Default OFF — student gets supervision only from v1 anchor
            # (image-frame + body-axis), hard targets, and multi-view.
            teacher_body = None
            if args.use_heavy_kd and "teacher_body_Identity" in batch:
                tb_valid = batch["teacher_body_valid"].to(device).to(dtype)
                if tb_valid.any():
                    teacher_body = {
                        "Identity":   batch["teacher_body_Identity"].to(device).to(dtype),
                        "Identity_4": batch["teacher_body_Identity_4"].to(device).to(dtype),
                        "valid":      tb_valid,
                    }
            elif args.use_heavy_kd and teacher is not None:
                with torch.no_grad():
                    teacher_body = {k: v.detach()
                                    for k, v in teacher(img_weak).items()}

            # Hand teacher: only fires when the cache holds Sárándi-aligned values
            teacher_hand = None
            if "teacher_hand_xyz" in batch:
                th_valid = batch["teacher_hand_valid"].to(device).to(dtype)
                if th_valid.any():
                    teacher_hand = {
                        "bp33_xyz_body": batch["teacher_hand_xyz"].to(device).to(dtype),
                        "bp33_present":  batch["teacher_hand_present"].to(device).to(dtype),
                        "valid":         th_valid,
                    }

            with torch.no_grad():
                anchor_out = {k: v.detach() for k, v in anchor(img_weak).items()}

            # Multi-view reprojection inputs (Ego-Exo4D samples carry these;
            # synth samples have mv_valid=0 → loss zeros out).
            multiview = None
            if "mv_valid" in batch:
                multiview = {
                    "mv_valid":      batch["mv_valid"],
                    "mv_K_norm":     batch["mv_K_norm"],
                    "mv_R_body2cam": batch["mv_R_body2cam"],
                    "mv_origin_cam": batch["mv_origin_cam"],
                    "mv_kp2d_norm":  batch["mv_kp2d_norm"],
                    "mv_present_2d": batch["mv_present_2d"],
                }

            s_out = student(img)

            losses = loss_fn(s_out, hard=hard, teacher_body=teacher_body,
                             teacher_hand=teacher_hand,
                             anchor=anchor_out, multiview=multiview)
            loss = losses["total"]

            # Step-1 sanity: confirm each component is firing or zero-by-design.
            # Catches plumbing bugs like "anchor_img stays at 0" early.
            if step == start_step + 1 and is_main:
                print(f"  [diag] step-1 loss components:")
                for k, v in losses.items():
                    if k == "total": continue
                    print(f"    {k:14s} = {v.item():.5f}")
                print(f"    {'total':14s} = {loss.item():.5f}")
                print(f"  [diag] heavy_kd={'on' if args.use_heavy_kd else 'OFF'}  "
                      f"freeze_backbone={'on' if args.freeze_backbone else 'OFF'}")

            # NaN guard: skip optimizer step instead of polluting weights.
            # Print which loss component blew up the first 3 times we hit it.
            if not torch.isfinite(loss):
                if not hasattr(main, "_nan_count"):
                    main._nan_count = 0
                main._nan_count += 1
                if main._nan_count <= 3 and is_main:
                    bad = [k for k, v in losses.items()
                           if not torch.isfinite(v).all().item()]
                    print(f"  [WARN step={step}] non-finite loss; "
                          f"components: {bad}; skipping optimizer step")
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(_unwrap(student))

            # Log (rank 0 only)
            if is_main and step % 20 == 0:
                writer.add_scalar("train/lr", lr, step)
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), step)
                print(f"  [step {step}/{total_steps}]  total={loss.item():.4f}  "
                      f"hard={losses['L_hard'].item():.4f}  "
                      f"kd_b={losses['L_kd_body'].item():.4f}  "
                      f"anc={losses['L_anchor'].item():.4f}  "
                      f"anc_img={losses['L_anchor_img'].item():.4f}  "
                      f"lr={lr:.2e}")

            # Periodic checkpoint (rank 0 only)
            if is_main and (time.time() - t_last_ckpt) > args.ckpt_every_min * 60:
                ckpt_path = args.out_root / f"{args.variant}_step{step:07d}.pt"
                torch.save({
                    "student": _unwrap(student).state_dict(),
                    "ema":     ema.shadow if ema is not None else None,
                    "opt":     opt.state_dict(),
                    "step":    step,
                }, ckpt_path)
                print(f"  [ckpt] -> {ckpt_path}")
                t_last_ckpt = time.time()

        # End-of-epoch validation (rank 0 only — runs on the unwrapped module
        # so DDP forward synchronization isn't required here).
        if is_main:
            eval_model = _unwrap(student)
            eval_model.eval()
            val_metrics = quick_val(eval_model, anchor, val_loader, device, dtype)
            bench_metrics = benchmark_eval(eval_model, device, dtype)
            # v1 baseline through the SAME path — the per-epoch benchmark uses
            # train.py's PyTorch port + naive resize-to-256 (destroys the
            # 448px ego-exo aspect ratio), so absolute numbers can be 100+ mm
            # worse than the .task pipeline's 90 mm.  Comparing student to v1
            # under the SAME broken metric tells us whether v2 is actually
            # worse than v1 or just suffering from a methodology artifact.
            v1_bench = benchmark_eval(anchor, device, dtype)
            per_kp_metrics = per_keypoint_breakdown(eval_model, anchor, val_loader,
                                                    device, dtype)
            all_metrics = {**val_metrics, **bench_metrics, **per_kp_metrics}
            all_metrics["bench_v1_baseline_mm"] = v1_bench["bench_pa_mpjpe_mm"]
            for k, v in all_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, step)
            eval_model.train()
            v2_mm = bench_metrics["bench_pa_mpjpe_mm"]
            v1_mm = v1_bench["bench_pa_mpjpe_mm"]
            delta = v2_mm - v1_mm
            print(f"  === epoch {epoch+1}/{args.epochs}  "
                  f"drift={val_metrics['val_anchor_drift_m']:.4f}  "
                  f"BENCH: v2={v2_mm:.1f}  v1={v1_mm:.1f}  Δ={delta:+.1f}mm  "
                  f"per_kp_max={per_kp_metrics.get('per_kp_drift_max_mm', float('nan')):.1f}mm  "
                  f"vis_agree={per_kp_metrics.get('vis_agreement_pct', 0):.1f}%")

            # Best-checkpoint selection (rank 0 only).  Smoke 6 epoch 2 had
            # Δ=-2.8 mm (BEAT v1) but later epochs degraded to +10 mm; saving
            # only the FINAL ckpt threw away the SOTA-quality intermediate
            # state.  Track best per-epoch Δ-vs-v1 and snapshot the EMA
            # weights at that point — that's the model we'll export.
            if not (delta != delta):  # not NaN
                if delta < best_delta_mm:
                    best_delta_mm = delta
                    best_path = args.out_root / f"{args.variant}_best.pt"
                    # Apply EMA to the unwrapped module BEFORE saving — that's
                    # the form export.py expects.  Snapshot then restore so
                    # training continues from the live (non-EMA) weights.
                    if ema is not None:
                        live_state = {k: v.detach().clone()
                                      for k, v in _unwrap(student).state_dict().items()}
                        ema.apply_to(_unwrap(student))
                        torch.save({
                            "student": _unwrap(student).state_dict(),
                            "ema":     ema.shadow,
                            "opt":     opt.state_dict(),
                            "step":    step,
                            "epoch":   epoch + 1,
                            "delta_mm": delta,
                        }, best_path)
                        # Restore live weights
                        _unwrap(student).load_state_dict(live_state)
                    else:
                        torch.save({
                            "student": _unwrap(student).state_dict(),
                            "ema":     None,
                            "opt":     opt.state_dict(),
                            "step":    step,
                            "epoch":   epoch + 1,
                            "delta_mm": delta,
                        }, best_path)
                    best_state_saved = True
                    print(f"  [best-ckpt] new best Δ={delta:+.2f}mm @ epoch {epoch+1}  "
                          f"-> {best_path}")
        # All ranks wait for rank 0 to finish val before starting next epoch
        if is_dist:
            dist.barrier()

    # Final save (rank 0 only — applies EMA to the underlying module)
    if is_main:
        final = args.out_root / f"{args.variant}_final.pt"
        if ema is not None:
            ema.apply_to(_unwrap(student))
        torch.save({
            "student": _unwrap(student).state_dict(),
            "ema":     ema.shadow if ema is not None else None,
            "opt":     opt.state_dict(),
            "step":    step,
        }, final)
        print(f"[train] DONE -> {final}")
        writer.close()
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
