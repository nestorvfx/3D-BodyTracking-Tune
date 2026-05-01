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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
def benchmark_eval(student, device, dtype, max_takes: int = 5) -> dict:
    """Run the SOTA benchmark on a small subset of frames.  Computes
    PA-MPJPE-vs-Heavy-v1 — directly the metric we want to track training on.

    Loads frames from `benchmark/frames/` (must already be extracted by
    the benchmark pipeline; on Vast this is part of the held-out artefact)
    and `benchmark/frames_manifest.json`.  Returns NaN-filled dict if the
    benchmark isn't on disk yet — non-fatal."""
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
            for fi in fidxs[:3]:
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
    ap.add_argument("--epochs",         type=int,   default=10)
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
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype  = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.runs_root / f"v2_{args.variant}_{int(time.time())}"
    run_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    print(f"[train] device={device}  dtype={dtype}  variant={args.variant}")
    print(f"[train] run_dir={run_dir}")

    # ── Models ──────────────────────────────────────────────────────────
    student_task = ASSETS / f"pose_landmarker_{args.variant}.task"
    print(f"[train] loading student from {student_task}")
    student, _ = load_task(student_task)
    student.train().to(device).to(dtype)

    print(f"[train] loading frozen anchor (v1 weights)")
    anchor, _ = load_task(student_task)
    for p in anchor.parameters():
        p.requires_grad_(False)
    anchor.eval().to(device).to(dtype)

    # Heavy teacher: only live if we don't have a cache for *every* sample
    teacher = None
    if args.teacher_cache is None or not args.teacher_cache.exists():
        print(f"[train] loading Heavy teacher live (slow CPU/GPU; "
              f"prefer pre-cached teachers via prep/cache_teachers.py)")
        teacher, _ = load_task(ASSETS / "pose_landmarker_heavy.task")
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval().to(device).to(dtype)

    # ── Datasets ────────────────────────────────────────────────────────
    synth_ds = SynthDataset(
        labels_jsonl=args.synth_root / "labels.jsonl",
        images_root=args.synth_root,
        teacher_cache_dir=args.teacher_cache,
        augment=True,
        limit=args.limit_synth)

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
        print(f"[train] no Ego-Exo4D manifest — synth-only training")
        train_ds = synth_ds

    val_loader = DataLoader(synth_ds, batch_size=min(args.batch_size, 8),
                            shuffle=False, num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0,
                              drop_last=True)

    # ── Optim + sched + EMA ─────────────────────────────────────────────
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # λ_anchor=0.4 (was 0.1): now per-joint masked, so the larger headline
    # weight concentrates 5× harder on the 20 unsupervised BP joints (with
    # the in-class mask), while the supervised joints see ~0.4 × 0.2 = 0.08
    # which is below L_hard / L_kd_body so they're still teacher-driven.
    loss_fn = V2DistillationLoss(lam_anchor=0.4)
    ema = EMA(student, decay=args.ema_decay)
    total_steps = len(train_loader) * args.epochs

    # ── Resume ──────────────────────────────────────────────────────────
    start_step = 0
    if args.resume and args.resume.exists():
        print(f"[train] resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["student"])
        opt.load_state_dict(ckpt["opt"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        start_step = ckpt.get("step", 0)
        print(f"[train] resumed at step {start_step}")
    else:
        # Find latest in out_root
        latest = sorted(args.out_root.glob(f"{args.variant}_step*.pt"))
        if latest:
            print(f"[train] auto-resume: {latest[-1]}")
            ckpt = torch.load(latest[-1], map_location=device, weights_only=False)
            student.load_state_dict(ckpt["student"])
            opt.load_state_dict(ckpt["opt"])
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
            start_step = ckpt.get("step", 0)

    # ── Train loop ──────────────────────────────────────────────────────
    step = start_step
    t_last_ckpt = time.time()
    print(f"[train] starting at step {step}/{total_steps}  "
          f"({len(train_loader)} steps/epoch × {args.epochs} epochs)")
    for epoch in range(args.epochs):
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

            # Body teacher: cache hit OR live Heavy on the weak crop
            teacher_body = None
            if "teacher_body_Identity" in batch:
                teacher_body = {
                    "Identity":   batch["teacher_body_Identity"].to(device).to(dtype),
                    "Identity_4": batch["teacher_body_Identity_4"].to(device).to(dtype),
                }
            elif teacher is not None:
                with torch.no_grad():
                    teacher_body = {k: v.detach()
                                    for k, v in teacher(img_weak).items()}

            # Hand teacher: only fires when the cache holds Sárándi-aligned values
            teacher_hand = None
            if "teacher_hand_xyz" in batch:
                teacher_hand = {
                    "bp33_xyz_body": batch["teacher_hand_xyz"].to(device).to(dtype),
                    "bp33_present":  batch["teacher_hand_present"].to(device).to(dtype),
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
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            ema.update(student)

            # Log
            if step % 20 == 0:
                writer.add_scalar("train/lr", lr, step)
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), step)
                print(f"  [step {step}/{total_steps}]  total={loss.item():.4f}  "
                      f"hard={losses['L_hard'].item():.4f}  "
                      f"kd_b={losses['L_kd_body'].item():.4f}  "
                      f"anchor={losses['L_anchor'].item():.4f}  lr={lr:.2e}")

            # Periodic checkpoint
            if (time.time() - t_last_ckpt) > args.ckpt_every_min * 60:
                ckpt_path = args.out_root / f"{args.variant}_step{step:07d}.pt"
                torch.save({
                    "student": student.state_dict(),
                    "ema":     ema.shadow,
                    "opt":     opt.state_dict(),
                    "step":    step,
                }, ckpt_path)
                print(f"  [ckpt] -> {ckpt_path}")
                t_last_ckpt = time.time()

        # End-of-epoch validation: anchor drift + actual benchmark PA-MPJPE
        student.eval()
        val_metrics = quick_val(student, anchor, val_loader, device, dtype)
        bench_metrics = benchmark_eval(student, device, dtype, max_takes=5)
        all_metrics = {**val_metrics, **bench_metrics}
        for k, v in all_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"val/{k}", v, step)
        student.train()
        print(f"  === epoch {epoch+1}/{args.epochs}  drift={val_metrics['val_anchor_drift_m']:.4f}  "
              f"BENCH_PA_MPJPE={bench_metrics['bench_pa_mpjpe_mm']:.1f} mm  "
              f"(N={bench_metrics['bench_n_frames']}, {bench_metrics['bench_note']})")

    # Final save
    final = args.out_root / f"{args.variant}_final.pt"
    ema.apply_to(student)
    torch.save({
        "student": student.state_dict(),
        "ema":     ema.shadow,
        "opt":     opt.state_dict(),
        "step":    step,
    }, final)
    print(f"[train] DONE -> {final}")
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
