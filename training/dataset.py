"""Dataset for BlazePose v2 distillation.

Synth + Ego-Exo4D unified; emits everything the V2DistillationLoss expects:
  - image:           (3, 256, 256) float32 in [0,1] NCHW
  - bp33_xyz_body:   (33, 3) hard 3-D supervision in BP body-axis frame
  - bp33_present:    (33,)   per-joint mask (1.0 if hard signal available)
  - sample_id:       str
  - source:          "synth" | "egoexo"
  - cached teachers (optional): teacher_body / teacher_hand / teacher_face
                                each with bp33_xyz_body + bp33_present

The augmentation pipeline (F1 occluder paste + F2 bg composite + photometric
jitter) is wired in via `tooling/sim2real_aug.py`.  Augmentation is applied
**after** teacher inference (we always cache teachers on the un-augmented
frame), and is gentler on real Ego-Exo4D frames than on synth.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmark"))
sys.path.insert(0, str(HERE))

from lib.keypoint_map import COCO17, BP_INDEX_FOR_COCO, HIP_L, HIP_R
from coords import hard_target_in_body_axis


# COCO-17 left↔right pairs for horizontal-flip KP swap
COCO17_LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                   (11, 12), (13, 14), (15, 16)]


def hflip_with_kp_swap(img_bgr: np.ndarray, kp17_cam: np.ndarray,
                       kp17_2d: np.ndarray, present17: np.ndarray):
    """Horizontal flip: mirror image AND swap L/R keypoint pairs AND
    negate cam-frame X.  Returns the flipped (img, kp_cam, kp_2d)."""
    img = cv2.flip(img_bgr, 1)
    H, W = img.shape[:2]
    # Flip cam-frame X (since X-right became X-left after image flip)
    kp17_cam_f = kp17_cam.copy()
    kp17_cam_f[:, 0] *= -1.0
    # Flip 2D x
    kp17_2d_f = kp17_2d.copy()
    kp17_2d_f[:, 0] = (W - 1) - kp17_2d_f[:, 0]
    # Swap L/R pairs
    for a, b in COCO17_LR_PAIRS:
        kp17_cam_f[[a, b]] = kp17_cam_f[[b, a]]
        kp17_2d_f[[a, b]]  = kp17_2d_f[[b, a]]
        present17[[a, b]]  = present17[[b, a]]
    return img, kp17_cam_f, kp17_2d_f, present17


# ─── 256×256 letterbox ────────────────────────────────────────────────────

def pad_to_square_256(img_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Letterbox to 256x256 BGR.

    Handles both directions:
      - synth (256x192):  scale=1, pad H by 32 each side  → 256x256
      - ego-exo (448p, e.g. 796x448): scale=0.322 (256/796), resize to
        (256x144), pad H by 56 each side → 256x256
      - already 256x256: pass-through

    Returns (padded_image, pad_h, pad_w).  Both pad values are post-resize.
    """
    H, W = img_bgr.shape[:2]
    if (H, W) == (256, 256):
        return img_bgr, 0, 0
    target = 256
    # Scale down if larger than target (preserve aspect ratio).
    s = min(target / W, target / H, 1.0)   # never upscale
    if s < 1.0:
        new_W, new_H = max(1, int(round(W * s))), max(1, int(round(H * s)))
        img_bgr = cv2.resize(img_bgr, (new_W, new_H), interpolation=cv2.INTER_AREA)
        H, W = new_H, new_W
    # Now W <= target and H <= target; pad remaining axes.
    pad_w = (target - W) // 2
    pad_h = (target - H) // 2
    out = cv2.copyMakeBorder(img_bgr, pad_h, target - H - pad_h,
                             pad_w, target - W - pad_w,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out, pad_h, pad_w


# ─── Uniform teacher-field emission for default_collate ───────────────────

def _attach_teacher_fields(item: dict, teach: dict | None) -> None:
    """Emit a uniform set of teacher tensors on every sample so PyTorch's
    default_collate sees identical keys across the batch.

    When the teacher cache is missing for a sample, we still emit zero
    placeholders + `teacher_*_valid = 0.0`; the loss multiplies its mask
    by `valid` so those samples contribute nothing to the loss term and
    nothing to its denominator.  This is the canonical pattern recommended
    by PyTorch (see issue #96085) instead of a custom collate_fn.
    """
    if teach is not None and "world33" in teach:
        world33 = teach["world33"][:, :3]                          # (33, 3)
        world39 = np.zeros((39, 3), dtype=np.float32)
        world39[:33] = world33
        img33 = teach.get("img33", np.zeros((33, 5), dtype=np.float32))
        img39 = np.zeros((39, 5), dtype=np.float32)
        img39[:33] = img33
        item["teacher_body_Identity"]   = torch.from_numpy(img39.flatten())
        item["teacher_body_Identity_4"] = torch.from_numpy(world39.flatten())
        item["teacher_body_valid"]      = torch.tensor(1.0, dtype=torch.float32)
    else:
        item["teacher_body_Identity"]   = torch.zeros(195, dtype=torch.float32)
        item["teacher_body_Identity_4"] = torch.zeros(117, dtype=torch.float32)
        item["teacher_body_valid"]      = torch.tensor(0.0, dtype=torch.float32)

    if teach is not None and "hand_bp33_xyz_body" in teach:
        item["teacher_hand_xyz"]     = torch.from_numpy(
            teach["hand_bp33_xyz_body"].astype(np.float32))
        item["teacher_hand_present"] = torch.from_numpy(
            teach["hand_bp33_present"].astype(np.float32))
        item["teacher_hand_valid"]   = torch.tensor(1.0, dtype=torch.float32)
    else:
        item["teacher_hand_xyz"]     = torch.zeros(33, 3, dtype=torch.float32)
        item["teacher_hand_present"] = torch.zeros(33, dtype=torch.float32)
        item["teacher_hand_valid"]   = torch.tensor(0.0, dtype=torch.float32)


# ─── Optional augmentation hook ───────────────────────────────────────────

def maybe_load_aug_corpus():
    """Return the Sim2RealAug instance, or None if unavailable.
    Failure is non-fatal (we fall back to clean frames)."""
    try:
        from augment import build_default
        return build_default()
    except Exception as e:
        print(f"[dataset] aug disabled: {e}")
        return None


# ─── Synth dataset (HF 500k + iter sanity set both supported) ─────────────

class SynthDataset(Dataset):
    """Synth corpus reader.

    `labels.jsonl` format follows the existing 17-COCO synth pipeline:
        keypoints_3d_cam: (17, 3) cam-frame metres
        keypoints_2d:     (17, 3) image-px + visibility flag (col 2)
        camera_K:         (3, 3)
        camera_extrinsics: {R, t}
        bbox_xywh:        (4,)
    """

    def __init__(self, labels_jsonl: Path, images_root: Path,
                 teacher_cache_dir: Path | None = None,
                 augment: bool = False,
                 limit: int | None = None,
                 split: str | None = "train"):
        """`split`: "train" | "val" | None (use all records).  Synth's
        clean.zip carries BOTH splits in the same labels.jsonl, partitioned
        by sha1(id)[:2] < 0x1A — deterministic, do NOT reshuffle.
        Default "train" filters to ~89.8 % of records (133,418)."""
        self.images_root = Path(images_root)
        self.records: list[dict[str, Any]] = []
        skipped = 0
        with Path(labels_jsonl).open() as fh:
            for ln in fh:
                rec = json.loads(ln)
                if split is not None and rec.get("split") != split:
                    skipped += 1
                    continue
                self.records.append(rec)
        if limit is not None:
            self.records = self.records[:limit]
        self.teacher_cache_dir = Path(teacher_cache_dir) if teacher_cache_dir else None
        self.aug = maybe_load_aug_corpus() if augment else None
        print(f"[synth] {len(self.records)} {split or 'all'} records  "
              f"(filtered {skipped} other-split)  augment={self.aug is not None}")

    def __len__(self):
        return len(self.records)

    def _load_teacher(self, sample_id: str) -> dict | None:
        if self.teacher_cache_dir is None:
            return None
        p = self.teacher_cache_dir / f"{sample_id}.npz"
        if not p.exists():
            return None
        d = np.load(p)
        return {k: d[k] for k in d.files}

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.images_root / rec["image_rel"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        img_padded, _, _ = pad_to_square_256(img_bgr)

        # Hard supervision: 17-COCO cam metres → BP-33 body-axis
        kp17_cam = np.asarray(rec["keypoints_3d_cam"], dtype=np.float32)
        kp2d_full = np.asarray(rec["keypoints_2d"], dtype=np.float32)
        present17 = ((kp2d_full[:, 2] > 0).astype(np.float32) if kp2d_full.shape[1] >= 3
                     else np.ones(17, dtype=np.float32))
        # Synth KPs come from MPFB2 RIG BONE heads/tails, NOT visual
        # landmarks.  BlazePose was trained with visual-landmark annotations,
        # so using rig joints as hard supervision injects systematic offsets:
        #   - face (COCO 0-4): nose ≈ face centre (not tip), ears ← eye bones
        #   - hips (COCO 11-12): MPFB femoral-head vs BP visual-hip surface
        #     (2-4 cm offset, documented in benchmark/results/RESULTS.md
        #     where v1 hip PA-MPJPE bottoms at ~100 mm across all variants)
        #   - knees (COCO 13-14): rig knee-bone joint vs BP visual centre
        # Mask all of these; let teacher_body KD cover them via Heavy v1
        # (which was trained on visual-landmark style).  This was the root
        # cause of the smoke run regressing v2_full lower body by 60-125 mm.
        present17[0:5] = 0.0    # face (nose, eyes, ears)
        present17[11:15] = 0.0  # hips + knees
        kp17_2d_native = kp2d_full[:, :2].astype(np.float32)
        # Random horizontal flip with KP-pair swap (p=0.5).  Must be done
        # BEFORE the body-axis transform so the swapped KPs feed coords.py.
        if random.random() < 0.5:
            img_padded, kp17_cam, kp17_2d_native, present17 = hflip_with_kp_swap(
                img_padded, kp17_cam, kp17_2d_native, present17.copy())
        bp33_xyz_body, bp33_present = hard_target_in_body_axis(kp17_cam, present17)

        # Multi-view consistency: synth has its own K + GT 2D so the same loss
        # applies.  Normalise to [0,1] image-frac coords.
        from coords import build_body_frame
        K_native = np.asarray(rec["camera_K"], dtype=np.float32)
        R_cam2body, _origin = build_body_frame(kp17_cam, present17)
        if R_cam2body is None:
            R_body2cam = np.zeros((3, 3), dtype=np.float32)
            origin_cam_ = np.zeros(3, dtype=np.float32)
            mv_valid = 0.0
        else:
            R_body2cam = R_cam2body.T.astype(np.float32)
            origin_cam_ = _origin.astype(np.float32)
            mv_valid = 1.0
        # Use explicit image_wh from synth label (more precise than 2*cx/cy)
        W_native, H_native = rec.get("image_wh", [256, 192])
        W_native = max(float(W_native), 1.0)
        H_native = max(float(H_native), 1.0)
        K_norm = K_native.copy()
        K_norm[0, :] /= W_native
        K_norm[1, :] /= H_native
        kp17_2d_norm = kp17_2d_native.copy()
        kp17_2d_norm[:, 0] /= W_native
        kp17_2d_norm[:, 1] /= H_native

        # FixMatch-style strong/weak split: student gets the strong-aug crop,
        # teacher + anchor get the weak crop.  Both share identity (no
        # geometric jitter), so 3-D KP targets remain valid for both.
        if self.aug is not None:
            try:
                img_strong = self.aug(img_padded.copy(), source="synth", strong=True)
                img_weak   = self.aug(img_padded.copy(), source="synth", strong=False)
            except Exception:
                img_strong = img_weak = img_padded
        else:
            img_strong = img_weak = img_padded

        def _to_tensor(b):
            rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        item: dict[str, Any] = {
            "image":          _to_tensor(img_strong),     # student input
            "image_weak":     _to_tensor(img_weak),       # teacher / anchor input
            "bp33_xyz_body":  torch.from_numpy(bp33_xyz_body),
            "bp33_present":   torch.from_numpy(bp33_present),
            "sample_id":      rec["id"],
            "source":         "synth",
            # Multi-view consistency (uniform across synth + egoexo so MixedDataset works)
            "mv_valid":       torch.tensor(mv_valid, dtype=torch.float32),
            "mv_K_norm":      torch.from_numpy(K_norm.astype(np.float32)),
            "mv_R_body2cam":  torch.from_numpy(R_body2cam),
            "mv_origin_cam":  torch.from_numpy(origin_cam_),
            "mv_kp2d_norm":   torch.from_numpy(kp17_2d_norm.astype(np.float32)),
            "mv_present_2d":  torch.from_numpy(present17.astype(np.float32)),
        }
        # Always emit teacher tensors (zero placeholders + per-sample valid flag)
        # so default_collate sees uniform keys across samples.  Loss gates on valid.
        _attach_teacher_fields(item, self._load_teacher(rec["id"]))
        return item


# ─── Ego-Exo4D dataset (train split, manifest-driven) ─────────────────────

class EgoExoTrainDataset(Dataset):
    """Reads `manifest_train.jsonl` written by prep/extract_egoexo_train.py
    and the body GT JSON files alongside.

    Each item is a (uid, cam, frame) triple.  Hard supervision: project
    world-frame GT into the cam, then transform to BP body-axis.
    """

    def __init__(self, manifest_jsonl: Path, frames_root: Path,
                 anno_root: Path, teacher_cache_dir: Path | None = None,
                 augment: bool = False, limit: int | None = None):
        self.frames_root = Path(frames_root)
        self.anno_root   = Path(anno_root)
        self.records: list[dict[str, Any]] = []
        with Path(manifest_jsonl).open() as fh:
            for ln in fh:
                self.records.append(json.loads(ln))
        if limit is not None:
            self.records = self.records[:limit]
        self._gt_cache: dict[str, dict] = {}        # uid -> body GT
        self._cp_cache: dict[str, dict] = {}        # uid -> camera_pose
        self.teacher_cache_dir = Path(teacher_cache_dir) if teacher_cache_dir else None
        self.aug = maybe_load_aug_corpus() if augment else None
        print(f"[egoexo] {len(self.records)} samples  augment={self.aug is not None}")

    def __len__(self):
        return len(self.records)

    def _load_gt(self, uid: str) -> dict:
        if uid in self._gt_cache:
            return self._gt_cache[uid]
        from lib.ego_exo_io import load_body_gt
        gt = load_body_gt(self.anno_root / "ego_pose" / "train" / "body" / "annotation"
                          / f"{uid}.json")
        self._gt_cache[uid] = gt
        return gt

    def _load_cp(self, uid: str) -> dict:
        if uid in self._cp_cache:
            return self._cp_cache[uid]
        from lib.ego_exo_io import load_camera_pose
        cp = load_camera_pose(self.anno_root / "ego_pose" / "train" / "camera_pose"
                              / f"{uid}.json")
        self._cp_cache[uid] = cp
        return cp

    def __getitem__(self, idx):
        rec = self.records[idx]
        uid = rec["take_uid"]; cam = rec["cam"]; fi = rec["frame"]
        img_path = self.frames_root / rec["image_path"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        H_in, W_in = img_bgr.shape[:2]
        img_padded, pad_h, pad_w = pad_to_square_256(img_bgr)

        # Hard target: world GT → cam → body-axis
        gt = self._load_gt(uid).get(fi)
        present17 = np.zeros(17, dtype=np.float32)
        kp17_cam  = np.zeros((17, 3), dtype=np.float32)
        kp17_2d   = np.zeros((17, 2), dtype=np.float32)
        present17_2d = np.zeros(17, dtype=np.float32)
        K_native = np.eye(3, dtype=np.float32)
        Rt_cam   = np.zeros((3, 4), dtype=np.float32)
        if gt is not None:
            from lib.keypoint_map import gt_to_coco17
            from lib.projection import world_to_cam
            kp_w, present17_b = gt_to_coco17(gt["annotation3D"])
            present17 = present17_b.astype(np.float32)
            cp = self._load_cp(uid)
            cam_data = cp["cams"].get(cam)
            if cam_data is not None:
                kp17_cam = world_to_cam(kp_w, cam_data["Rt"]).astype(np.float32)
                K_native = cam_data["K"].astype(np.float32)
                Rt_cam   = cam_data["Rt"].astype(np.float32)
                ann2d = gt.get("annotation2D", {}).get(cam, {})
                for j_idx, jname in enumerate(COCO17):
                    a = ann2d.get(jname)
                    if a is not None:
                        kp17_2d[j_idx] = (a["x"], a["y"])
                        present17_2d[j_idx] = 1.0 if a.get("placement") == "manual" else 0.5

        # Random horizontal flip (BEFORE body-axis transform).  Note: we have
        # to flip in PADDED pixel space because that's the image we feed the net.
        # 2-D coords from `annotation2D` are in NATIVE pixel space — flip
        # those by the native width.
        if random.random() < 0.5:
            img_padded = cv2.flip(img_padded, 1)
            # Native flip
            cx_native = float(K_native[0, 2])
            W_native_est = max(2.0 * cx_native, 1.0)
            kp17_cam[:, 0] *= -1.0
            kp17_2d[:, 0]  = (W_native_est - 1) - kp17_2d[:, 0]
            for a, b in COCO17_LR_PAIRS:
                kp17_cam[[a, b]]    = kp17_cam[[b, a]]
                kp17_2d[[a, b]]     = kp17_2d[[b, a]]
                present17[[a, b]]   = present17[[b, a]]
                present17_2d[[a, b]] = present17_2d[[b, a]]

        bp33_xyz_body, bp33_present = hard_target_in_body_axis(
            kp17_cam, present17)

        # Multi-view consistency inputs.  Normalise K + 2D to [0, 1] image-frac
        # coords so the loss is scale-agnostic across synth (256-px) and
        # Ego-Exo4D (3840-px) sources.
        from coords import build_body_frame
        R_cam2body, _origin = build_body_frame(kp17_cam, present17.astype(np.float32))
        if R_cam2body is None:
            R_body2cam = np.zeros((3, 3), dtype=np.float32)
            origin_cam_ = np.zeros(3, dtype=np.float32)
            mv_valid = 0.0
        else:
            R_body2cam = R_cam2body.T.astype(np.float32)
            origin_cam_ = _origin.astype(np.float32)
            mv_valid = 1.0
        # Estimate native image (W, H) from K's principal point: cx ≈ W/2, cy ≈ H/2
        W_native = max(2.0 * float(K_native[0, 2]), 1.0)
        H_native = max(2.0 * float(K_native[1, 2]), 1.0)
        # Normalised K
        K_norm = K_native.copy()
        K_norm[0, :] /= W_native
        K_norm[1, :] /= H_native
        # Normalised 2D
        kp17_2d_norm = kp17_2d.copy()
        kp17_2d_norm[:, 0] /= W_native
        kp17_2d_norm[:, 1] /= H_native

        if self.aug is not None:
            try:
                img_strong = self.aug(img_padded.copy(), source="egoexo", strong=True)
                img_weak   = self.aug(img_padded.copy(), source="egoexo", strong=False)
            except Exception:
                img_strong = img_weak = img_padded
        else:
            img_strong = img_weak = img_padded

        def _to_tensor(b):
            rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        sample_id = f"{uid}__{cam}__{fi:06d}"
        item: dict[str, Any] = {
            "image":          _to_tensor(img_strong),
            "image_weak":     _to_tensor(img_weak),
            "bp33_xyz_body":  torch.from_numpy(bp33_xyz_body),
            "bp33_present":   torch.from_numpy(bp33_present),
            "sample_id":      sample_id,
            "source":         "egoexo",
            # ── Multi-view reprojection inputs (normalised to [0,1] image-frac) ──
            "mv_valid":       torch.tensor(mv_valid, dtype=torch.float32),
            "mv_K_norm":      torch.from_numpy(K_norm.astype(np.float32)),
            "mv_R_body2cam":  torch.from_numpy(R_body2cam),
            "mv_origin_cam":  torch.from_numpy(origin_cam_),
            "mv_kp2d_norm":   torch.from_numpy(kp17_2d_norm.astype(np.float32)),
            "mv_present_2d":  torch.from_numpy(present17_2d.astype(np.float32)),
        }
        teach = None
        if self.teacher_cache_dir is not None:
            tp = self.teacher_cache_dir / f"{sample_id}.npz"
            if tp.exists():
                d = np.load(tp)
                teach = {k: d[k] for k in d.files}
        _attach_teacher_fields(item, teach)
        return item


# ─── Mix wrapper: replay buffer / source-aware mixing ─────────────────────

class MixedDataset(Dataset):
    """Per-step deterministic mix: synth_ratio of synth, rest egoexo.
    Length = max(len(synth), len(egoexo)) so every sample seen at least once
    per epoch on the longer side."""

    def __init__(self, synth_ds: Dataset, egoexo_ds: Dataset,
                 synth_ratio: float = 0.4):
        self.s = synth_ds
        self.e = egoexo_ds
        self.synth_ratio = synth_ratio
        self.length = max(len(synth_ds), len(egoexo_ds))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if random.random() < self.synth_ratio:
            return self.s[idx % len(self.s)]
        return self.e[idx % len(self.e)]


if __name__ == "__main__":
    """Smoke: synth iter sanity set."""
    SYNTH = Path(r"../3D-Body-Tracking-Approach/dataset/output/synth_iter")
    ds = SynthDataset(SYNTH / "labels.jsonl", SYNTH, limit=4, augment=False)
    for i, item in enumerate(ds):
        bp33 = item["bp33_xyz_body"].numpy()
        pres = item["bp33_present"].numpy()
        print(f"  [{i}] id={item['sample_id']}  source={item['source']}  "
              f"img={tuple(item['image'].shape)}  "
              f"hard_present={int(pres.sum())}/33  "
              f"hip_mid_body={bp33[[23, 24]].mean(0).tolist()}")
    print("[smoke] dataset OK")
