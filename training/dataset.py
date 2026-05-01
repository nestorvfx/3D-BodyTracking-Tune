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


# ─── 256×256 letterbox ────────────────────────────────────────────────────

def pad_to_square_256(img_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    H, W = img_bgr.shape[:2]
    if (H, W) == (256, 256):
        return img_bgr, 0, 0
    target = 256
    pad_w = (target - W) // 2 if W < target else 0
    pad_h = (target - H) // 2 if H < target else 0
    out = cv2.copyMakeBorder(img_bgr, pad_h, target - H - pad_h,
                             pad_w, target - W - pad_w,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out, pad_h, pad_w


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
                 limit: int | None = None):
        self.images_root = Path(images_root)
        self.records: list[dict[str, Any]] = []
        with Path(labels_jsonl).open() as fh:
            for ln in fh:
                self.records.append(json.loads(ln))
        if limit is not None:
            self.records = self.records[:limit]
        self.teacher_cache_dir = Path(teacher_cache_dir) if teacher_cache_dir else None
        self.aug = maybe_load_aug_corpus() if augment else None
        print(f"[synth] {len(self.records)} records  augment={self.aug is not None}")

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
        kp2d = np.asarray(rec["keypoints_2d"], dtype=np.float32)
        present17 = ((kp2d[:, 2] > 0).astype(np.float32) if kp2d.shape[1] >= 3
                     else np.ones(17, dtype=np.float32))
        bp33_xyz_body, bp33_present = hard_target_in_body_axis(kp17_cam, present17)

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
        }
        teach = self._load_teacher(rec["id"])
        if teach is not None and "world33" in teach:
            # Heavy teacher cache → port-format dict (pad 33 → 39)
            world33 = teach["world33"][:, :3]                         # (33, 3)
            world39 = np.zeros((39, 3), dtype=np.float32)
            world39[:33] = world33
            img33 = teach.get("img33", np.zeros((33, 5), dtype=np.float32))
            img39 = np.zeros((39, 5), dtype=np.float32)
            img39[:33] = img33
            item["teacher_body_Identity"]   = torch.from_numpy(img39.flatten())
            item["teacher_body_Identity_4"] = torch.from_numpy(world39.flatten())
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
        img_padded, _, _ = pad_to_square_256(img_bgr)

        # Hard target: world GT → cam → body-axis
        gt = self._load_gt(uid).get(fi)
        present17 = np.zeros(17, dtype=np.float32)
        kp17_cam  = np.zeros((17, 3), dtype=np.float32)
        if gt is not None:
            from lib.keypoint_map import gt_to_coco17
            from lib.projection import world_to_cam
            kp_w, present17 = gt_to_coco17(gt["annotation3D"])
            cp = self._load_cp(uid)
            cam_data = cp["cams"].get(cam)
            if cam_data is not None:
                kp17_cam = world_to_cam(kp_w, cam_data["Rt"]).astype(np.float32)
        bp33_xyz_body, bp33_present = hard_target_in_body_axis(
            kp17_cam, present17.astype(np.float32))

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
        }
        if self.teacher_cache_dir is not None:
            tp = self.teacher_cache_dir / f"{sample_id}.npz"
            if tp.exists():
                d = np.load(tp)
                if "world33" in d.files:
                    world39 = np.zeros((39, 3), dtype=np.float32)
                    world39[:33] = d["world33"][:, :3]
                    img39 = np.zeros((39, 5), dtype=np.float32)
                    if "img33" in d.files:
                        img39[:33] = d["img33"]
                    item["teacher_body_Identity"]   = torch.from_numpy(img39.flatten())
                    item["teacher_body_Identity_4"] = torch.from_numpy(world39.flatten())
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
