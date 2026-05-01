"""Source-aware augmentation wrapper for v2 training.

Wraps the pure-numpy primitives in `tooling/sim2real_aug.py`:
  - F1 occluder paste     (Sárándi 2018)
  - F2 BG composite       (BEDLAM-CLIFF lineage)
  - FDA frequency mixing  (Yang & Soatto, CVPR'20)
plus light photometric jitter (brightness / contrast / Gaussian noise),
applied in the canonical order:
    geometric → BG composite → occluder paste → photometric

Per the user's strategy:
  - synth   (clean Blender renders): heavy aug, all primitives active
  - egoexo  (real GoPro frames): light aug, NO BG-composite (already real bg)
  - replay  (close-frontal anti-regression set): minimal aug (preserve v1 dist)

The augmenter is INSTANTIATED ONCE per dataloader worker; corpora are
loaded into RAM at __init__ (occluders ~270 MB, bg crops ~10 MB).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tooling"))

try:
    from sim2real_aug import (load_occluders_from_dir, occlude_with_objects,
                              load_bg_corpus, composite_on_real_bg,
                              apply_fda, load_fda_refs)
except Exception as e:
    print(f"[augment] sim2real_aug unavailable: {e}")
    raise


class Sim2RealAug:
    """Source-aware augmenter.  Call as a function: aug(img_bgr, source).

    Parameters per `source`:
      synth   — heavy: BG composite (p=0.5) + occluder (p=0.6) + FDA (p=0.3)
                + photometric (always)
      egoexo  — light: occluder (p=0.3) + photometric (p=0.5); no BG composite
      replay  — minimal: photometric (p=0.3) only; preserves v1 distribution
    """

    def __init__(self,
                 occluders_dir: str | Path,
                 bg_dir:        str | Path | None = None,
                 fda_dir:       str | Path | None = None,
                 max_occluders_per_image: int = 4):
        occ_dir = Path(occluders_dir)
        if not occ_dir.exists():
            self.occluders = []
        else:
            self.occluders = load_occluders_from_dir(occ_dir)
        if bg_dir and Path(bg_dir).exists():
            self.bg_corpus = load_bg_corpus(bg_dir)
        else:
            self.bg_corpus = []
        if fda_dir and Path(fda_dir).exists():
            # FDA needs target_wh (W, H) so refs match input shape after resize
            self.fda_refs = load_fda_refs(fda_dir, target_wh=(256, 256))
        else:
            self.fda_refs = []
        self.max_occluders = max_occluders_per_image

        # Per-source probabilities — tuned per the SOTA recommendation
        # (research synthesis on commercial-clean BlazePose distillation,
        # Sárándi/RTMPose/DWPose/FixMatch lineage, 2024-2026):
        #   - synth:  heavy aug — needed to bridge sim-real gap (F1 0.8, F2 0.9, FDA 0.5)
        #   - egoexo: light aug — already realistic, NEVER bg-comp real frames
        #   - replay: minimal aug — preserve v1 distribution for anti-regression
        # Plus a "weak" tier per source for FixMatch-style strong/weak split:
        # the teacher sees the weak crop, the student sees the strong crop.
        self.cfg = {
            "synth":  {"p_bg": 0.9, "p_occ": 0.8, "p_fda": 0.5, "p_photo": 1.0,
                       "photo_strength": "strong",  "p_jpeg": 0.3},
            "egoexo": {"p_bg": 0.0, "p_occ": 0.4, "p_fda": 0.0, "p_photo": 0.5,
                       "photo_strength": "light",   "p_jpeg": 0.15},
            "replay": {"p_bg": 0.0, "p_occ": 0.1, "p_fda": 0.0, "p_photo": 0.3,
                       "photo_strength": "minimal", "p_jpeg": 0.0},
        }
        # Weak variant of each (just light photometric, no occluders/bg/fda):
        self.cfg_weak = {
            "synth":  {"p_photo": 0.8, "photo_strength": "light"},
            "egoexo": {"p_photo": 0.5, "photo_strength": "light"},
            "replay": {"p_photo": 0.3, "photo_strength": "minimal"},
        }
        print(f"[augment] occluders={len(self.occluders)} bg={len(self.bg_corpus)} "
              f"fda={len(self.fda_refs)}")

    # ------------- photometric (in-place, BGR uint8) ----------------------
    def _photometric(self, img: np.ndarray, strength: str) -> np.ndarray:
        if strength == "minimal":
            jb, jc, ns = 0.05, 0.05, 0.0
        elif strength == "light":
            jb, jc, ns = 0.15, 0.15, 0.005
        else:  # strong
            jb, jc, ns = 0.3, 0.3, 0.01
        # Brightness
        if jb > 0:
            delta = (random.uniform(-jb, jb) * 255)
            img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        # Contrast
        if jc > 0:
            f = 1.0 + random.uniform(-jc, jc)
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = np.clip((img.astype(np.float32) - mean) * f + mean,
                          0, 255).astype(np.uint8)
        # Noise (Gaussian in pixel-units)
        if ns > 0:
            noise = np.random.normal(0, ns * 255, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    # ------------- entry point --------------------------------------------
    def __call__(self, img_bgr: np.ndarray, source: str = "synth",
                 person_mask: np.ndarray | None = None,
                 strong: bool = True) -> np.ndarray:
        """If `strong=False`, apply the FixMatch *weak* tier (light photometric
        only) so the teacher / anchor sees a clean image while the student
        gets the full F1+F2+FDA pipeline."""
        if not strong:
            wcfg = self.cfg_weak.get(source, self.cfg_weak["synth"])
            if random.random() < wcfg["p_photo"]:
                img_bgr = self._photometric(img_bgr, wcfg["photo_strength"])
            return img_bgr
        cfg = self.cfg.get(source, self.cfg["synth"])

        # F2 BG composite (synth only — egoexo bg is real)
        if cfg["p_bg"] > 0 and self.bg_corpus and person_mask is not None \
                and random.random() < cfg["p_bg"]:
            try:
                img_bgr = composite_on_real_bg(img_bgr, person_mask, self.bg_corpus)
            except Exception:
                pass

        # F1 occluder paste
        if cfg["p_occ"] > 0 and self.occluders and random.random() < cfg["p_occ"]:
            try:
                k = random.randint(1, self.max_occluders)
                img_bgr = occlude_with_objects(img_bgr, self.occluders,
                                               max_occluders=k)
            except Exception:
                pass

        # FDA: convert BGR->RGB for fda call, then back
        if cfg["p_fda"] > 0 and self.fda_refs and random.random() < cfg["p_fda"]:
            try:
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                rgb = apply_fda(rgb, self.fda_refs)
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                pass

        # Photometric (always last, applies to occluders too)
        if random.random() < cfg["p_photo"]:
            img_bgr = self._photometric(img_bgr, cfg["photo_strength"])

        return img_bgr


def build_default(asset_root: Path | None = None) -> Sim2RealAug | None:
    """Best-effort factory: returns Sim2RealAug pointed at standard paths,
    or None if any required corpus is missing."""
    asset_root = asset_root or HERE.parent / "assets" / "sim2real_refs"
    occ = asset_root / "occluders"
    bg  = asset_root / "bg"
    fda = asset_root / "fda"
    if not occ.exists() or len(list(occ.glob("*"))[:1]) == 0:
        print(f"[augment] no occluders at {occ} — augmentation disabled")
        return None
    return Sim2RealAug(occluders_dir=occ, bg_dir=bg, fda_dir=fda)


if __name__ == "__main__":
    """Smoke: load (if corpus present) and apply to one synth frame."""
    from PIL import Image
    aug = build_default()
    if aug is None:
        print("[smoke] no corpus locally — Vast will load it; smoke skipped")
        sys.exit(0)
    test_img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    for src in ("synth", "egoexo", "replay"):
        out = aug(test_img.copy(), source=src)
        print(f"  {src}: in={test_img.shape} out={out.shape} "
              f"different={not np.array_equal(test_img, out)}")
    print("[smoke] augment.py OK")
