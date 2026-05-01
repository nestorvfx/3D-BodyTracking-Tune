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
        # Sárándi-2024 update: also load human-shaped occluders if present.
        # These are MORE punishing for pose models (similar texture statistics
        # to the target person, blurs the foreground/background separation
        # that pure-object occluders make easy).
        human_dir = occ_dir.parent / "occluders_human"
        if human_dir.exists():
            try:
                hu = load_occluders_from_dir(human_dir)
                self.occluders.extend(hu)
                print(f"[augment] +{len(hu)} human-shaped occluders (Sárándi-2024)")
            except Exception as e:
                print(f"[augment] human occluders skipped: {e}")
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
        # Sárándi/RTMPose/DWPose/FixMatch lineage, 2024-2026).
        self.cfg = {
            "synth":  {"p_bg": 0.9, "p_occ": 0.8, "p_fda": 0.5,
                       "p_photo": 1.0, "photo_strength": "strong",
                       "p_jpeg": 0.30, "p_blur": 0.40},
            "egoexo": {"p_bg": 0.0, "p_occ": 0.4, "p_fda": 0.0,
                       "p_photo": 0.5, "photo_strength": "light",
                       "p_jpeg": 0.15, "p_blur": 0.15},
            "replay": {"p_bg": 0.0, "p_occ": 0.1, "p_fda": 0.0,
                       "p_photo": 0.3, "photo_strength": "minimal",
                       "p_jpeg": 0.0,  "p_blur": 0.05},
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
            jb, jc, js, jh, ns = 0.05, 0.05, 0.05, 0.02, 0.0
        elif strength == "light":
            jb, jc, js, jh, ns = 0.15, 0.15, 0.10, 0.03, 0.005
        else:  # strong
            jb, jc, js, jh, ns = 0.30, 0.30, 0.15, 0.05, 0.01
        # Brightness
        if jb > 0:
            delta = random.uniform(-jb, jb) * 255
            img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        # Contrast
        if jc > 0:
            f = 1.0 + random.uniform(-jc, jc)
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = np.clip((img.astype(np.float32) - mean) * f + mean,
                          0, 255).astype(np.uint8)
        # Saturation + hue (HSV space)
        if js > 0 or jh > 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            if jh > 0:
                hsv[..., 0] = (hsv[..., 0] + random.uniform(-jh, jh) * 180) % 180
            if js > 0:
                hsv[..., 1] = np.clip(
                    hsv[..., 1] * (1.0 + random.uniform(-js, js)), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Gaussian noise
        if ns > 0:
            noise = np.random.normal(0, ns * 255, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    # ------------- motion blur (synth has none, real does) ---------------
    @staticmethod
    def _motion_blur(img: np.ndarray, k_min: int = 3, k_max: int = 7) -> np.ndarray:
        k = random.randint(k_min, k_max)
        if k % 2 == 0:
            k += 1
        kernel = np.zeros((k, k), dtype=np.float32)
        # Random direction
        angle = random.uniform(0, np.pi)
        cx, cy = k // 2, k // 2
        dx, dy = np.cos(angle), np.sin(angle)
        for i in range(k):
            x = int(cx + (i - k // 2) * dx)
            y = int(cy + (i - k // 2) * dy)
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1.0
        kernel /= kernel.sum() if kernel.sum() > 0 else 1.0
        return cv2.filter2D(img, -1, kernel)

    # ------------- JPEG quality jitter (real frames are re-encoded; synth isn't) ---
    @staticmethod
    def _jpeg_jitter(img: np.ndarray, q_min: int = 40, q_max: int = 95) -> np.ndarray:
        q = random.randint(q_min, q_max)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

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

        # Motion blur (real GoPros have it, synth is sharp — closes domain gap)
        if cfg.get("p_blur", 0) > 0 and random.random() < cfg["p_blur"]:
            img_bgr = self._motion_blur(img_bgr)

        # Photometric (always last so it applies to occluders too)
        if random.random() < cfg["p_photo"]:
            img_bgr = self._photometric(img_bgr, cfg["photo_strength"])

        # JPEG quality jitter (Ego-Exo4D is re-encoded h264→jpg; synth is
        # lossless PNG — easiest sim-real domain tell)
        if cfg.get("p_jpeg", 0) > 0 and random.random() < cfg["p_jpeg"]:
            img_bgr = self._jpeg_jitter(img_bgr)

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
