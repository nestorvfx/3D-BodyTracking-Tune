"""F1 (occluder pasting) + F2 (BG compositing + FDA) sim-to-real ops.

This module replaces the previous Albumentations-FDA path because
Albumentations 2.x requires per-call `fda_metadata` kwargs (the legacy
`reference_images=` constructor arg is now ignored at call time).  We
instead invoke the standalone `albumentations.fourier_domain_adaptation`
numpy function with a per-call sampled reference.


These are pure-numpy transforms applied OUTSIDE the Albumentations pipeline,
because they need either:
  (a) per-sample side-data (a person matte for compositing), which doesn't
      fit Albumentations' single-image model cleanly, or
  (b) a randomised number of pasted RGBA cutouts at random positions/scales,
      which is awkward as an Albumentations ImageOnlyTransform.

Order in the training pipeline (see data.py _transform):
  geometric -> BG composite -> occluder paste -> photometric (incl. FDA)

Why that order:
  - BG-composite first so subsequent photometric augs (color jitter, gamma,
    blur, JPEG) operate on the FULL composite — unifies fg+bg colorimetry
    and softens the cut-paste seam.
  - Occluder paste BEFORE photometric so the occluders also pick up the
    same color/JPEG/blur — they look like real objects in the scene rather
    than pasted-on stickers.

References:
  - F1: I. Sárándi et al., "How Robust is 3D HPE to Occlusion?" (IROS'18)
        + ECCV'18 PoseTrack 3D-pose challenge winner
        github.com/isarandi/synthetic-occlusion (we faithfully port the
        load_occluders + occlude_with_objects + paste_over algorithm).
  - F2 (FDA): Yang & Soatto, "FDA: Fourier Domain Adaptation" (CVPR 2020).
        Uses Albumentations' built-in A.FDA with a real-image reference set.
  - F2 (BG composite): BEDLAM-CLIFF training pipeline composites synth
        humans onto random COCO/Places backgrounds; we do the same with
        commercial-clean refs (the user's own captured footage by default,
        easy to swap to OpenImages-CC for production scale).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# F1: Occluder pasting (Sárándi 2018, ECCV PoseTrack 2018 winner)
# -----------------------------------------------------------------------------

def load_occluders_from_dir(occluder_dir: str | Path,
                             min_alpha_pixels: int = 200) -> list[np.ndarray]:
    """Load all RGBA PNGs from `occluder_dir` and return them as numpy arrays.

    Each occluder is shape [h, w, 4], dtype=uint8, RGBA ordering (cv2's
    default BGRA is converted).  Cutouts smaller than `min_alpha_pixels`
    non-zero pixels are dropped.

    Edges are softened in the on-disk PNGs already (Sárándi's loader erodes
    by 8-px ellipse, where eroded < original alpha is set to 192).

    The data.py training pipeline operates on RGB images (post BGR2RGB
    conversion right after imread), so we must return RGBA — pasting a
    BGRA cutout onto an RGB image would visibly swap red and blue.
    """
    p = Path(occluder_dir)
    if not p.exists():
        return []
    out: list[np.ndarray] = []
    for png in sorted(p.glob("*.png")):
        bgra = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.ndim != 3 or bgra.shape[-1] != 4:
            continue
        if int((bgra[..., 3] > 8).sum()) < min_alpha_pixels:
            continue
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        out.append(rgba)
    return out


def _resize_rgba(rgba: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return rgba
    h, w = rgba.shape[:2]
    new_w = max(2, int(round(w * factor)))
    new_h = max(2, int(round(h * factor)))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(rgba, (new_w, new_h), interpolation=interp)


def _paste_rgba(im_dst: np.ndarray, occ_rgba: np.ndarray,
                 center_xy: tuple[float, float]) -> None:
    """Alpha-paste `occ_rgba` (HxWx4 BGRA or RGBA, must match `im_dst` colour
    convention in the first 3 channels) onto `im_dst` (HxWx3) IN PLACE,
    with `center_xy` being the centre pixel in im_dst.  Out-of-bounds parts
    are clipped (only the visible portion gets pasted).

    This faithfully ports Sárándi's `paste_over` so that visual results
    reproduce ECCV-PoseTrack-2018 numbers.
    """
    H_dst, W_dst = im_dst.shape[:2]
    h_src, w_src = occ_rgba.shape[:2]

    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    raw_x0, raw_y0 = cx - w_src // 2, cy - h_src // 2
    raw_x1, raw_y1 = raw_x0 + w_src, raw_y0 + h_src

    x0 = max(0, raw_x0)
    y0 = max(0, raw_y0)
    x1 = min(W_dst, raw_x1)
    y1 = min(H_dst, raw_y1)
    if x1 <= x0 or y1 <= y0:
        return  # entirely off-image

    sx0 = x0 - raw_x0
    sy0 = y0 - raw_y0
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    region_dst = im_dst[y0:y1, x0:x1]
    region_src = occ_rgba[sy0:sy1, sx0:sx1]
    rgb_src = region_src[..., :3]
    alpha = (region_src[..., 3:4].astype(np.float32) / 255.0)
    out = alpha * rgb_src.astype(np.float32) + (1.0 - alpha) * region_dst.astype(np.float32)
    im_dst[y0:y1, x0:x1] = out.astype(np.uint8)


def occlude_with_objects(
    img: np.ndarray,                         # [H, W, 3] uint8
    occluders: Sequence[np.ndarray],         # list of [h, w, 4] uint8
    *,
    rng: random.Random | None = None,
    n_holes_range: tuple[int, int] = (1, 8),
    scale_range: tuple[float, float] = (0.2, 1.0),
    rotate_prob: float = 0.5,
    flip_prob: float = 0.5,
) -> np.ndarray:
    """Sárándi-style occluder pasting.  Returns a NEW image; caller's input
    is not modified.

    `n_holes_range` matches the original (1..7 inclusive in the paper, the
    random.randint(1, 8) call returns 1..7).  `scale_range` is also from
    the original.

    Each pasted occluder is independently jittered by:
      - random scale in `scale_range` × (min(W,H) / 256)
      - random in-plane rotation U(-180°, 180°) with probability `rotate_prob`
      - random horizontal flip with probability `flip_prob`
    """
    if not occluders:
        return img
    rng = rng or random
    H, W = img.shape[:2]
    out = img.copy()
    base_scale = min(W, H) / 256.0
    n = rng.randint(n_holes_range[0], n_holes_range[1] - 1)
    for _ in range(n):
        occ = rng.choice(occluders).copy()
        if rng.random() < flip_prob:
            occ = occ[:, ::-1].copy()
        if rng.random() < rotate_prob:
            ang = rng.uniform(-180.0, 180.0)
            ho, wo = occ.shape[:2]
            M = cv2.getRotationMatrix2D((wo * 0.5, ho * 0.5), ang, 1.0)
            occ = cv2.warpAffine(
                occ, M, (wo, ho),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0))
        s = rng.uniform(*scale_range) * base_scale
        occ = _resize_rgba(occ, s)
        cx = rng.uniform(0, W)
        cy = rng.uniform(0, H)
        _paste_rgba(out, occ, (cx, cy))
    return out


# -----------------------------------------------------------------------------
# F2b: Background compositing using a person matte
# -----------------------------------------------------------------------------

def load_bg_corpus(bg_dir: str | Path) -> list[np.ndarray]:
    """Load all images from `bg_dir` as a list of HxWx3 uint8 RGB arrays.

    cv2.imread returns BGR; we convert to RGB so that compositing matches
    the data-pipeline convention (post-BGR2RGB synth crop).

    Caller is responsible for ensuring these are person-free (or near so)
    if they want clean compositing.  See _iter_compute_mattes.py for the
    auto-extraction pipeline that filters by selfie-segmentation coverage.
    """
    p = Path(bg_dir)
    if not p.exists():
        return []
    out: list[np.ndarray] = []
    for img_p in sorted(list(p.glob("*.jpg")) + list(p.glob("*.png"))):
        bg = cv2.imread(str(img_p))
        if bg is None:
            continue
        out.append(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    return out


def _random_crop_to(bg: np.ndarray, target_wh: tuple[int, int],
                    rng: random.Random) -> np.ndarray:
    """Random spatial crop of `bg` to (target_w, target_h).  If the bg is
    too small along either axis, upscale-with-aspect first."""
    Tw, Th = target_wh
    H, W = bg.shape[:2]
    # Upscale if needed.  Match the smaller-axis shortfall and keep aspect.
    sx = Tw / W
    sy = Th / H
    s = max(sx, sy, 1.0)
    if s > 1.0:
        bg = cv2.resize(bg, (int(np.ceil(W * s)), int(np.ceil(H * s))),
                         interpolation=cv2.INTER_LINEAR)
        H, W = bg.shape[:2]
    x = rng.randint(0, max(0, W - Tw))
    y = rng.randint(0, max(0, H - Th))
    return bg[y:y + Th, x:x + Tw].copy()


def composite_on_real_bg(
    img: np.ndarray,                  # [H, W, 3] uint8 (BGR or RGB; ANY,
                                       # the matte is colour-agnostic)
    matte: np.ndarray,                # [H, W] uint8, 0=bg, 255=fg
    bg_corpus: Sequence[np.ndarray], # list of HxWx3 uint8 (same colour as img)
    *,
    rng: random.Random | None = None,
    histogram_match: bool = True,
) -> np.ndarray:
    """Composite the foreground (where matte > 0) onto a random crop of a
    bg-corpus image.

    out = (alpha * fg + (1 - alpha) * bg)  with alpha = matte / 255.

    If `histogram_match=True`, the bg's per-channel mean is shifted
    toward the FG's per-channel mean (computed in the matte > 128 region)
    by a small fraction (10%).  This eliminates the most jarring
    illumination mismatches without distorting the bg appearance.
    """
    if matte is None or not bg_corpus:
        return img
    rng = rng or random
    H, W = img.shape[:2]
    bg = rng.choice(bg_corpus)
    bg = _random_crop_to(bg, (W, H), rng)

    if histogram_match:
        m_hard = (matte > 128)
        if m_hard.any():
            fg_mean = img[m_hard].mean(axis=0)        # [3]
            bg_mean = bg.reshape(-1, 3).mean(axis=0)   # [3]
            shift = 0.1 * (fg_mean - bg_mean)
            bg = np.clip(bg.astype(np.float32) + shift, 0, 255).astype(np.uint8)

    a = (matte.astype(np.float32) / 255.0)[..., None]   # [H,W,1]
    out = a * img.astype(np.float32) + (1.0 - a) * bg.astype(np.float32)
    return out.astype(np.uint8)


# -----------------------------------------------------------------------------
# F2a: FDA — handled inside the Albumentations Compose, see
# build_sim2real_aug() in augmentation.py.  The reference-image loader
# is here so all sim2real-corpus loading lives in one module.
# -----------------------------------------------------------------------------

def apply_fda(img_rgb: np.ndarray,
              fda_refs: Sequence[np.ndarray],
              *,
              rng: random.Random | None = None,
              beta_range: tuple[float, float] = (0.002, 0.015),
              ) -> np.ndarray:
    """Fourier Domain Adaptation (Yang & Soatto, CVPR 2020).

    Replaces the low-frequency amplitude of `img_rgb`'s spectrum with the
    one from a randomly chosen reference image.  Same shape required.

    `beta_range`: fraction of the spectrum to swap.  We swept β on actual
    synth crops and observed:
      β ∈ [0.002, 0.015]  →  subtle realistic warm/cool tone shifts
      β ∈ [0.02, 0.04]    →  visible colour shifts, edges of "rainbow"
      β >= 0.05           →  heavy multi-band rainbow patches, image
                              becomes unusable.
    Yang & Soatto's GTA5→Cityscapes experiments used β≈0.005–0.01;
    Albumentations' default (0, 0.1) sits at the destructive end.  We
    pick (0.002, 0.015) — clearly inside the safe zone with non-zero
    floor so the call has effect when sampled.  Caller controls
    application probability separately.
    """
    if not fda_refs:
        return img_rgb
    rng = rng or random
    # Albumentations 2.x exports the standalone numpy implementation that
    # A.FDA wraps, but only via the submodule.  Same algorithm
    # Yang & Soatto, CVPR 2020 used.
    from albumentations.augmentations.mixing.domain_adaptation_functional import (
        fourier_domain_adaptation,
    )
    ref = rng.choice(fda_refs)
    if ref.shape != img_rgb.shape:
        ref = cv2.resize(ref, (img_rgb.shape[1], img_rgb.shape[0]),
                         interpolation=cv2.INTER_AREA)
    beta = rng.uniform(*beta_range)
    out = fourier_domain_adaptation(img_rgb, ref, beta)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_fda_refs(fda_dir: str | Path,
                   target_wh: tuple[int, int]) -> list[np.ndarray]:
    """Load all images in `fda_dir`, resize to target_wh (W, H), return as
    a list of HxWx3 uint8 RGB arrays.  FDA needs source/target same shape;
    Albumentations applies its color-space mode (RGB) to refs and source
    consistently.

    Caller passes these into A.FDA(reference_images=..., beta_limit=...).
    """
    p = Path(fda_dir)
    if not p.exists():
        return []
    out: list[np.ndarray] = []
    Tw, Th = target_wh
    for img_p in sorted(list(p.glob("*.jpg")) + list(p.glob("*.png"))):
        ref = cv2.imread(str(img_p))
        if ref is None:
            continue
        if ref.shape[:2] != (Th, Tw):
            ref = cv2.resize(ref, (Tw, Th), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    return out
