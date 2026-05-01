"""One-off: compute (a) per-image person mattes for the iter synth dataset,
(b) person-free background crops from assets/testvideo.mp4.

Uses MediaPipe PoseLandmarker (heavy) with output_segmentation_masks=True —
the same model the user already ships under assets/.  Apache-2.0.

Outputs:
  dataset/output/synth_iter/mattes/<id>.png  — single-channel uint8 alpha
  assets/sim2real_refs/bg/  — overwritten with person-free crops
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = Path(r"c:/Users/Mihajlo/Documents/Body Tracking")
ITER_DIR = ROOT / "dataset" / "output" / "synth_iter"
MATTES_DIR = ITER_DIR / "mattes"
TEST_VIDEO = ROOT / "assets" / "testvideo.mp4"
BG_OUT = ROOT / "assets" / "sim2real_refs" / "bg"
POSE_MODEL = ROOT / "assets" / "pose_landmarker_heavy.task"

CROP_W, CROP_H = 256, 192
MAX_PERSON_FRAC = 0.02         # <2% person pixels in a "person-free" bg crop
N_BG_FRAMES_TARGET = 240
TRIES_PER_FRAME = 12


def make_segmenter():
    base = mp_python.BaseOptions(model_asset_path=str(POSE_MODEL))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=True,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def get_person_mask(seg, frame_rgb: np.ndarray,
                    *, soft: bool = False) -> np.ndarray:
    """Returns uint8 [H,W] mask where 255 = person, 0 = bg.  If no person
    detected, returns all-zeros.

    `soft=True` keeps MediaPipe's confidence ramp (used for matte
    computation where we want a 1-pixel anti-alias edge rather than a
    chunky stair-stepped one).  `soft=False` does a hard threshold (used
    for filtering bg crops by person-coverage)."""
    H, W = frame_rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    res = seg.detect(mp_img)
    if not res.segmentation_masks:
        return np.zeros((H, W), dtype=np.uint8)
    m_np = res.segmentation_masks[0].numpy_view()
    # MediaPipe returns shape (H, W, 1); squeeze to (H, W) for cv2 ops.
    if m_np.ndim == 3:
        m_np = m_np[..., 0]
    if soft:
        return np.clip(m_np * 255.0, 0, 255).astype(np.uint8)
    return ((m_np > 0.5) * 255).astype(np.uint8)


def compute_synth_mattes(seg, force: bool = False) -> int:
    """For each iter image, compute a person matte and save as PNG.

    Strategy that minimises silhouette ghosting in compositing:
      1. Keep MediaPipe's soft confidence (no hard threshold).
      2. Erode the >128 region by 1 pixel so the silhouette is pushed
         INWARD by ~1px — this prevents the synth's HDRI-bg-tinted
         half-alpha edge pixels from being included in "person" (they
         become bg, taking from the real-bg corpus instead, which
         eliminates the "halo" artefact from the v1 mattes).
      3. 1-pixel Gaussian on the result for clean anti-alias.
    """
    MATTES_DIR.mkdir(parents=True, exist_ok=True)
    labels = ITER_DIR / "labels.jsonl"
    n_done = 0
    n_empty = 0
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    with labels.open() as fh:
        for line in fh:
            rec = json.loads(line)
            img_path = ITER_DIR / rec["image_rel"]
            out_path = MATTES_DIR / f"{rec['id']}.png"
            if out_path.exists() and not force:
                n_done += 1
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            soft = get_person_mask(seg, img_rgb, soft=True)
            if soft.sum() == 0:
                n_empty += 1
                cv2.imwrite(str(out_path), soft)
                n_done += 1
                continue
            # Step 2: erode the high-confidence core by 1 px.
            core = (soft > 128).astype(np.uint8) * 255
            core_eroded = cv2.erode(core, erode_kernel, iterations=1)
            # Where the soft confidence said "definitely person" but the
            # erosion clipped it, restore at half intensity (avoids losing
            # thin limbs).  Where the eroded core kept the pixel, take
            # the soft confidence (which is already nicely ramped).
            hybrid = np.minimum(soft, core_eroded.astype(np.uint16) +
                                  ((core - core_eroded) // 2)).astype(np.uint8)
            # Step 3: 1-px Gaussian for anti-alias on the silhouette.
            matte = cv2.GaussianBlur(hybrid, (3, 3), 0.7)
            cv2.imwrite(str(out_path), matte)
            n_done += 1
            if n_done % 200 == 0:
                print(f"  mattes: {n_done} (empty={n_empty})")
    print(f"mattes done: {n_done} total ({n_empty} had no detected person)")
    return n_done


def harvest_bg_crops(seg) -> int:
    """Sweep the test video, for each frame try to find a person-free 256x192
    crop and save as bg ref."""
    BG_OUT.mkdir(parents=True, exist_ok=True)
    # Wipe old bg/ contents (which had the person in them).
    for old in BG_OUT.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("[error] cannot open testvideo")
        return 0
    rng = np.random.default_rng(42)
    written = 0
    seen = 0
    stride = max(1, total // (N_BG_FRAMES_TARGET * 4))
    for fi in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        seen += 1
        # Resize long edge to 720 to keep storage moderate (this is the bg
        # the synth person will composite on, so 720p is plenty).
        H, W = frame_bgr.shape[:2]
        if max(H, W) > 720:
            f = 720.0 / max(H, W)
            frame_bgr = cv2.resize(
                frame_bgr, (int(W * f), int(H * f)),
                interpolation=cv2.INTER_AREA)
            H, W = frame_bgr.shape[:2]
        if H < CROP_H or W < CROP_W:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = get_person_mask(seg, frame_rgb)

        # Find a 256x192 rect with <MAX_PERSON_FRAC person pixels.
        thresh_pix = int(MAX_PERSON_FRAC * CROP_W * CROP_H)
        found = False
        for _ in range(TRIES_PER_FRAME):
            x = int(rng.integers(0, W - CROP_W + 1))
            y = int(rng.integers(0, H - CROP_H + 1))
            sub = mask[y:y + CROP_H, x:x + CROP_W]
            if int((sub > 127).sum()) <= thresh_pix:
                crop = frame_bgr[y:y + CROP_H, x:x + CROP_W]
                cv2.imwrite(str(BG_OUT / f"bg_{written:04d}.jpg"),
                            crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                written += 1
                found = True
                break
        if written >= N_BG_FRAMES_TARGET:
            break
        if seen % 50 == 0:
            print(f"  bg: seen {seen}, written {written}")
    cap.release()
    print(f"bg crops done: {written} written from {seen} frames seen")
    return written


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    if not POSE_MODEL.exists():
        print(f"[error] pose model not found: {POSE_MODEL}")
        return 1
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mattes-only", action="store_true")
    p.add_argument("--bg-only", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing mattes (use after changing the algo)")
    args = p.parse_args()

    seg = make_segmenter()
    if not args.bg_only:
        print("=== computing synth person mattes ===")
        compute_synth_mattes(seg, force=args.force)
    if not args.mattes_only:
        print()
        print("=== harvesting person-free bg crops ===")
        harvest_bg_crops(seg)
    seg.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
