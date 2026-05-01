"""Open Images V7 → sim-to-real corpora (occluders + person-free bg crops).

Reads the assets/openimages_raw/* files produced by _iter_setup_openimages.sh,
fetches the images we actually need from the open-images-dataset S3 bucket
(anonymous-read public bucket), and produces:

  assets/sim2real_refs/occluders/  RGBA PNG cutouts (Sárándi 2018-style)
  assets/sim2real_refs/bg/         person-free 256x192 BGR JPG crops

Augmentation only — does NOT touch the synthetic dataset itself.

Faithfully reproduces Sárándi's "load_occluders" recipe:
  - skip person-related classes (16 MIDs covering Person/Man/Woman/Boy/
    Girl + 11 human-body-part classes — the Open Images analog of
    VOC's single "person" class)
  - skip too-small instances (< 500 alpha pixels)
  - edge-soften by 8x8 ellipse erosion: where eroded < orig, set α=192
  - downscale 0.5x for storage
The bg corpus filters images that have NO person-related image-level label
(image-level labels are human-verified at confidence ≥ 0.5).

Licensing:
  Open Images V7 images:        CC BY 2.0 (Flickr photographers)
  Open Images V7 annotations:   Apache-2.0 (Google)
We emit assets/sim2real_refs/ATTRIBUTION.txt with the required attribution
on first run.

Idempotent: skips already-downloaded images / already-extracted cutouts.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "openimages_raw"
IMG_CACHE = RAW / "images_validation"

OUT_OCC = ROOT / "assets" / "sim2real_refs" / "occluders"
OUT_BG = ROOT / "assets" / "sim2real_refs" / "bg"
ATTRIBUTION = ROOT / "assets" / "sim2real_refs" / "ATTRIBUTION.txt"

# Human-related class MIDs — exclude from occluder corpus (Sárándi convention).
# Verified against assets/openimages_raw/oidv7-class-descriptions-boxable.csv:
HUMAN_MIDS = {
    "/m/01g317",  # Person
    "/m/04yx4",   # Man
    "/m/03bt1vf", # Woman
    "/m/05r655",  # Girl
    "/m/01bl7v",  # Boy
    "/m/0dzct",   # Human face
    "/m/04hgtk",  # Human head
    "/m/0k0pj",   # Human nose
    "/m/0283dt1", # Human mouth
    "/m/039xj_",  # Human ear
    "/m/014sv8",  # Human eye
    "/m/02p0tk3", # Human body
    "/m/0dzf4",   # Human arm
    "/m/0k65p",   # Human hand
    "/m/031n1",   # Human foot
    "/m/035r7c",  # Human leg
}

# Sárándi parameters (verbatim from github.com/isarandi/synthetic-occlusion).
MIN_ALPHA_PIXELS = 500
EDGE_RIM_ALPHA = 192
ERODE_ELLIPSE = (8, 8)
OCC_DOWNSCALE = 0.5

# Bg crop parameters.
BG_CROP_WH = (256, 192)
BG_PER_IMAGE_MAX = 2

# S3 public bucket pattern (used by Google's official downloader.py via
# unsigned boto3; we hit the same URL directly with urllib to avoid the
# boto3 dependency).  Anonymous read is allowed by the bucket policy.
S3_URL_TEMPLATE = "https://s3.amazonaws.com/open-images-dataset/validation/{image_id}.jpg"


def emit_attribution() -> None:
    OUT_OCC.parent.mkdir(parents=True, exist_ok=True)
    if ATTRIBUTION.exists():
        return
    ATTRIBUTION.write_text(
        "Sim-to-real augmentation corpora derived from Open Images V7\n"
        "============================================================\n\n"
        "Images:        Creative Commons Attribution 2.0 license (CC BY 2.0)\n"
        "                — sourced from Flickr photographers via Open Images V7\n"
        "                — https://creativecommons.org/licenses/by/2.0/\n\n"
        "Annotations:   Apache License 2.0 (Google)\n"
        "                — Open Images V7 segmentation masks + image labels\n\n"
        "Required attribution string for downstream products:\n"
        '  "This product uses augmentation imagery derived from Open Images '
        'V7 (https://storage.googleapis.com/openimages/web/, images CC BY 2.0, '
        'annotations Apache-2.0)."\n\n'
        "Reference: https://storage.googleapis.com/openimages/web/factsfigures_v7.html\n"
    )


def load_seg_metadata(csv_path: Path) -> list[dict]:
    """Read masks_data.csv as a list of dicts."""
    out: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


def find_image_ids_with_human_label(labels_csv: Path,
                                     conf_threshold: float = 0.5) -> set[str]:
    """Return set of ImageIDs that have any positive human-related label
    (Person/Man/Woman/Boy/Girl/Human-*) at confidence >= threshold.
    These images are EXCLUDED from the bg corpus (we want person-free)."""
    ids: set[str] = set()
    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["LabelName"] not in HUMAN_MIDS:
                continue
            try:
                conf = float(row["Confidence"])
            except (KeyError, ValueError):
                continue
            if conf >= conf_threshold:
                ids.add(row["ImageID"])
    return ids


def download_image(image_id: str, dst: Path) -> bool:
    """Fetch a single OI validation image into dst.  Returns True on success.
    Idempotent: returns True immediately if dst exists and is non-empty.
    Uses urllib (no extra deps) with a 30s timeout and 2 retries."""
    if dst.exists() and dst.stat().st_size > 0:
        return True
    url = S3_URL_TEMPLATE.format(image_id=image_id)
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "BodyTracking/1.0 (+research)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            tmp = dst.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(dst)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            if attempt == 1:
                return False
            time.sleep(1.0)
    return False


def parallel_fetch(image_ids: Iterable[str], cache_dir: Path,
                   *, n_workers: int = 16, label: str = "fetch") -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_ids = list(image_ids)
    todo = [(iid, cache_dir / f"{iid}.jpg") for iid in image_ids
            if not (cache_dir / f"{iid}.jpg").exists()]
    if not todo:
        print(f"[{label}] all {len(image_ids)} already cached")
        return len(image_ids)
    print(f"[{label}] downloading {len(todo)} of {len(image_ids)} images "
          f"with {n_workers} threads")
    n_ok = 0
    n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        fut_to_id = {ex.submit(download_image, iid, dst): iid
                     for iid, dst in todo}
        for i, fut in enumerate(as_completed(fut_to_id)):
            ok = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            if (i + 1) % 200 == 0:
                rate = (i + 1) / max(0.01, time.time() - t0)
                print(f"  {i+1}/{len(todo)}  ok={n_ok} fail={n_fail}  "
                      f"({rate:.0f} img/s)")
    print(f"[{label}] done — {n_ok} ok, {n_fail} failed, "
          f"in {time.time() - t0:.1f}s")
    return n_ok


def extract_occluder(img_rgb: np.ndarray, mask_path: Path,
                      bbox_xyxy_norm: tuple[float, float, float, float]
                      ) -> np.ndarray | None:
    """Extract one Sárándi-style RGBA cutout.

    `img_rgb` is the FULL image (uint8 RGB).
    `mask_path` is a binary PNG (non-zero pixels = the object instance).
        The mask is at the FULL image resolution.
    `bbox_xyxy_norm` is the normalized starting bbox (XMin, YMin, XMax, YMax)
        from masks_data.csv — used to crop the relevant region.

    Returns RGBA uint8 [h, w, 4] or None if the instance is too small.
    """
    H, W = img_rgb.shape[:2]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != (H, W):
        # Open Images masks are stored at the image's native resolution;
        # if the cached image was downscaled (rare), align them.
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Mask should be binary uint8 (0 or 255).  Some files use 0/1 values.
    mask = (mask > 0).astype(np.uint8) * 255

    # Crop to the bbox supplied by the annotation, with a small padding so
    # we don't lose silhouette antialias at the edge.
    xmin, ymin, xmax, ymax = bbox_xyxy_norm
    pad = 0.01
    x0 = max(0, int(round((xmin - pad) * W)))
    y0 = max(0, int(round((ymin - pad) * H)))
    x1 = min(W, int(round((xmax + pad) * W)))
    y1 = min(H, int(round((ymax + pad) * H)))
    if x1 <= x0 or y1 <= y0:
        return None

    obj_mask = mask[y0:y1, x0:x1]
    if int(cv2.countNonZero(obj_mask)) < MIN_ALPHA_PIXELS:
        return None
    obj_im = img_rgb[y0:y1, x0:x1]

    # Sárándi edge-softening: erode by 8x8 ellipse, where eroded < orig
    # set alpha = 192.  Creates a half-transparent rim that hides the
    # cut-paste seam at composite time.
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ERODE_ELLIPSE)
    eroded = cv2.erode(obj_mask, se)
    obj_mask = obj_mask.copy()
    obj_mask[eroded < obj_mask] = EDGE_RIM_ALPHA

    rgba = np.concatenate([obj_im, obj_mask[..., None]], axis=-1)

    # Match Sárándi's storage choice (0.5x).
    if OCC_DOWNSCALE != 1.0:
        new_w = max(2, int(round(rgba.shape[1] * OCC_DOWNSCALE)))
        new_h = max(2, int(round(rgba.shape[0] * OCC_DOWNSCALE)))
        rgba = cv2.resize(rgba, (new_w, new_h),
                          interpolation=cv2.INTER_AREA)
    return rgba


def extract_bg_crops(img_bgr: np.ndarray, n_max: int,
                     rng: np.random.Generator) -> list[np.ndarray]:
    """Random 256x192 BGR crops from a person-free image."""
    H, W = img_bgr.shape[:2]
    Tw, Th = BG_CROP_WH
    if H < Th or W < Tw:
        # Upscale so the smallest axis fits.
        s = max(Th / H, Tw / W) * 1.05
        img_bgr = cv2.resize(img_bgr,
                             (int(np.ceil(W * s)), int(np.ceil(H * s))),
                             interpolation=cv2.INTER_LINEAR)
        H, W = img_bgr.shape[:2]
    crops: list[np.ndarray] = []
    for _ in range(n_max):
        x = int(rng.integers(0, W - Tw + 1))
        y = int(rng.integers(0, H - Th + 1))
        crops.append(img_bgr[y:y + Th, x:x + Tw].copy())
    return crops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-occluders", type=int, default=15000,
                    help="Cap on occluder cutouts produced")
    ap.add_argument("--max-bg", type=int, default=8000,
                    help="Cap on bg crops produced")
    ap.add_argument("--n-workers", type=int, default=16,
                    help="Parallel image downloaders")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing-occluders", action="store_true",
                    help="Don't redownload+re-extract already-present occluders")
    ap.add_argument("--human-occluders-only", action="store_true",
                    help="Extract human-shaped occluders (Sárándi-2024) into "
                         "occluders_human/ instead of objects into occluders/.")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    rng = np.random.default_rng(args.seed)
    OUT_OCC_DIR = OUT_OCC.parent / "occluders_human" if args.human_occluders_only else OUT_OCC
    OUT_OCC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_BG.mkdir(parents=True, exist_ok=True)
    IMG_CACHE.mkdir(parents=True, exist_ok=True)
    emit_attribution()

    # ---- 1. Load segmentation metadata, filter to non-human classes -------
    seg_csv = RAW / "validation-annotations-object-segmentation.csv"
    if not seg_csv.exists():
        print(f"[error] {seg_csv} missing — run _iter_setup_openimages.sh first")
        return 1
    print(f"[seg] reading {seg_csv}")
    seg_rows = load_seg_metadata(seg_csv)
    print(f"[seg] {len(seg_rows)} total instances")

    if args.human_occluders_only:
        seg_filtered = [r for r in seg_rows if r["LabelName"] in HUMAN_MIDS]
        print(f"[seg] {len(seg_filtered)} HUMAN instances (Sárándi-2024 mode)")
    else:
        seg_filtered = [r for r in seg_rows if r["LabelName"] not in HUMAN_MIDS]
        print(f"[seg] {len(seg_filtered)} non-human instances")
    seg_nonhuman = seg_filtered                                    # naming preserved

    rng.shuffle(seg_nonhuman)
    seg_nonhuman = seg_nonhuman[:args.max_occluders * 2]
    occluder_image_ids = sorted({r["ImageID"] for r in seg_nonhuman})
    print(f"[seg] need {len(occluder_image_ids)} unique images for occluders")

    # ---- 2. BG corpus: validation images with NO human label --------------
    labels_csv = RAW / "oidv7-val-annotations-human-imagelabels.csv"
    print(f"[bg] reading {labels_csv}")
    human_image_ids = find_image_ids_with_human_label(labels_csv)
    print(f"[bg] {len(human_image_ids)} val images carry a human label")
    # Build a candidate bg set: validation images that don't have human label.
    # We use the seg CSV's image IDs (a known-bounded set of ~13k images
    # in val) as our universe — they all have ≥1 seg mask which means
    # they're real annotated photos, not blank/error images.
    all_seg_ids = sorted({r["ImageID"] for r in seg_rows})
    bg_candidates = [iid for iid in all_seg_ids if iid not in human_image_ids]
    rng.shuffle(bg_candidates)
    bg_candidates = bg_candidates[:args.max_bg]
    print(f"[bg] {len(bg_candidates)} candidate bg images")

    # ---- 3. Download images we need (union of occluder + bg image sets) --
    needed_ids = set(occluder_image_ids) | set(bg_candidates)
    parallel_fetch(needed_ids, IMG_CACHE,
                   n_workers=args.n_workers, label="img-fetch")

    # ---- 4. Extract occluders (Sárándi-style) -----------------------------
    masks_dir = RAW / "masks"
    n_occ = 0
    n_skip_size = 0
    n_skip_load = 0
    print(f"[occ] extracting...")
    t0 = time.time()
    for row in seg_nonhuman:
        if n_occ >= args.max_occluders:
            break
        image_id = row["ImageID"]
        mask_name = row["MaskPath"]
        out_name = f"{Path(mask_name).stem}.png"
        out_path = OUT_OCC_DIR / out_name
        if args.skip_existing_occluders and out_path.exists():
            n_occ += 1
            continue
        img_path = IMG_CACHE / f"{image_id}.jpg"
        mask_path = masks_dir / mask_name
        if not img_path.exists() or not mask_path.exists():
            n_skip_load += 1
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            n_skip_load += 1
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bbox = (float(row["BoxXMin"]), float(row["BoxYMin"]),
                float(row["BoxXMax"]), float(row["BoxYMax"]))
        rgba = extract_occluder(img_rgb, mask_path, bbox)
        if rgba is None:
            n_skip_size += 1
            continue
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(out_path), bgra)
        n_occ += 1
        if n_occ % 500 == 0:
            print(f"  occ {n_occ}/{args.max_occluders}  "
                  f"(skip-size={n_skip_size} skip-load={n_skip_load})")
    print(f"[occ] {n_occ} cutouts in {time.time()-t0:.1f}s "
          f"(skip-size={n_skip_size} skip-load={n_skip_load})")

    # ---- 5. Extract bg crops ---------------------------------------------
    n_bg = 0
    print(f"[bg] extracting crops...")
    t0 = time.time()
    for image_id in bg_candidates:
        if n_bg >= args.max_bg:
            break
        img_path = IMG_CACHE / f"{image_id}.jpg"
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        crops = extract_bg_crops(img_bgr,
                                 n_max=BG_PER_IMAGE_MAX,
                                 rng=rng)
        for j, crop in enumerate(crops):
            if n_bg >= args.max_bg:
                break
            out_path = OUT_BG / f"oi_{image_id}_{j}.jpg"
            cv2.imwrite(str(out_path), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            n_bg += 1
        if n_bg % 500 == 0 and n_bg > 0:
            print(f"  bg {n_bg}/{args.max_bg}")
    print(f"[bg] {n_bg} crops in {time.time()-t0:.1f}s")

    print(f"\nDONE.")
    print(f"  occluders: {OUT_OCC}  ({n_occ} files)")
    print(f"  bg crops:  {OUT_BG}   ({n_bg} new files)")
    print(f"  attribution: {ATTRIBUTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
