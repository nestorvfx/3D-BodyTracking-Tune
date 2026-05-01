#!/bin/bash
# ----------------------------------------------------------------------------
# Open Images V7 — sim-to-real corpus download (occluders + bg crops).
#
# All assets here are commercial-clean per Open Images V7 licence:
#   - Images:      CC BY 2.0  (Flickr-sourced photos chosen by photographers)
#   - Annotations: Apache-2.0 (Google)
# Reference: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
#
# What this downloads:
#   - validation-annotations-object-segmentation.csv  (~50 MB)
#       Per-instance metadata: MaskPath, ImageID, LabelName (MID), BBox.
#   - oidv7-class-descriptions-boxable.csv            (~20 KB)
#       MID → human-readable class name.
#   - oidv7-val-annotations-human-imagelabels.csv     (~5 MB)
#       Image-level labels (human-verified).  Used to filter person-free
#       images for the BG corpus.
#   - validation-masks/validation-masks-{0..f}.zip    (~60 MB total, 16 files)
#       Binary PNG masks, one per object instance.
#
# The actual photographer images are pulled by _iter_extract_openimages.py
# via direct S3 GET (open-images-dataset bucket, anonymous read).  We only
# download the images we need (those with non-person seg masks); typically
# ~5-10k images, ~1-2 GB.
#
# Idempotent: re-running skips files that already exist with correct size.
# ----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/assets/openimages_raw"
mkdir -p "${OUT}/masks"

log() { echo -e "\n\033[1;34m[oi-setup]\033[0m $*"; }

# Helper: download $1 to $2 only if $2 doesn't already exist with non-zero size.
fetch() {
    local url="$1"
    local dst="$2"
    if [ -s "${dst}" ]; then
        log "skip (exists): ${dst##*/}"
    else
        log "fetching ${dst##*/}"
        curl --fail --location --silent --show-error --output "${dst}" "${url}"
    fi
}

# ---------------------------------------------------------------------------
# 1. Annotation CSVs.
# ---------------------------------------------------------------------------
fetch "https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv" \
      "${OUT}/validation-annotations-object-segmentation.csv"

fetch "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv" \
      "${OUT}/oidv7-class-descriptions-boxable.csv"

fetch "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv" \
      "${OUT}/oidv7-val-annotations-human-imagelabels.csv"

# ---------------------------------------------------------------------------
# 2. Validation mask ZIPs (16 files, sharded by hex prefix of ImageID).
# ---------------------------------------------------------------------------
for HEX in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
    fetch "https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-${HEX}.zip" \
          "${OUT}/masks/validation-masks-${HEX}.zip"
done

# ---------------------------------------------------------------------------
# 3. Extract masks (idempotent — unzip -n skips existing).
# ---------------------------------------------------------------------------
log "extracting mask ZIPs (skipping any already extracted)"
for HEX in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
    unzip -q -n "${OUT}/masks/validation-masks-${HEX}.zip" -d "${OUT}/masks/" || true
done

log "DONE — annotations + masks at ${OUT}"
log "next: python training/_iter_extract_openimages.py"
