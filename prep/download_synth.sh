#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Download the 148,543-record synthetic body-pose corpus from HuggingFace.
# Repo: nestorvfx/3DBodyTrackingDatabase  (single clean.zip, 10.70 GB)
#
# The HF release ships:
#   - data/clean.zip — single zip with images/<id>.png (148,543) + labels.jsonl
#   - Already deduplicated, depth-filtered (Z ∈ [0.3, 50] m), bbox-filtered (>16 px)
#   - Train (~89.8 %) and val (~10.2 %) splits both inside via sha1-hash split field
#
# Steps:
#   1. hf download data/clean.zip
#   2. unzip flat to OUT_DIR/  (yields images/ + labels.jsonl)
#   3. quick sanity check (record count, image count match)
#
# Env:
#   HF_TOKEN  — required, HuggingFace read token
#   OUT_DIR   — defaults to /data/synth
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

OUT_DIR="${OUT_DIR:-/data/synth}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var required}"

log() { echo -e "\n\033[1;34m[synth-dl]\033[0m $*"; }

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# ── 0. Make sure hf-transfer is enabled (multi-Gbps HF downloads) ───────
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q -U "huggingface_hub[hf_xet]" hf_transfer >/dev/null 2>&1 || true

# ── 1. Download the single clean.zip (~10.7 GB) ────────────────────────
log "downloading data/clean.zip from nestorvfx/3DBodyTrackingDatabase"
HF_TOKEN="$HF_TOKEN" hf download nestorvfx/3DBodyTrackingDatabase \
    data/clean.zip --repo-type dataset --local-dir .

ZIP_PATH="$OUT_DIR/data/clean.zip"
[ -f "$ZIP_PATH" ] || { echo "[fatal] clean.zip not found at $ZIP_PATH"; exit 1; }
zsize=$(du -h "$ZIP_PATH" | cut -f1)
log "downloaded: $ZIP_PATH ($zsize)"

# ── 2. Verify zip integrity ────────────────────────────────────────────
log "verifying zip integrity"
unzip -tq "$ZIP_PATH" >/dev/null || { echo "[fatal] clean.zip is corrupt"; exit 1; }

# ── 3. Extract — single file, single pass ──────────────────────────────
log "extracting (~12 GB; 148,543 images + labels.jsonl)"
unzip -q -o "$ZIP_PATH" -d .

# Sanity: image count + labels.jsonl present
N_PNG=$(find images -maxdepth 1 -name "*.png" -type f | wc -l)
[ -f labels.jsonl ] || { echo "[fatal] labels.jsonl missing in clean.zip"; exit 1; }
N_LBL=$(wc -l < labels.jsonl)
log "extracted: $N_PNG PNGs in images/, $N_LBL records in labels.jsonl"
[ "$N_PNG" -eq "$N_LBL" ] || \
    { echo "[warn] PNG count $N_PNG ≠ labels count $N_LBL"; }

# ── 4. Cleanup zip after successful extract ────────────────────────────
log "cleanup: removing source zip (extract verified)"
rm -f "$ZIP_PATH"
rmdir data 2>/dev/null || true

# ── 5. Quick split-distribution check ──────────────────────────────────
log "split distribution"
python3 - <<'PY'
import json
from collections import Counter
counts = Counter()
with open("labels.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        counts[rec.get("split", "?")] += 1
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")
PY

log "DONE — synth corpus ready at $OUT_DIR/  ($N_LBL records, both splits)"
