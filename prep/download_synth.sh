#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Download the 500k synthetic body-pose corpus from HuggingFace.
# Repo: nestorvfx/3DBodyTrackingDatabase  (10 batches × ~4 GB)
#
# Steps:
#   1. curl 10 zips (3-at-a-time, resumable via curl -C -)
#   2. integrity-check each zip
#   3. flat-extract (cross-zip duplicates byte-identical → dedupe to ~149k)
#   4. build labels.jsonl from per-image JSONs (parallelised)
#   5. drop NaN-z and bbox<16px outliers
#
# Env:
#   HF_TOKEN  — required, HuggingFace read token
#   OUT_DIR   — defaults to /data/synth
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

OUT_DIR="${OUT_DIR:-/data/synth}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var required}"
HF_REPO="datasets/nestorvfx/3DBodyTrackingDatabase"

log() { echo -e "\n\033[1;34m[synth-dl]\033[0m $*"; }

mkdir -p "$OUT_DIR/data"
cd "$OUT_DIR"

log "downloading 10 zips to $OUT_DIR/data/"
ALL=(01 02 03 04 05 06 07 08 09 10)

for ((i=0; i<${#ALL[@]}; i+=3)); do
    GROUP=("${ALL[@]:i:3}")
    log "group: ${GROUP[*]}"
    for N in "${GROUP[@]}"; do
        (
            curl -L -C - -fsS -H "Authorization: Bearer $HF_TOKEN" \
                -o "data/batch_${N}.zip" \
                "https://huggingface.co/${HF_REPO}/resolve/main/data/batch_${N}.zip" \
            && echo "[done] batch_${N}" \
            || echo "[FAIL] batch_${N}"
        ) &
    done
    wait
done

# ── 2. integrity ──────────────────────────────────────────────────────
log "verifying zip integrity"
BAD=0
for z in data/batch_*.zip; do
    if unzip -tq "$z" >/dev/null 2>&1; then
        echo "  OK $z"
    else
        echo "  BAD $z"
        BAD=$((BAD + 1))
    fi
done
[ "$BAD" -eq 0 ] || { echo "[fatal] $BAD bad zips — re-run this script (curl -C - resumes)"; exit 1; }

# ── 3. flat-extract (dedupe via -o) ───────────────────────────────────
log "extracting (sequential, disk-pressure-safe)"
for z in data/batch_*.zip; do
    log "  $z"
    unzip -q -o "$z" -d .
    df -h "$OUT_DIR" | tail -1
done

N_PNG=$(find . -maxdepth 1 -name "*.png"  -type f | wc -l)
N_JSON=$(find . -maxdepth 1 -name "*.json" -type f | wc -l)
log "extracted: $N_PNG PNGs, $N_JSON JSONs (expect ~149,292 each)"

# ── 4. build labels.jsonl ─────────────────────────────────────────────
log "building labels.jsonl (parallel)"
python3 - <<'PY'
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

R = Path(".")
files = list(R.glob("*.json"))
print(f"  found {len(files)} JSON files")

def parse(p):
    try:
        rec = json.loads(p.read_text())
        rec["image_rel"] = f"{rec['id']}.png"
        rec.pop("mask_rel", None)
        rec.pop("depth_rel", None)
        if not (R / rec["image_rel"]).exists():
            return None
        return json.dumps(rec, separators=(",", ":"))
    except Exception:
        return None

n_ok = n_bad = 0
with open("labels.jsonl", "w") as out, ProcessPoolExecutor(max_workers=8) as ex:
    for line in ex.map(parse, files, chunksize=500):
        if line is None:
            n_bad += 1; continue
        out.write(line + "\n"); n_ok += 1
print(f"  labels.jsonl: {n_ok} written, {n_bad} skipped")
PY

# ── 5. drop outliers ──────────────────────────────────────────────────
log "filtering: drop NaN-z and bbox<16 px"
python3 - <<'PY'
import json, os, math
src, tmp = "labels.jsonl", "labels.jsonl.tmp"
kept = dropped = 0
with open(src) as fi, open(tmp, "w") as fo:
    for line in fi:
        rec = json.loads(line)
        bw, bh = rec["bbox_xywh"][2], rec["bbox_xywh"][3]
        if bw < 16 or bh < 16:
            dropped += 1; continue
        z = rec.get("root_joint_cam", [0, 0, 0])[-1]
        if not math.isfinite(z) or z < 0.3 or z > 50:
            dropped += 1; continue
        fo.write(line); kept += 1
os.replace(tmp, src)
print(f"  kept={kept} dropped={dropped}")
PY

# ── 6. cleanup zips after successful extract ─────────────────────────
log "cleanup: removing source zips (extract verified)"
rm -f data/batch_*.zip
rmdir data 2>/dev/null || true

log "DONE — synth corpus ready at $OUT_DIR/  ($(wc -l < labels.jsonl) records)"
