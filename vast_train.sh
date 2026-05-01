#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# End-to-end training orchestrator for Vast.ai.
#
# Stages:
#   1. Download synth corpus from HuggingFace → /data/synth     (~30 min, ~25 GB)
#   2. Download Ego-Exo4D train split via egoexo CLI            (~3-6 hr, ~400 GB transient)
#      + extract annotated frames per take, deleting videos after.
#   3. Pre-cache Heavy + (optional) Hand + Face teachers        (~3-6 hr GPU)
#   4. Train Lite v2 (10 epochs, ~7 hr A100)                    (~$11)
#   5. Train Full v2 (10 epochs, ~7 hr A100)                    (~$11)
#   6. Export both as .task                                      (~5 min)
#   7. Score against the SOTA benchmark                          (~5 min)
#
# Each stage is idempotent — re-running picks up where it left off.
#
# Required env (set before running):
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  (Ego-Exo4D bucket)
#   HF_TOKEN                                  (HuggingFace synth)
#
# Optional env:
#   STAGE=N        run only stage N
#   SKIP_HAND_FACE=1   skip caching Hand + Face teachers (body-only KD)
#   FAST=1         use --limit-* flags for a quick smoke pass
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

log() { echo -e "\n\033[1;36m[run]\033[0m $*"; }
die() { echo -e "\033[1;31m[run:fatal]\033[0m $*" >&2; exit 1; }

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

[ -f /workspace/.bptune_setup_done ] || die "run vast_setup.sh first"

# Stages -----------------------------------------------------------------
STAGE="${STAGE:-all}"

want_stage() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# 1. Synth -------------------------------------------------------------
if want_stage 1 && [ ! -f /data/synth/labels.jsonl ]; then
    log "stage 1: download + prepare synth corpus"
    bash prep/download_synth.sh
fi

# 2. Ego-Exo4D ---------------------------------------------------------
if want_stage 2; then
    if [ ! -d /data/egoexo/annotations ]; then
        log "stage 2a: pull Ego-Exo4D train annotations"
        egoexo -o /data/egoexo \
               --parts annotations metadata \
               --benchmarks egopose --splits train -y
    fi
    if [ ! -f /data/egoexo/manifest_train.jsonl ]; then
        log "stage 2b: stream-extract Ego-Exo4D train frames"
        EXTRA=()
        [ -n "${FAST:-}" ] && EXTRA+=(--limit-takes 50)
        python3 prep/extract_egoexo_train.py \
            --anno-root /data/egoexo/annotations \
            --raw-root /data/egoexo \
            --frames-root /data/egoexo/frames \
            --manifest-out /data/egoexo/manifest_train.jsonl \
            "${EXTRA[@]}"
    fi
fi

# 3. Teacher cache -----------------------------------------------------
if want_stage 3; then
    log "stage 3a: cache Heavy on synth"
    HAND_FACE_FLAGS=""
    [ -z "${SKIP_HAND_FACE:-}" ] && HAND_FACE_FLAGS="--include-hand --include-face"
    python3 training/cache_teachers.py \
        --source-dir /data/synth --labels-jsonl /data/synth/labels.jsonl \
        --out-dir /data/teacher_cache $HAND_FACE_FLAGS

    if [ -f /data/egoexo/manifest_train.jsonl ]; then
        log "stage 3b: cache Heavy on Ego-Exo4D"
        # cache_teachers walks images by source-dir; manifest-driven later if needed
        python3 training/cache_teachers.py \
            --source-dir /data/egoexo/frames \
            --out-dir /data/teacher_cache $HAND_FACE_FLAGS
    fi
fi

# 4. Lite training -----------------------------------------------------
if want_stage 4; then
    log "stage 4: train Lite v2"
    EXTRA=()
    [ -n "${FAST:-}" ] && EXTRA+=(--limit-synth 5000 --limit-egoexo 10000 --epochs 1)
    python3 training/train.py \
        --variant lite \
        --synth-root /data/synth \
        --egoexo-root /data/egoexo \
        --anno-root /data/egoexo/annotations \
        --teacher-cache /data/teacher_cache \
        --out-root /workspace/ckpts \
        --runs-root /workspace/runs \
        --epochs 20 --batch-size 96 --lr 5e-5 --bf16 --num-workers 4 \
        --ckpt-every-min 10 \
        "${EXTRA[@]}"
fi

# 5. Full training -----------------------------------------------------
if want_stage 5; then
    log "stage 5: train Full v2"
    EXTRA=()
    [ -n "${FAST:-}" ] && EXTRA+=(--limit-synth 5000 --limit-egoexo 10000 --epochs 1)
    python3 training/train.py \
        --variant full \
        --synth-root /data/synth \
        --egoexo-root /data/egoexo \
        --anno-root /data/egoexo/annotations \
        --teacher-cache /data/teacher_cache \
        --out-root /workspace/ckpts \
        --runs-root /workspace/runs \
        --epochs 20 --batch-size 96 --lr 5e-5 --bf16 --num-workers 4 \
        --ckpt-every-min 10 \
        "${EXTRA[@]}"
fi

# 6. Export ------------------------------------------------------------
# Export-time deps (litert-torch + tf-nightly) are intentionally NOT in the
# main requirements.txt because tf-nightly conflicts with PyTorch nightly
# cu128 during training.  Install them here, after training is done, just
# for the export step.  If pip resolution fails, model/export.py
# auto-falls-back to its custom flatbuffer rewriter (no external deps).
if want_stage 6; then
    log "stage 6: install export deps + export .task files"
    python3 -m pip install --quiet --pre \
        litert-torch tf-nightly || \
        log "[warn] converter install failed; relying on custom flatbuffer rewriter"
    python3 model/export.py --variant lite \
        --ckpt /workspace/ckpts/lite_final.pt \
        --out  /workspace/exports/lite_v2.task
    python3 model/export.py --variant full \
        --ckpt /workspace/ckpts/full_final.pt \
        --out  /workspace/exports/full_v2.task
fi

# 7. Benchmark ---------------------------------------------------------
if want_stage 7; then
    log "stage 7: benchmark vs Lite/Full/Heavy v1"
    # The benchmark tooling expects student .task files dropped into
    # benchmark/assets/.  For now, rerun with the v1 baseline scripts on
    # the v2 .task files (the user wires this up; benchmark/run_eval.py
    # has --variant; extending MODEL_PATHS is one line).
    log "  (manual step) extend benchmark/run_eval.py MODEL_PATHS to include "
    log "  student_v2_lite + student_v2_full pointing at /workspace/exports/*.task"
    log "  then: cd benchmark && python analyze.py --manual-only --variants lite full heavy student_v2_lite student_v2_full"
fi

log "DONE"
