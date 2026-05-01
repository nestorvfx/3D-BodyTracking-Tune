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

# Knobs (env overrides) --------------------------------------------------
# BATCH        per-GPU batch size for stages 4 + 5 (default 32 for the
#              16 GB 5070 Ti box; bump to 96 on 24 GB+ cards)
# LR           AdamW peak LR (default 5e-5)
# NUM_WORKERS  dataloader workers per rank (default 4)
# EPOCHS       training epochs (default 20; FAST=1 forces 1)
# NPROC_TRAIN  number of GPUs for stages 4+5 DDP (auto-detect from nvidia-smi
#              if unset; set to 1 to force single-GPU)
# LIMIT_SYNTH / LIMIT_EGOEXO  cap dataset size (empty = unlimited).  Useful
#              for the *first* production run to verify convergence before
#              committing to the full 133k-synth + ~600k-egoexo run.
BATCH="${BATCH:-32}"
LR="${LR:-5e-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-20}"
NPROC_TRAIN="${NPROC_TRAIN:-$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo 1)}"
# BF16=1 (default) enables --bf16; set BF16=0 to fall back to fp32 for
# debugging numerical issues (NaN / Inf).  fp32 is ~2× slower but stable.
BF16="${BF16:-1}"
BF16_FLAG=""
[ "$BF16" != "0" ] && BF16_FLAG="--bf16"

TRAIN_LIMIT_FLAGS=()
[ -n "${LIMIT_SYNTH:-}" ]  && TRAIN_LIMIT_FLAGS+=(--limit-synth  "$LIMIT_SYNTH")
[ -n "${LIMIT_EGOEXO:-}" ] && TRAIN_LIMIT_FLAGS+=(--limit-egoexo "$LIMIT_EGOEXO")

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
        log "stage 2b: batched-bulk download + extract Ego-Exo4D train frames"
        EXTRA=()
        [ -n "${FAST:-}" ] && EXTRA+=(--limit-takes 50)
        # Default batch=200 takes/call (~30 GB peak transient).  For FAST=1
        # smoke (50 takes) we can fit all in one batch — set higher for full.
        EE_BATCH="${EGOEXO_BATCH:-200}"
        EE_WORKERS="${EGOEXO_WORKERS:-32}"
        python3 prep/extract_egoexo_train.py \
            --anno-root /data/egoexo/annotations \
            --raw-root /data/egoexo \
            --frames-root /data/egoexo/frames \
            --manifest-out /data/egoexo/manifest_train.jsonl \
            --batch-size "$EE_BATCH" --num-workers "$EE_WORKERS" \
            "${EXTRA[@]}"
    fi
fi

# 3. Teacher cache -----------------------------------------------------
# Heavy teacher covers all 33 BP joints (incl. face idx 0-10) — Face Mesh
# is REDUNDANT and tripled inference cost.  Default: Heavy + Hand only.
# Add SKIP_HAND_FACE=1 for Heavy-only runs (fastest possible KD signal).
# Set INCLUDE_FACE=1 to opt back into Face Mesh KD (slow, marginal value).
if want_stage 3; then
    HAND_FACE_FLAGS=""
    [ -z "${SKIP_HAND_FACE:-}" ] && HAND_FACE_FLAGS="--include-hand"
    [ -n "${INCLUDE_FACE:-}"   ] && HAND_FACE_FLAGS="$HAND_FACE_FLAGS --include-face"

    # Cache only the frames training will actually consume.  FAST=1 caps both
    # passes to 2000 frames (smoke parity).  Otherwise track LIMIT_SYNTH /
    # LIMIT_EGOEXO so a "first production" run doesn't burn ~7 hr caching
    # frames it'll never see.  Empty = full corpus.
    SYNTH_CACHE_LIMIT=""
    EGOEXO_CACHE_LIMIT=""
    [ -n "${LIMIT_SYNTH:-}" ]  && SYNTH_CACHE_LIMIT="--limit $LIMIT_SYNTH"
    [ -n "${LIMIT_EGOEXO:-}" ] && EGOEXO_CACHE_LIMIT="--limit $LIMIT_EGOEXO"
    if [ -n "${FAST:-}" ]; then
        SYNTH_CACHE_LIMIT="--limit 2000"
        EGOEXO_CACHE_LIMIT="--limit 2000"
    fi

    # GPU delegate is roughly even with CPU on 0.10.33 (regression vs 0.10.20),
    # but no worse — opt-in via TEACHER_GPU=1 if you want to test.
    GPU_FLAG=""
    [ -n "${TEACHER_GPU:-}" ] && GPU_FLAG="--gpu"

    # Multi-process parallelism across N GPUs.  Default 4 (the box has 4).
    # Each worker gets 1/N of the frames + pinned to its own GPU via
    # CUDA_VISIBLE_DEVICES.  Set TEACHER_NPROC=1 to run sequentially.
    NPROC="${TEACHER_NPROC:-4}"

    log "stage 3a: cache teachers on synth ($NPROC parallel workers)"
    pids=()
    for ((i=0; i<NPROC; i++)); do
        CUDA_VISIBLE_DEVICES=$i python3 training/cache_teachers.py \
            --source-dir /data/synth --labels-jsonl /data/synth/labels.jsonl \
            --out-dir /data/teacher_cache \
            --worker-id "$i" --num-workers "$NPROC" \
            $HAND_FACE_FLAGS $SYNTH_CACHE_LIMIT $GPU_FLAG &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || log "[warn] cache worker $p failed"; done

    if [ -f /data/egoexo/manifest_train.jsonl ]; then
        log "stage 3b: cache teachers on Ego-Exo4D ($NPROC parallel workers)"
        pids=()
        for ((i=0; i<NPROC; i++)); do
            CUDA_VISIBLE_DEVICES=$i python3 training/cache_teachers.py \
                --source-dir /data/egoexo/frames \
                --out-dir /data/teacher_cache \
                --worker-id "$i" --num-workers "$NPROC" \
                $HAND_FACE_FLAGS $EGOEXO_CACHE_LIMIT $GPU_FLAG &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p" || log "[warn] cache worker $p failed"; done
    fi
fi

# 4. Lite training -----------------------------------------------------
# Multi-GPU via torchrun: NPROC_TRAIN ranks, each pinned to its own GPU.
# The training loop wraps the student in DistributedDataParallel and shards
# data across ranks via DistributedSampler — effective batch = NPROC × BATCH.
TRAIN_CMD=(torchrun --standalone --nproc-per-node="$NPROC_TRAIN")
[ "$NPROC_TRAIN" = "1" ] && TRAIN_CMD=(python3)

if want_stage 4; then
    log "stage 4: train Lite v2  (DDP nproc=$NPROC_TRAIN, batch/rank=$BATCH, "
    log "         effective batch=$((NPROC_TRAIN * BATCH)))"
    EXTRA=()
    [ -n "${FAST:-}" ] && EXTRA+=(--limit-synth 5000 --limit-egoexo 10000 --epochs 1)
    "${TRAIN_CMD[@]}" training/train.py \
        --variant lite \
        --synth-root /data/synth \
        --egoexo-root /data/egoexo \
        --anno-root /data/egoexo/annotations \
        --teacher-cache /data/teacher_cache \
        --out-root /workspace/ckpts \
        --runs-root /workspace/runs \
        --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" $BF16_FLAG --num-workers "$NUM_WORKERS" \
        --ckpt-every-min 10 \
        "${TRAIN_LIMIT_FLAGS[@]}" \
        "${EXTRA[@]}"
fi

# 5. Full training -----------------------------------------------------
if want_stage 5; then
    log "stage 5: train Full v2  (DDP nproc=$NPROC_TRAIN, batch/rank=$BATCH, "
    log "         effective batch=$((NPROC_TRAIN * BATCH)))"
    EXTRA=()
    [ -n "${FAST:-}" ] && EXTRA+=(--limit-synth 5000 --limit-egoexo 10000 --epochs 1)
    "${TRAIN_CMD[@]}" training/train.py \
        --variant full \
        --synth-root /data/synth \
        --egoexo-root /data/egoexo \
        --anno-root /data/egoexo/annotations \
        --teacher-cache /data/teacher_cache \
        --out-root /workspace/ckpts \
        --runs-root /workspace/runs \
        --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" $BF16_FLAG --num-workers "$NUM_WORKERS" \
        --ckpt-every-min 10 \
        "${TRAIN_LIMIT_FLAGS[@]}" \
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
