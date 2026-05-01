#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# End-to-end training orchestrator for Vast.ai.
#
# Stages:
#   0. Recreate benchmark/frames/ (26-take SOTA val subset)      (~10-20 min, ~1.1 GB)
#   1. Download synth corpus from HuggingFace → /data/synth     (~30 min, ~25 GB)
#   2. Download Ego-Exo4D train split via egoexo CLI            (~3-6 hr, ~400 GB transient)
#      + extract annotated frames per take, deleting videos after.
#   3. Pre-cache Heavy + (optional) Hand + Face teachers        (~3-6 hr GPU)
#   4. Train Lite v2 (DDP across NPROC_TRAIN GPUs)              (~7 hr A100 / ~3-4 hr 4×5070Ti)
#   5. Train Full v2 (DDP across NPROC_TRAIN GPUs)              (~7 hr A100 / ~3-4 hr 4×5070Ti)
#   6. Export both as .task                                      (~5 min)
#   7. Score MediaPipe(.task) on benchmark vs v1 Lite/Full/Heavy (~25 min)
#      → benchmark/results/analysis_v2.json
#      v1 in-spec PA-MPJPE: Lite 90.7mm  Full 86.0mm  Heavy 93.2mm
#      target: student_v2_full ≤ 86 mm at the same inference cost
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

# Loss-weight + LR-rule knobs (env overrides).
# Smoke 2 (after first SOTA-aim revision) was unchanged from smoke 1; root
# cause: Heavy KD pulls toward Heavy's distribution, which is *worse* than
# v1 Full on Ego-Exo4D exo views (-5.6 mm; benchmark/results/RESULTS.md).
# New defaults: heavy KD off, image-frame anchor 5.0× to dominate, hard 0.3×
# because synth's MPFB-rig joints have systematic 1-4 cm offset everywhere.
# LAM_HARD=0 default: smoke 5 confirmed L_hard pulls student away from v1
# even with synth contribution disabled, because egoexo's body-axis hard
# labels are computed from egoexo-annotated hip/shoulders (visual-landmark
# annotation with 14-px reprojection floor), while L_anchor pulls toward
# v1's body-axis predictions computed from v1's OWN hip/shoulder estimates.
# These are different frames — student can't satisfy both simultaneously.
# Drop L_hard entirely; rely on L_anchor + L_anchor_img + L_multiview.
# Multi-view loss provides 2D image-frame supervision via reprojection
# (avoids the body-frame mismatch).
LAM_HARD="${LAM_HARD:-0.0}"
LAM_KD_B="${LAM_KD_B:-0.5}"
LAM_ANCHOR="${LAM_ANCHOR:-1.0}"
LAM_ANCHOR_IMG="${LAM_ANCHOR_IMG:-5.0}"
LAM_MV="${LAM_MV:-0.5}"   # boosted from 0.2 — now load-bearing image-frame GT signal
USE_HEAVY_KD="${USE_HEAVY_KD:-0}"          # 0 = off (default); 1 = enable Heavy KD
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"    # 0 = full fine-tune; 1 = heads only
# Synth labels are MPFB2 rig joints with 1-4 cm offset from BlazePose visual
# landmarks across the WHOLE skeleton.  Smoke 4 confirmed this offset pulls
# the student toward MPFB convention while L_anchor pulls toward visual,
# regressing benchmark by 110+ mm.  Default ON: synth becomes a pure
# image-diversity source via anchor distillation; ego-exo provides the only
# L_hard signal.
DISABLE_SYNTH_HARD="${DISABLE_SYNTH_HARD:-1}"
LR_RULE="${LR_RULE:-sqrt}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
VARIANT_LR_LITE="${VARIANT_LR_LITE:-1.0}"
VARIANT_LR_FULL="${VARIANT_LR_FULL:-0.5}"

LOSS_FLAGS=(--lam-hard "$LAM_HARD" --lam-kd-b "$LAM_KD_B"
            --lam-anchor "$LAM_ANCHOR" --lam-anchor-img "$LAM_ANCHOR_IMG"
            --lam-mv "$LAM_MV"
            --lr-scale-rule "$LR_RULE" --warmup-steps "$WARMUP_STEPS")
[ "$USE_HEAVY_KD"       = "1" ] && LOSS_FLAGS+=(--use-heavy-kd)
[ "$FREEZE_BACKBONE"    = "1" ] && LOSS_FLAGS+=(--freeze-backbone)
[ "$DISABLE_SYNTH_HARD" = "1" ] && LOSS_FLAGS+=(--disable-synth-hard)
# NO_AUTO_RESUME=1: useful when iterating on loss config — old optimizer
# state can carry stale gradient statistics and re-cause the previous
# regression even with new lambdas.  Default 1 (don't auto-resume) for
# this iteration phase; flip to 0 once we're in production.
NO_AUTO_RESUME="${NO_AUTO_RESUME:-1}"
[ "$NO_AUTO_RESUME" = "1" ] && LOSS_FLAGS+=(--no-auto-resume)

want_stage() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# 0. Benchmark -----------------------------------------------------------
# Recreate `benchmark/frames/` + `benchmark/raw/annotations/` on Vast by
# re-extracting the 26-take SOTA val subset from Ego-Exo4D.  The take_uids
# (subset.json) and per-frame manifest (frames_manifest.json) are already
# in the repo, so the result is byte-stable with the local benchmark.
# Idempotent — skips if frames are already on disk.
# Powers `train.py:benchmark_eval()` (per-epoch BENCH_PA_MPJPE) and stage 7
# (post-export apples-to-apples eval against v1 Lite/Full/Heavy).
if want_stage 0; then
    if [ ! -d "$ROOT/benchmark/frames" ] || \
       [ -z "$(ls -A "$ROOT/benchmark/frames" 2>/dev/null)" ]; then
        log "stage 0: extract benchmark val frames (26 takes, ~1.1 GB, ~10-20 min)"
        python3 prep/extract_benchmark_val.py \
            --raw-root    "$ROOT/benchmark/raw" \
            --frames-root "$ROOT/benchmark/frames"
    else
        log "stage 0: benchmark frames already on disk; skipping"
    fi
fi

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
        --variant-lr-scale "$VARIANT_LR_LITE" \
        "${LOSS_FLAGS[@]}" \
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
        --variant-lr-scale "$VARIANT_LR_FULL" \
        "${LOSS_FLAGS[@]}" \
        "${TRAIN_LIMIT_FLAGS[@]}" \
        "${EXTRA[@]}"
fi

# 6. Export ------------------------------------------------------------
# Pure byte-substitution into v1's .tflite — preserves flatbuffer
# structure, op graph, BuiltinOptions, NHWC input layout, and the
# TFLITE_METADATA buffer (NormalizationOptions for MediaPipe).
# Missing checkpoints are skipped (so a Lite-only iteration that ran
# only stage 4 doesn't fail here).
if want_stage 6; then
    log "stage 6: export .task files (byte-substitution into v1 flatbuffer)"
    for v in lite full; do
        ckpt="/workspace/ckpts/${v}_final.pt"
        if [ -f "$ckpt" ]; then
            python3 model/export.py --variant "$v" \
                --ckpt "$ckpt" \
                --out  "/workspace/exports/${v}_v2.task"
        else
            log "  $ckpt missing — skipping $v export"
        fi
    done
fi

# 7. Benchmark ---------------------------------------------------------
# Apples-to-apples PA-MPJPE vs v1 Lite/Full/Heavy on the same 26-take /
# 1,292-frame subset that produced benchmark/results/RESULTS.md.  Uses
# MediaPipe + the exported .task files (NOT the PyTorch port that runs
# during training), so the numbers are directly comparable to v1.
#
# v1 in-spec ≤ 4 m: Lite 90.7mm  Full 86.0mm  Heavy 93.2mm
# Target: student_v2_full ≤ 86 mm beats v1 Full at the same inference cost.
if want_stage 7; then
    if [ ! -d "$ROOT/benchmark/frames" ] || \
       [ -z "$(ls -A "$ROOT/benchmark/frames" 2>/dev/null)" ]; then
        log "stage 7: benchmark frames missing — running stage 0 first"
        python3 prep/extract_benchmark_val.py \
            --raw-root    "$ROOT/benchmark/raw" \
            --frames-root "$ROOT/benchmark/frames"
    fi
    log "stage 7: run MediaPipe inference on available variants"
    # v1 predictions are .gitignored, so regenerate on Vast.  run_eval.py
    # caches per-take JSON and skips if already present, so reruns are fast.
    # student_v2_* skipped if their .task isn't on disk (Lite-only iteration).
    export V2_EXPORTS_DIR="/workspace/exports"
    cd "$ROOT/benchmark"
    VARIANTS_TO_RUN=(lite full heavy)
    [ -f "/workspace/exports/lite_v2.task" ] && VARIANTS_TO_RUN+=(student_v2_lite)
    [ -f "/workspace/exports/full_v2.task" ] && VARIANTS_TO_RUN+=(student_v2_full)
    for v in "${VARIANTS_TO_RUN[@]}"; do
        python3 run_eval.py --variant "$v"
    done
    log "stage 7: PA-MPJPE analysis vs v1 baselines"
    python3 analyze.py --manual-only \
        --variants "${VARIANTS_TO_RUN[@]}" \
        --out results/analysis_v2.json
    log "stage 7: results saved to benchmark/results/analysis_v2.json"
    log "         v1 baseline (from results/RESULTS.md):"
    log "           in-spec ≤ 4 m: Lite 90.7mm  Full 86.0mm  Heavy 93.2mm"
    log "           full set:      Lite 99.4mm  Full 96.0mm  Heavy 101.6mm"
    log "         target: student_v2_full ≤ 86 mm (matches v1 Full)"
    cd "$ROOT"
fi

log "DONE"
