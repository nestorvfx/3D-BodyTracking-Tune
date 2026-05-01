#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Vast.ai bootstrap for BlazePose Lite/Full v2 distillation.
# Idempotent — safe to re-run on instance restart.
#
# Provisions:
#  1. System libs (ffmpeg, awscli, rsync, tmux)
#  2. PyTorch CUDA wheels appropriate for the detected GPU
#  3. Project Python deps (requirements.txt)
#  4. /data scratch directories on the instance NVMe
#  5. AWS credentials for the Ego-Exo4D bucket
#  6. HuggingFace token for the synth corpus
#
# Required env vars (export before running):
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  — from your Ego-Exo4D licence email
#   HF_TOKEN                                  — HuggingFace read token
#
# Usage on Vast:
#   git clone https://github.com/nestorvfx/3D-BodyTracking-Tune.git
#   cd 3D-BodyTracking-Tune
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... HF_TOKEN=hf_...
#   bash vast_setup.sh
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

log() { echo -e "\n\033[1;34m[setup]\033[0m $*"; }
die() { echo -e "\033[1;31m[setup:fatal]\033[0m $*" >&2; exit 1; }

# Idempotency marker
MARKER=/workspace/.bptune_setup_done
[ -f "$MARKER" ] && { log "already set up — delete $MARKER to force"; exit 0; }

# 1. System packages -----------------------------------------------------
# Note: `awscli` was dropped from Ubuntu 24.04 apt repos; it's installed via
# pip in step 3 (requirements.txt).  Don't apt-install it here.
log "system packages"
apt-get update -qq
apt-get install -y -qq ffmpeg rsync tmux git unzip curl jq libgl1 libglib2.0-0 >/dev/null

# Ubuntu 24.04 / PEP 668: the system Python is marked externally-managed,
# which blocks `pip install` system-wide.  PEP 668 itself endorses removing
# the marker for single-application container images (Vast is exactly that —
# throwaway container, no other Python apps to break).  This is cleaner than
# threading --break-system-packages through every pip call (and through the
# pip subprocess calls inside our Python scripts) on a per-script basis.
# Reference: https://peps.python.org/pep-0668/ §"Container images".
for f in /usr/lib/python3.*/EXTERNALLY-MANAGED; do
    [ -f "$f" ] && rm -f "$f" && log "removed PEP 668 marker: $f"
done

# 2. Python + PyTorch ----------------------------------------------------
log "python deps"
python3 -m pip install --quiet --upgrade pip || \
    log "[warn] pip self-upgrade refused; continuing with system pip"

# Detect GPU architecture
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
log "detected GPU: $GPU_NAME"

if echo "$GPU_NAME" | grep -qiE "RTX 50|B200|H200"; then
  log "Blackwell-class GPU → PyTorch nightly cu128"
  python3 -m pip install --quiet --pre torch torchvision \
      --index-url https://download.pytorch.org/whl/nightly/cu128
elif echo "$GPU_NAME" | grep -qiE "RTX 4|A100|A40|A6000|H100|A10|L40|L4|T4|V100|RTX 3"; then
  log "Hopper/Ampere/Ada GPU → PyTorch 2.6 + cu124"
  python3 -m pip install --quiet torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124
else
  log "[warn] unrecognised GPU '$GPU_NAME' — falling back to default cu124 wheels"
  python3 -m pip install --quiet torch torchvision \
      --index-url https://download.pytorch.org/whl/cu124
fi

# 3. Project deps --------------------------------------------------------
log "project requirements"
python3 -m pip install --quiet -r requirements.txt

# 4. Verify ---------------------------------------------------------------
log "GPU sanity check"
python3 - <<'PY'
import torch
print(f"  torch:       {torch.__version__}")
print(f"  cuda avail:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu {i}:      {p.name}  sm_{p.major}{p.minor}  {p.total_memory/1e9:.0f} GB")
        x = torch.randn(8, 8, device=f"cuda:{i}")
        (x @ x.T).sum()  # force kernel load — fails fast if wrong cu wheel
        print(f"             matmul OK")
PY

# 5. Scratch dirs --------------------------------------------------------
log "scratch dirs on instance disk"
mkdir -p /data/{synth,egoexo,oiv7,teacher_cache} \
         /workspace/{ckpts,runs,logs,exports}

# 6. Credentials ---------------------------------------------------------
if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  log "configuring AWS credentials"
  mkdir -p ~/.aws
  cat > ~/.aws/credentials <<EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
  cat > ~/.aws/config <<EOF
[default]
region = us-east-2
output = json
EOF
else
  log "[warn] AWS_ACCESS_KEY_ID not set — Ego-Exo4D download will fail"
fi

if [ -n "${HF_TOKEN:-}" ]; then
  log "configuring HuggingFace token"
  python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$HF_TOKEN')"
else
  log "[warn] HF_TOKEN not set — synth-corpus download will fail"
fi

# 7. Pose-landmark .task weights (Apache-2.0 from Google CDN) -----------
log "downloading BlazePose Lite/Full/Heavy .task files"
mkdir -p assets
declare -A POSE_URLS=(
  [lite]="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
  [full]="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
  [heavy]="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
for variant in lite full heavy; do
    target="assets/pose_landmarker_${variant}.task"
    if [ -s "$target" ]; then
        echo "  ✓ $target ($(du -h "$target" | cut -f1))"
    else
        curl -fsSL -o "$target" "${POSE_URLS[$variant]}"
        echo "  ✓ downloaded $target"
    fi
done

# 8. Hand + Face teachers (auto-download via teachers.py on first use) ---
log "warming teacher download cache (Hand + Face .task files)"
python3 - <<'PY' || true
import sys
sys.path.insert(0, "training")
from teachers import download_teacher_weights
from pathlib import Path
download_teacher_weights(Path("assets/teachers"))
PY

# 9. Open Images V7 sim2real corpus -------------------------------------
# Two occluder corpora extracted (Sárándi-2024 SOTA: object + human-shaped):
#   occluders/        — non-human classes (8000 cutouts, original Sárándi-2018)
#   occluders_human/  — human silhouettes (4000 cutouts, Sárándi-2024 update)
if [ ! -d "assets/sim2real_refs/occluders" ] || \
   [ "$(ls -1 assets/sim2real_refs/occluders 2>/dev/null | wc -l)" -lt 1000 ]; then
    log "extracting Open Images V7 sim2real corpus (~30 min, ~270 MB output)"
    bash tooling/_iter_setup_openimages.sh
    python3 tooling/_iter_extract_openimages.py \
        --max-occluders 8000 --max-bg 5000 --n-workers 8 \
        --skip-existing-occluders
fi
if [ ! -d "assets/sim2real_refs/occluders_human" ] || \
   [ "$(ls -1 assets/sim2real_refs/occluders_human 2>/dev/null | wc -l)" -lt 500 ]; then
    log "extracting human-shaped occluders (Sárándi-2024)"
    python3 tooling/_iter_extract_openimages.py \
        --max-occluders 4000 --max-bg 0 --n-workers 8 \
        --human-occluders-only --skip-existing-occluders
fi

touch "$MARKER"
log "DONE — env ready.  Next: bash vast_train.sh"
