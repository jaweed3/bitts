#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# BitJETS Training Launcher
#
# Usage:
#   ./scripts/train.sh                          # fresh training
#   ./scripts/train.sh --resume                 # auto-resume from latest
#   ./scripts/train.sh --resume --no-wandb      # offline mode
#   ./scripts/train.sh --batch-size 64 --steps 1000000
#
# Presets (RTX 4060 8GB VRAM):
#   default  : bs=32, accum=1  (safe)
#   high     : bs=64, accum=1  (faster if it fits)
#   cpu      : bs=2, accum=1   (debug only)
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ───────────────────────────────────────────────────────────────
MODE="train"
RESUME=""
NO_WANDB=""
BATCH_SIZE=""
ACCUM_STEPS=""
NUM_STEPS=""
DEVICE="auto"
SEED=""
EXTRA=()

# ── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)       RESUME="--auto-resume" ;;
        --no-wandb)     NO_WANDB="--no-wandb" ;;
        --batch-size)   BATCH_SIZE="--batch-size $2"; shift ;;
        --accum-steps)  ACCUM_STEPS="--accum-steps $2"; shift ;;
        --steps)        NUM_STEPS="--num-steps $2"; shift ;;
        --device)       DEVICE="$2"; shift ;;
        --seed)         SEED="--seed $2"; shift ;;
        --preset)
            case "$2" in
                high)
                    BATCH_SIZE="--batch-size 64"
                    ACCUM_STEPS="--accum-steps 1"
                    ;;
                safe)
                    BATCH_SIZE="--batch-size 32"
                    ACCUM_STEPS="--accum-steps 1"
                    ;;
                cpu)
                    BATCH_SIZE="--batch-size 2"
                    ACCUM_STEPS="--accum-steps 1"
                    DEVICE="cpu"
                    ;;
                *) echo "Unknown preset: $2 (use: high, safe, cpu)"; exit 1 ;;
            esac
            shift ;;
        *) EXTRA+=("$1") ;;
    esac
    shift
done

# ── Show config ────────────────────────────────────────────────────────────
echo "=============================================="
echo "  BitJETS Training"
echo "=============================================="
echo "  Device:      $DEVICE"
echo "  Resume:      ${RESUME:-no (fresh)}"
echo "  WandB:       $([ -n "$NO_WANDB" ] && echo 'off' || echo 'on')"
echo "  Batch size:  ${BATCH_SIZE:-default}"
echo "  Accum steps: ${ACCUM_STEPS:-default}"
echo "  Steps:       ${NUM_STEPS:-default (500K)}"
echo "  Seed:        ${SEED:-default (42)}"
echo "=============================================="
echo ""

# ── Launch ─────────────────────────────────────────────────────────────────
exec uv run python main.py train \
    --device "$DEVICE" \
    $RESUME \
    $NO_WANDB \
    $BATCH_SIZE \
    $ACCUM_STEPS \
    $NUM_STEPS \
    $SEED \
    "${EXTRA[@]}"
