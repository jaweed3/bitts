#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# BitJETS Inference Launcher
#
# Usage:
#   ./scripts/infer.sh --text "hello world"
#   ./scripts/infer.sh --text "the quick brown fox" --output my_output.wav
#   ./scripts/infer.sh --text "slow speech" --speed 0.8
#   ./scripts/infer.sh --model checkpoints/bitjets_ckpt_180.pth --text "test"
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ───────────────────────────────────────────────────────────────
TEXT="Hello world, this is BitNet TTS."
MODEL_PATH=""
OUTPUT="output.wav"
SPEED="1.0"
DEVICE="auto"
EXTRA=()

# ── Auto-detect latest checkpoint ──────────────────────────────────────────
find_latest_checkpoint() {
    local latest=""
    # Sort numerically by step number embedded in filename
    for ckpt in $(ls checkpoints/bitjets_ckpt_*.pth 2>/dev/null | sort -t_ -k3 -n); do
        [ -f "$ckpt" ] || continue
        # Skip git-lfs pointer files (<1KB)
        [ "$(stat -c%s "$ckpt" 2>/dev/null || stat -f%z "$ckpt")" -lt 1024 ] && continue
        latest="$ckpt"
    done
    # Fall back to latest.pth
    if [ -z "$latest" ] && [ -f "checkpoints/latest.pth" ]; then
        latest="checkpoints/latest.pth"
    fi
    echo "$latest"
}

# ── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --text)     TEXT="$2"; shift ;;
        --model)     MODEL_PATH="$2"; shift ;;
        --output)   OUTPUT="$2"; shift ;;
        --speed)    SPEED="$2"; shift ;;
        --device)   DEVICE="$2"; shift ;;
        *)          EXTRA+=("$1") ;;
    esac
    shift
done

# ── Find model ─────────────────────────────────────────────────────────────
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH=$(find_latest_checkpoint)
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No checkpoint found. Train first or pass --model <path>"
        exit 1
    fi
    echo "Auto-detected checkpoint: $MODEL_PATH"
fi

# ── Launch ─────────────────────────────────────────────────────────────────
exec uv run python main.py infer \
    --device "$DEVICE" \
    --model-path "$MODEL_PATH" \
    --text "$TEXT" \
    --output "$OUTPUT" \
    --speed "$SPEED" \
    "${EXTRA[@]}"
