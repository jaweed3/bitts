#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# BitJETS Demo — generates showcase artifacts for README / LinkedIn / blog
#
# Usage:
#   ./scripts/demo.sh
#   ./scripts/demo.sh --model checkpoints/bitjets_ckpt_180.pth
#
# Output dir: demo_output/
#   ├── 01_hello_world.wav
#   ├── 02_quick_brown_fox.wav
#   ├── 03_rain_in_spain.wav
#   ├── 04_abstract.wav
#   ├── 05_pop_culture.wav
#   ├── 06_long_form.wav
#   ├── mel_grid.png
#   ├── benchmark.txt
#   └── summary.txt
# ---------------------------------------------------------------------------
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
NC="\033[0m"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="demo_output"

# ── Find checkpoint ────────────────────────────────────────────────────────
find_latest_checkpoint() {
    local latest=""
    for ckpt in $(ls checkpoints/bitjets_ckpt_*.pth 2>/dev/null | sort -t_ -k3 -n); do
        [ -f "$ckpt" ] || continue
        SIZE=$(stat -c%s "$ckpt" 2>/dev/null || stat -f%z "$ckpt")
        [ "$SIZE" -lt 1024 ] && continue  # skip git-lfs pointers
        latest="$ckpt"
    done
    [ -z "$latest" ] && [ -f "checkpoints/latest.pth" ] && latest="checkpoints/latest.pth"
    echo "$latest"
}

MODEL_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
    esac
    shift
done

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH=$(find_latest_checkpoint)
fi

if [ -z "$MODEL_PATH" ]; then
    echo "Error: No checkpoint found. Train first or pass --model <path>"
    echo "  Quick test: ./scripts/demo.sh --model checkpoints/UNIVERSAL_V1/g_02500000"
    exit 1
fi

# ── Setup ──────────────────────────────────────────────────────────────────
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo -e "${BOLD}${CYAN}============================================================${NC}"
echo -e "${BOLD}${CYAN}  BitJETS Demo Generator${NC}"
echo -e "${BOLD}${CYAN}============================================================${NC}"
echo ""
echo -e "  Model:  ${BOLD}$MODEL_PATH${NC}"
echo -e "  Output: ${BOLD}$OUTPUT_DIR/${NC}"
echo ""

# ── Sample texts (diverse: benchmark lines, pop culture, technical) ────────
declare -A SAMPLES
SAMPLES=(
    ["01_hello_world"]="hello world, this is BitNet TTS"
    ["02_quick_brown_fox"]="the quick brown fox jumps over the lazy dog"
    ["03_rain_in_spain"]="the rain in Spain stays mainly in the plain"
    ["04_science"]="bitnet is a one point five eight bit transformer"
    ["05_evening_calm"]="a gentle evening breeze carried the scent of distant rain across the empty fields as the last light faded behind the hills"
    ["06_technical"]="quantization aware training replaces all convolutional weights with ternary values while keeping activations at eight bits"
)

echo -e "${BOLD}Generating audio samples...${NC}"
echo ""

for name in "${!SAMPLES[@]}"; do
    text="${SAMPLES[$name]}"
    printf "  [%s] %s\n" "$name" "$text"
    uv run python main.py infer \
        --model-path "$MODEL_PATH" \
        --text "$text" \
        --output "$OUTPUT_DIR/${name}.wav" \
        --speed 1.0 \
        2>&1 | grep -E "DONE|Gagal|Error|✨|🐢" || true
done

# ── Benchmark ──────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Running benchmark...${NC}"
echo ""

uv run python main.py benchmark --model-path "$MODEL_PATH" 2>&1 | tee "$OUTPUT_DIR/benchmark.txt"

# ── Mel spectrogram grid (via Python one-liner using sample_gen) ───────────
echo ""
echo -e "${BOLD}Generating mel spectrogram visualizations...${NC}"

uv run python -c "
import os, sys
sys.path.insert(0, 'src')
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
from hparams import HParams
from models import BitJETS
from inference import main as infer_main

# Generate single mel plot for the longest sample
text = '${SAMPLES[06_technical]}'
print(f'  Plotting mel for: {text[:60]}...')

from dataset import text_to_sequence
model = BitJETS(
    vocab_size=HParams.VOCAB_SIZE,
    embed_dim=HParams.EMBED_DIM,
    hidden_dim=HParams.ENCODER_DIM,
    decoder_dim=HParams.DECODER_DIM,
    out_mel_dim=HParams.N_MELS,
).to(HParams.DEVICE)

ckpt = torch.load('$MODEL_PATH', map_location=HParams.DEVICE, weights_only=False)
sd = ckpt.get('model_state_dict', ckpt)
if '__packed_keys__' in sd:
    from packing import unpack_state_dict
    sd = unpack_state_dict(sd)
model.load_state_dict(sd, strict=False)
model.eval()

seq = text_to_sequence(text)
text_tensor = torch.tensor(seq).unsqueeze(0).to(HParams.DEVICE)
with torch.no_grad():
    mel_pred, _ = model(text_tensor, duration_control=1.0)

mel_plot = mel_pred.squeeze(0).transpose(0, 1).cpu().numpy()

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(mel_plot, origin='lower', aspect='auto', cmap='magma')
ax.set_title(f'BitJETS Mel Spectrogram\n\"{text[:80]}\"', fontsize=12)
ax.set_xlabel('Time Frames', fontsize=10)
ax.set_ylabel('Mel Bands (80)', fontsize=10)
plt.colorbar(im, ax=ax, label='Log-Mel Energy')
plt.tight_layout()
plt.savefig('$OUTPUT_DIR/mel_spectrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  → demo_output/mel_spectrogram.png saved')
" 2>&1

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo -e "${BOLD}${GREEN}  Demo Complete!${NC}"
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo ""

cat > "$OUTPUT_DIR/summary.txt" <<SUMMARY
BitJETS Demo Output
===================
Model: $MODEL_PATH
Date:  $(date)

Generated Files:
$(find "$OUTPUT_DIR" -type f | sort | while read f; do
    size=$(du -h "$f" | cut -f1)
    echo "  $f ($size)"
done)
SUMMARY

cat "$OUTPUT_DIR/summary.txt"
echo ""
echo -e "  ${BOLD}Audio samples:${NC} $OUTPUT_DIR/*.wav"
echo -e "  ${BOLD}Benchmark:${NC}     $OUTPUT_DIR/benchmark.txt"
echo -e "  ${BOLD}Spectrogram:${NC}   $OUTPUT_DIR/mel_spectrogram.png"
echo ""
echo -e "  ${CYAN}For GIF/terminal recording:${NC}"
echo -e "    asciinema rec demo.cast --command './scripts/demo.sh'"
echo -e "    asciinema play demo.cast"
echo -e "    # or convert to GIF: https://github.com/asciinema/asciicast2gif"
echo ""
