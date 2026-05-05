#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# BitJETS Bootstrap — one-command setup for training devices (WSL/Linux)
#
# Usage:
#   chmod +x scripts/bootstrap.sh
#   ./scripts/bootstrap.sh
#
# What it does:
#   1. Installs uv (if missing)
#   2. Installs Python 3.10+ via uv (if needed)
#   3. Pulls git-lfs files (HiFi-GAN vocoder)
#   4. uv sync — creates venv + installs all deps (PyTorch CUDA 12.1 on Linux)
#   5. Downloads LJSpeech dataset (~2.6 GB)
#   6. Creates .env config
#   7. Verifies setup with a quick smoke test
# ---------------------------------------------------------------------------
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m" # No Color

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

step()  { echo -e "\n${BOLD}${CYAN}[$1]${NC} ${BOLD}$2${NC}"; }
ok()    { echo -e "  ${GREEN}✓${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✗${NC} $1"; exit 1; }

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  BitJETS Bootstrap — Training Device Setup${NC}"
echo -e "${BOLD}============================================================${NC}"

# ------------------------------------------------------------------
# 1. uv
# ------------------------------------------------------------------
step 1 "Checking uv package manager..."

if command -v uv &>/dev/null; then
    ok "uv $(uv --version) found at $(which uv)"
else
    warn "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source shell config so uv is on PATH for the rest of this script
    export PATH="$HOME/.cargo/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv installed: $(uv --version)"
    else
        fail "uv installation failed. Try: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
fi

# ------------------------------------------------------------------
# 2. Python 3.10+
# ------------------------------------------------------------------
step 2 "Checking Python 3.10+..."

PYTHON_OK=false
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
        ok "Python $PY_VER found at $(which python3)"
        PYTHON_OK=true
    else
        warn "Python $PY_VER is too old (need >=3.10)"
    fi
else
    warn "python3 not found on PATH"
fi

if [ "$PYTHON_OK" = false ]; then
    warn "Installing Python 3.12 via uv..."
    uv python install 3.12
    ok "Python 3.12 installed via uv"
fi

# ------------------------------------------------------------------
# 3. Git LFS (HiFi-GAN vocoder checkpoints)
# ------------------------------------------------------------------
step 3 "Pulling git-lfs files (HiFi-GAN vocoder)..."

if command -v git-lfs &>/dev/null; then
    git lfs pull
    if [ -f "checkpoints/UNIVERSAL_V1/g_02500000" ]; then
        VOC_SIZE=$(du -h checkpoints/UNIVERSAL_V1/g_02500000 | cut -f1)
        ok "HiFi-GAN vocoder ready (generator: ${VOC_SIZE})"
    else
        warn "git-lfs pull succeeded but vocoder files still missing — check .gitattributes"
    fi
else
    warn "git-lfs not installed. Install it first:"
    warn "  Ubuntu/Debian: sudo apt install git-lfs && git lfs install"
    warn "  Then re-run: git lfs pull"
    warn "Skipping vocoder — inference will use Griffin-Lim fallback."
fi

# ------------------------------------------------------------------
# 4. System dependencies (Linux/WSL only)
# ------------------------------------------------------------------
step 4 "Checking system libraries..."

if [[ "$(uname -s)" == "Linux" ]]; then
    MISSING_DEPS=()

    # libsndfile — needed by soundfile for audio I/O
    if ! ldconfig -p 2>/dev/null | grep -q libsndfile; then
        MISSING_DEPS+=("libsndfile1")
    fi

    # sox — optional, used by some audio pipelines
    if ! command -v sox &>/dev/null; then
        MISSING_DEPS+=("sox")
    fi

    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        warn "Missing system packages: ${MISSING_DEPS[*]}"
        warn "Install with: sudo apt install ${MISSING_DEPS[*]}"
        if [ "$(id -u)" -eq 0 ]; then
            apt-get update -qq && apt-get install -y "${MISSING_DEPS[@]}"
            ok "Installed: ${MISSING_DEPS[*]}"
        else
            warn "Re-run with sudo or install manually if audio loading fails."
        fi
    else
        ok "System libraries OK"
    fi
else
    ok "macOS detected — skipping Linux system deps"
fi

# ------------------------------------------------------------------
# 5. uv sync — venv + dependencies
# ------------------------------------------------------------------
step 5 "Installing dependencies (uv sync)..."

uv sync --no-dev
ok "Dependencies installed (PyTorch CUDA 12.1 on Linux, MPS on macOS)"

# ------------------------------------------------------------------
# 6. LJSpeech Dataset
# ------------------------------------------------------------------
step 6 "Checking LJSpeech dataset..."

DATA_DIR="$PROJECT_ROOT/data/speech"
LJSPEECH_URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJSPEECH_ARCHIVE="/tmp/LJSpeech-1.1.tar.bz2"

if [ -f "$DATA_DIR/metadata.csv" ] && [ -d "$DATA_DIR/wavs" ]; then
    WAV_COUNT=$(ls "$DATA_DIR/wavs"/*.wav 2>/dev/null | wc -l)
    ok "LJSpeech already present ($WAV_COUNT utterances)"
else
    warn "LJSpeech not found at $DATA_DIR"
    echo -e "  ${YELLOW}Downloading LJSpeech-1.1 (~2.6 GB)...${NC}"

    mkdir -p "$DATA_DIR"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$LJSPEECH_ARCHIVE" "$LJSPEECH_URL"
    elif command -v curl &>/dev/null; then
        curl -L -o "$LJSPEECH_ARCHIVE" "$LJSPEECH_URL"
    else
        fail "Neither wget nor curl found. Install one and re-run."
    fi

    warn "Extracting to $DATA_DIR ..."
    tar -xjf "$LJSPEECH_ARCHIVE" -C /tmp/
    mv /tmp/LJSpeech-1.1/wavs  "$DATA_DIR/"
    mv /tmp/LJSpeech-1.1/metadata.csv "$DATA_DIR/"
    rm -rf /tmp/LJSpeech-1.1 "$LJSPEECH_ARCHIVE"

    WAV_COUNT=$(ls "$DATA_DIR/wavs"/*.wav 2>/dev/null | wc -l)
    ok "LJSpeech ready ($WAV_COUNT utterances)"
fi

# ------------------------------------------------------------------
# 7. .env
# ------------------------------------------------------------------
step 7 "Creating .env config..."

if [ -f ".env" ]; then
    ok ".env already exists — skipping"
else
    cat > .env <<'DOTENV'
# BitJETS environment config
# WandB API key (optional — leave empty to disable)
WANDB_API_KEY=

# Override data path (default: ./data/speech)
# BITTS_DATA_PATH=./data/speech

# Override checkpoint dir (default: ./checkpoints)
# BITTS_CHECKPOINT_DIR=./checkpoints
DOTENV
    ok ".env created (edit to add WANDB_API_KEY if using WandB)"
fi

# ------------------------------------------------------------------
# 8. Verify
# ------------------------------------------------------------------
step 8 "Verifying setup..."

# Quick import check
uv run python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
" || warn "PyTorch import check failed — check uv sync output"

# Run unit tests (no dataset needed)
echo ""
uv run python -m pytest tests/test_layers.py tests/test_model.py tests/test_packing.py -v --tb=short 2>&1 | tail -20

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo -e "${BOLD}${GREEN}  Setup Complete!${NC}"
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo ""
echo -e "  Next steps:"
echo -e ""
echo -e "  ${BOLD}Start training:${NC}"
echo -e "    ./scripts/train.sh"
echo -e ""
echo -e "  ${BOLD}Resume from latest checkpoint:${NC}"
echo -e "    ./scripts/train.sh --resume"
echo -e ""
echo -e "  ${BOLD}Train offline (no WandB):${NC}"
echo -e "    ./scripts/train.sh --resume --no-wandb"
echo -e ""
echo -e "  ${BOLD}Custom batch size / steps:${NC}"
echo -e "    ./scripts/train.sh --batch-size 64 --steps 1000000"
echo -e ""
echo -e "  ${BOLD}Inference:${NC}"
echo -e "    ./scripts/infer.sh --text \"hello world\""
echo -e ""
echo -e "  ${BOLD}Ngopi dulu ☕, biar GPU yang kerja.${NC}"
echo ""

# ------------------------------------------------------------------
# 9. Cleanup
# ------------------------------------------------------------------
step 9 "Cleaning up..."

# Remove Python bytecode cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Clean uv cache (optional — uncomment if disk space is tight)
# uv cache clean

# Remove any leftover temp archives
rm -f /tmp/LJSpeech-1.1.tar.bz2 2>/dev/null || true

ok "Cleanup done"
echo ""
echo -e "  ${GREEN}Disk usage:${NC}"
df -h "$PROJECT_ROOT" 2>/dev/null | tail -1 || true
echo ""
