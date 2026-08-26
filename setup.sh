#!/bin/bash
# ========================================================
#  Galatea-Core v3.2.0 - Linux Setup & Launch Script
# ========================================================
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh          # Setup venv + install deps + launch WebUI
#    ./setup.sh --train  # Setup + launch CLI training
#    ./setup.sh --check  # Only check environment, don't launch
# ========================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}     🌟 Galatea-Core v3.2.0 - Linux Environment Setup       ${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# -------------------------------------------------------
# 1. Check Python
# -------------------------------------------------------
echo -e "${YELLOW}[1/5] Checking Python...${NC}"

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
    if command -v "$cmd" &> /dev/null; then
        PY_VER=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        if [ "$(echo "$PY_VER >= 3.8" | bc 2>/dev/null || echo 0)" = 1 ] || [ "${PY_VER%.*}" -ge 3 ] && [ "${PY_VER#*.}" -ge 8 ] 2>/dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    # Fallback: try to handle version strings like "3.11" without bc
    for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
        if command -v "$cmd" &> /dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}[ERROR] Python 3.8+ not found!${NC}"
    echo "Install it first:"
    echo "  Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
    echo "  Fedora/CentOS: sudo dnf install python3.10"
    echo "  Arch:          sudo pacman -S python"
    exit 1
fi

echo -e "${GREEN}  ✓ Found: $($PYTHON_CMD --version)${NC}"

# -------------------------------------------------------
# 2. Create / Activate Virtual Environment
# -------------------------------------------------------
echo -e "${YELLOW}[2/5] Setting up virtual environment...${NC}"

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}  ✓ Created venv at $VENV_DIR${NC}"
else
    echo -e "${GREEN}  ✓ venv already exists, skipping creation${NC}"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}  ✓ venv activated${NC}"

# -------------------------------------------------------
# 3. Upgrade pip
# -------------------------------------------------------
echo -e "${YELLOW}[3/5] Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}  ✓ pip upgraded${NC}"

# -------------------------------------------------------
# 4. Detect CUDA and Install PyTorch
# -------------------------------------------------------
echo -e "${YELLOW}[4/5] Detecting CUDA & installing dependencies...${NC}"

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' 2>/dev/null || echo "")
    if [ -n "$CUDA_VER" ]; then
        echo -e "${GREEN}  ✓ NVIDIA GPU detected (CUDA $CUDA_VER)${NC}"
        echo -e "  Installing PyTorch with CUDA support..."
        # PyTorch official index uses cu121 for CUDA 11.8/12.1 compatible
        pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    else
        echo -e "${YELLOW}  ⚠ NVIDIA GPU found but couldn't detect CUDA version${NC}"
        echo -e "  Installing PyTorch with CUDA support (cu121)..."
        pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    fi
else
    echo -e "${YELLOW}  ⚠ No NVIDIA GPU detected, installing CPU-only PyTorch${NC}"
    echo -e "  (Training will be slow - a GPU is strongly recommended)"
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# Install remaining dependencies from requirements.txt
pip install -r "$PROJECT_DIR/requirements.txt" --quiet

# 使用与 Windows 一键包相同的检查器验证依赖能够实际导入
python "$PROJECT_DIR/environment_setup.py" --verify-imports

echo -e "${GREEN}  ✓ All dependencies installed${NC}"

# -------------------------------------------------------
# 5. Check OCGCore library & GLIBC compatibility
# -------------------------------------------------------
echo -e "${YELLOW}[5/5] Checking OCGCore library...${NC}"

if [ ! -f "$PROJECT_DIR/ocgcore.so" ]; then
    echo -e "${RED}[ERROR] ocgcore.so not found in project root!${NC}"
    echo "Make sure you cloned the full repository including the prebuilt .so file."
    exit 1
fi

# Quick Python ctypes load test to catch GLIBC version mismatch
GLIBC_CHECK=$("$PYTHON_CMD" -c "
import ctypes, sys, re
try:
    lib = ctypes.cdll.LoadLibrary('$PROJECT_DIR/ocgcore.so')
    print('OK')
except OSError as e:
    msg = str(e)
    # Extract required GLIBC version from error message
    m = re.search(r'GLIBC_(\d+\.\d+)', msg)
    if m:
        print('GLIBC:' + m.group(1))
    else:
        print('FAIL:' + msg)
" 2>&1)

if [ "$GLIBC_CHECK" = "OK" ]; then
    echo -e "${GREEN}  ✓ ocgcore.so found and GLIBC-compatible${NC}"
elif echo "$GLIBC_CHECK" | grep -q "^GLIBC:"; then
    REQUIRED_VER=$(echo "$GLIBC_CHECK" | cut -d: -f2)
    CURRENT_VER=$(/lib/x86_64-linux-gnu/libc.so.6 --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' | head -1 || ldd --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' | head -1 || echo "unknown")
    echo -e "${RED}[ERROR] GLIBC version mismatch!${NC}"
    echo -e "  ocgcore.so requires: ${YELLOW}GLIBC >= $REQUIRED_VER${NC}"
    echo -e "  Your system has:      ${YELLOW}GLIBC $CURRENT_VER${NC}"
    echo ""
    echo "To fix this, upgrade your system's GLIBC or use a newer Linux distro:"
    echo "  Ubuntu 22.04: sudo apt update && sudo apt upgrade libc6"
    echo "  Ubuntu 20.04: Consider upgrading to 22.04+ or use Docker"
    echo "  CentOS 7:     Consider migrating to Rocky Linux 9 / Ubuntu 22.04"
    echo "  Docker (any): docker run -it --gpus all ubuntu:22.04 bash"
    exit 1
else
    echo -e "${RED}[ERROR] ocgcore.so failed to load:${NC}"
    echo -e "  $GLIBC_CHECK"
    exit 1
fi

# Check for card database
if [ ! -f "$PROJECT_DIR/cards.cdb" ] || [ ! -d "$PROJECT_DIR/script" ] || [ -z "$(ls -A "$PROJECT_DIR/script" 2>/dev/null)" ]; then
    echo -e "${YELLOW}  ⚠ Card database or scripts not found, downloading...${NC}"
    python main.py update --data
fi

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}  ✅ Setup complete! Environment is ready.${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# -------------------------------------------------------
# 6. Launch
# -------------------------------------------------------
case "${1:-}" in
    --check)
        echo -e "${GREEN}Environment check passed. Run ./setup.sh to launch WebUI.${NC}"
        ;;
    --train)
        echo -e "${CYAN}Launching CLI training mode...${NC}"
        echo -e "  (Pass additional args after --train, e.g.: ./setup.sh --train --additional-iterations 2000)"
        shift
        python main.py train --no_compile "$@"
        ;;
    --duel)
        echo -e "${CYAN}Launching Arena mode...${NC}"
        shift
        python main.py duel "$@"
        ;;
    *)
        echo -e "${CYAN}🚀 Launching WebUI...${NC}"
        echo -e "  Open ${GREEN}http://localhost:8501${NC} in your browser"
        echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop"
        echo ""
        streamlit run "$PROJECT_DIR/app.py" --server.port 8501
        ;;
esac
