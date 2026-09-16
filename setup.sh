#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/collector.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/collector.env"
    set +a
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3.10 >/dev/null 2>&1; then
        PYTHON_BIN="python3.10"
    else
        PYTHON_BIN="python3"
    fi
fi
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export REQUIRE_CUDA
VLM_BACKEND="${ROBOCOIN_VLM_BACKEND:-local}"
REQUIREMENTS_FILE="${ROBOCOIN_REQUIREMENTS_FILE:-requirements-project.txt}"
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: ./setup.sh [--check]"
    exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Missing system dependency: ffmpeg"
    echo "Ubuntu/Debian: sudo apt install ffmpeg"
    exit 1
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "Unsupported Python: $PYTHON_VERSION (required: 3.10)"
    echo "Run with: PYTHON_BIN=python3.10 ./setup.sh"
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        echo "Virtual environment not found: $VENV_DIR"
        exit 1
    fi
    echo "Creating virtual environment: $VENV_DIR"
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        echo "python venv is unavailable; installing virtualenv for this user…"
        "$PYTHON_BIN" -m pip install --user --upgrade virtualenv
        "$PYTHON_BIN" -m virtualenv "$VENV_DIR"
    fi
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if [[ "$CHECK_ONLY" -eq 0 ]]; then
    # PyTorch 2.11 requires setuptools<82. Keep the bootstrap toolchain within
    # that range so repeated setup runs do not upgrade and immediately
    # downgrade setuptools while resolving the runtime requirements.
    "$VENV_PYTHON" -m pip install --upgrade pip "setuptools<82" wheel

    if ! "$VENV_PYTHON" -c 'import torch, torchvision; assert torch.version.cuda' >/dev/null 2>&1; then
        echo "Installing PyTorch/TorchVision from: $PYTORCH_INDEX_URL"
        "$VENV_PYTHON" -m pip install torch torchvision --index-url "$PYTORCH_INDEX_URL"
    fi

    "$VENV_PYTHON" -m pip install --no-build-isolation -r "$REQUIREMENTS_FILE"
    "$VENV_PYTHON" -m pip install -e ./sam3
    if ! "$VENV_PYTHON" -c 'import clip' >/dev/null 2>&1; then
        "$VENV_PYTHON" -m pip install git+https://github.com/openai/CLIP.git
    fi
    "$VENV_PYTHON" -m nltk.downloader -q wordnet
    "$VENV_PYTHON" migrate_paths.py --apply
    if [[ ! -f "models/realesrgan/RealESRGAN_x2plus.pth" ]]; then
        "$VENV_PYTHON" - <<'PY'
import hashlib
import urllib.request
from pathlib import Path

url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
expected = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
target = Path("models/realesrgan/RealESRGAN_x2plus.pth")
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(".pth.download")
print(f"Downloading RealESRGAN_x2plus: {url}")
urllib.request.urlretrieve(url, temporary)
actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
if actual != expected:
    temporary.unlink(missing_ok=True)
    raise SystemExit(f"RealESRGAN_x2plus checksum mismatch: {actual}")
temporary.replace(target)
PY
    fi
fi

"$VENV_PYTHON" - <<'PY'
import importlib
import os
import sys

required = [
    "torch", "torchvision", "cv2", "numpy", "PIL", "fastapi",
    "uvicorn", "nltk", "clip", "sam3",
]
if os.environ.get("ROBOCOIN_VLM_BACKEND", "local") == "local":
    required.append("transformers")
failed = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {exc}")
if failed:
    print("Environment check failed:")
    print("\n".join(f"  - {item}" for item in failed))
    raise SystemExit(1)

import torch
from super_resolution import check_dependencies
check_dependencies()
print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is unavailable; SAM3/SAM3.1 inference cannot run in the collector workflow.")
    if os.environ.get("REQUIRE_CUDA", "1") == "1":
        raise SystemExit(1)
PY

missing=0
for required_path in \
    "sam3_weights/sam3.pt" \
    "sam3_weights/sam3.1_multiplex.pt" \
    "models/realesrgan/RealESRGAN_x2plus.pth" \
    "models/clip/ViT-L-14.pt"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing model asset: $required_path"
        missing=1
    fi
done
if [[ "$VLM_BACKEND" == "local" ]]; then
    for required_path in \
        "models/Qwen3-VL-2B-Instruct/model.safetensors" \
        "models/Qwen3-VL-2B-Instruct/config.json"; do
        if [[ ! -e "$required_path" ]]; then
            echo "Missing model asset: $required_path"
            missing=1
        fi
    done
fi
if [[ ! -d "RoboCOIN_datasets" ]]; then
    echo "No dataset directory yet: RoboCOIN_datasets"
    echo "Run: $VENV_PYTHON download_head_videos.py --limit 10"
fi

echo
echo "Environment setup/check completed."
echo "Activate with: source .venv/bin/activate"
echo "Start viewer:  python viewer.py"
if [[ "$missing" -eq 1 ]]; then
    echo "Copy/download the model assets listed above before running the full pipeline."
    exit 1
fi
