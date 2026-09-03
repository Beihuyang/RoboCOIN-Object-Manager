#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

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

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        echo "Virtual environment not found: $VENV_DIR"
        exit 1
    fi
    echo "Creating virtual environment: $VENV_DIR"
    if ! "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"; then
        echo "python venv is unavailable; installing virtualenv for this user…"
        "$PYTHON_BIN" -m pip install --user --upgrade virtualenv
        "$PYTHON_BIN" -m virtualenv --system-site-packages "$VENV_DIR"
    fi
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if [[ "$CHECK_ONLY" -eq 0 ]]; then
    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

    if ! "$VENV_PYTHON" -c 'import torch, torchvision' >/dev/null 2>&1; then
        echo "Installing PyTorch/TorchVision from: $PYTORCH_INDEX_URL"
        "$VENV_PYTHON" -m pip install torch torchvision --index-url "$PYTORCH_INDEX_URL"
    fi

    "$VENV_PYTHON" -m pip install --no-build-isolation -r requirements-project.txt
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
import sys

required = (
    "torch", "torchvision", "cv2", "numpy", "PIL", "fastapi",
    "uvicorn", "transformers", "nltk", "clip", "sam3",
)
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
    print("Warning: CUDA is unavailable; SAM3/Qwen inference will be very slow or unusable.")
PY

missing=0
for required_path in \
    "sam3_weights/sam3.pt" \
    "sam3_weights/sam3.1_multiplex.pt" \
    "models/Qwen3-VL-2B-Instruct" \
    "models/realesrgan/RealESRGAN_x2plus.pth"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing model asset: $required_path"
        missing=1
    fi
done
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
fi
