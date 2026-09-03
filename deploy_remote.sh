#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-}"
REMOTE_DIR="${2:-RoboCOIN-Object-Manager}"

usage() {
    cat <<'EOF'
Usage:
  ./deploy_remote.sh USER@HOST [REMOTE_DIR]

Examples:
  ./deploy_remote.sh collector@192.168.1.120
  ./deploy_remote.sh collector@192.168.1.120 projects/RoboCOIN-Object-Manager

REMOTE_DIR defaults to RoboCOIN-Object-Manager under the remote user's home.
Set SKIP_SYSTEM_PACKAGES=1 if Python, ffmpeg, git, and rsync are already installed.
EOF
}

if [[ "$TARGET" == "-h" || "$TARGET" == "--help" ]]; then
    usage
    exit 0
fi
if [[ -z "$TARGET" || $# -gt 2 ]]; then
    usage
    exit 2
fi
if [[ ! "$TARGET" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid target: $TARGET"
    echo "Expected USER@HOST, for example collector@192.168.1.120"
    exit 2
fi
if [[ ! "$REMOTE_DIR" =~ ^[A-Za-z0-9._/-]+$ ]] \
    || [[ "$REMOTE_DIR" == /* ]] \
    || [[ "/$REMOTE_DIR/" == *"/../"* ]]; then
    echo "Invalid remote directory: $REMOTE_DIR"
    echo "Use a safe path relative to the remote user's home."
    exit 2
fi

for command_name in ssh rsync; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Missing local command: $command_name"
        echo "Ubuntu/Debian: sudo apt install openssh-client rsync"
        exit 1
    fi
done

CONTROL_DIR="$(mktemp -d)"
CONTROL_SOCKET="$CONTROL_DIR/socket"
SSH_OPTIONS=(
    -o ControlMaster=auto
    -o ControlPersist=600
    -o "ControlPath=$CONTROL_SOCKET"
)
cleanup() {
    ssh "${SSH_OPTIONS[@]}" -O exit "$TARGET" >/dev/null 2>&1 || true
    rmdir "$CONTROL_DIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[1/5] Testing SSH connection to $TARGET"
ssh "${SSH_OPTIONS[@]}" "$TARGET" true

if [[ "${SKIP_SYSTEM_PACKAGES:-0}" != "1" ]]; then
    echo "[2/5] Installing required system packages on the new machine"
    echo "The new machine may ask for its sudo password."
    ssh "${SSH_OPTIONS[@]}" -t "$TARGET" \
        "sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv ffmpeg git rsync"
else
    echo "[2/5] Skipping remote system packages"
fi

echo "[3/5] Preparing remote directory: $REMOTE_DIR"
ssh "${SSH_OPTIONS[@]}" "$TARGET" "mkdir -p -- '$REMOTE_DIR'"

echo "[4/5] Copying project, models, datasets, caches, and manual results"
echo "Interrupted transfers can be resumed by running this command again."
rsync -a --partial --info=progress2 \
    -e "ssh -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=$CONTROL_SOCKET" \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    "$PROJECT_ROOT/" "$TARGET:$REMOTE_DIR/"

echo "[5/5] Creating the remote Python environment and checking deployment"
ssh "${SSH_OPTIONS[@]}" -t "$TARGET" \
    "cd '$REMOTE_DIR' && chmod +x setup.sh && PYTHON_BIN=python3.10 ./setup.sh && ./setup.sh --check"

cat <<EOF

Deployment completed.

The old machine is no longer needed. On the NEW machine, open a terminal and run:
  cd $REMOTE_DIR
  source .venv/bin/activate
  python viewer.py

Then open http://127.0.0.1:8888/review in a browser on the NEW machine.
EOF
