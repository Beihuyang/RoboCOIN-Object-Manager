#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${ROBOCOIN_COLLECTOR_PACKAGE:-$PROJECT_ROOT/../RoboCOIN-Collector}"
TARGET="${1:-}"
REMOTE_DIR="${2:-RoboCOIN-Collector}"

if [[ -z "$TARGET" || $# -gt 2 ]]; then
    echo "用法：./deploy_collector_ssh.sh 用户名@数采电脑IP [目标目录]"
    exit 2
fi
if [[ ! -d "$PACKAGE_DIR" ]]; then
    echo "找不到部署目录：$PACKAGE_DIR"
    echo "请先运行：.venv/bin/python build_collector_package.py"
    exit 1
fi
if [[ "$REMOTE_DIR" == /* || "/$REMOTE_DIR/" == *"/../"* ]]; then
    echo "目标目录必须是数采员用户主目录下的安全相对路径。"
    exit 2
fi

echo "正在检查数采员电脑并安装系统依赖……"
ssh -t "$TARGET" \
    "sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv ffmpeg git rsync"

echo "正在通过 SSH 传输数采员完整程序……"
rsync -a --partial --info=progress2 \
    --exclude='.venv/' \
    --exclude='collector.env' \
    --exclude='objects/logs/' \
    "$PACKAGE_DIR/" "$TARGET:$REMOTE_DIR/"

echo "正在数采员电脑创建运行环境……"
ssh -t "$TARGET" \
    "cd '$REMOTE_DIR' && chmod +x setup.sh start_collector.sh && \
     ROBOCOIN_VLM_BACKEND=api ROBOCOIN_REQUIREMENTS_FILE=requirements-collector.txt ./setup.sh"

echo "部署完成。请在数采员电脑填写 $REMOTE_DIR/collector.env，然后运行："
echo "  cd $REMOTE_DIR && ./start_collector.sh"
