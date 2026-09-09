#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG_FILE="${ROBOCOIN_CONFIG_FILE:-$PROJECT_ROOT/collector.env}"
if [[ -f "$CONFIG_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    set +a
fi

export ROBOCOIN_HARDWARE_PROFILE="${ROBOCOIN_HARDWARE_PROFILE:-local}"
export ROBOCOIN_VLM_BACKEND="${ROBOCOIN_VLM_BACKEND:-api}"
export VLM_MODEL="${VLM_MODEL:-glm-5.3-flash}"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    echo "尚未安装运行环境，请先执行：./setup.sh"
    exit 1
fi
if [[ "$ROBOCOIN_VLM_BACKEND" == "api" ]] \
    && { [[ -z "${VLM_API_BASE:-}" ]] || [[ -z "${VLM_API_KEY:-}" ]] \
        || [[ "${VLM_API_KEY:-}" == "请填写实际密钥" ]]; }; then
    echo "尚未配置 GLM API。"
    echo "请复制 collector.env.example 为 collector.env，并填写 VLM_API_BASE 和 VLM_API_KEY。"
    exit 1
fi

echo "正在检查显卡、依赖和模型文件……"
"$PROJECT_ROOT/setup.sh" --check

LOG_DIR="$PROJECT_ROOT/objects/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/viewer.log"
URL="http://127.0.0.1:8888"

if "$PROJECT_ROOT/.venv/bin/python" -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8888/review", timeout=1)' \
    >/dev/null 2>&1; then
    echo "审核程序已经运行，正在打开浏览器……"
    command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true
    exit 0
fi

echo "正在启动数采审核程序……"
"$PROJECT_ROOT/.venv/bin/python" viewer.py --host 127.0.0.1 --port 8888 \
    >>"$LOG_FILE" 2>&1 &
VIEWER_PID=$!
cleanup() {
    if kill -0 "$VIEWER_PID" >/dev/null 2>&1; then
        kill "$VIEWER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup INT TERM EXIT

for _attempt in $(seq 1 60); do
    if ! kill -0 "$VIEWER_PID" >/dev/null 2>&1; then
        echo "启动失败，请查看日志：$LOG_FILE"
        tail -n 30 "$LOG_FILE" || true
        exit 1
    fi
    if "$PROJECT_ROOT/.venv/bin/python" -c \
        'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8888/review", timeout=1)' \
        >/dev/null 2>&1; then
        echo "启动成功：$URL"
        echo "请保持本窗口开启；关闭窗口会停止审核程序。"
        command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 || true
        wait "$VIEWER_PID"
        exit $?
    fi
    sleep 0.5
done

echo "启动超时，请查看日志：$LOG_FILE"
tail -n 30 "$LOG_FILE" || true
exit 1
