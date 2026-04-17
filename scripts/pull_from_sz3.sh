#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 从远程 sz3 的当前工作目录同步到本地当前目录
# 用法：
#   ./scripts/pull_from_sz3.sh          默认同步，不删除本地文件
#   ./scripts/pull_from_sz3.sh --delete 同步并删除本地多余文件
# =============================================================================

REMOTE_HOST="sz3"
REMOTE_DIR="/home/zwj/workspace/repos/marag"
LOCAL_DIR=$(pwd)

# 解析参数
DELETE_FLAG=""
if [[ "${1:-}" == "--delete" ]]; then
    DELETE_FLAG="--delete"
fi

echo "=================================="
echo "  远程: $REMOTE_HOST:$REMOTE_DIR"
echo "  本地: $LOCAL_DIR"
if [[ -n "$DELETE_FLAG" ]]; then
    echo "  模式: 同步 + 删除本地多余文件"
else
    echo "  模式: 同步（保留本地多余文件）"
fi
echo "=================================="

# 确认执行
read -r -p "确认同步? [Y/n] " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo "已取消"
    exit 0
fi

# rsync 同步
# -a: 归档模式（保留权限、时间戳等）
# -v: 详细输出
# -z: 压缩传输
rsync -avz ${DELETE_FLAG} \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='.mypy_cache' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.DS_Store' \
    --exclude='node_modules' \
    --exclude='*.log' \
    "$REMOTE_HOST:$REMOTE_DIR/" \
    "$LOCAL_DIR/"

echo ""
echo "同步完成"
