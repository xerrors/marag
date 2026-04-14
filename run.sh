#!/bin/bash
set -e
export $(cat .env | grep -v '^#' | xargs)

CMD=${1:-batch}
DATASET=${2:-demo}
WORKERS=${3:-10}
LIMIT=${4:-0}

export RAG_MODEL=${RAG_MODEL:-gpt-4o-mini}

case $CMD in
  index)
    uv run python scripts/build_index.py --dataset "$DATASET" \
      --model Qwen/Qwen3-Embedding-0.6B \
      --device cuda:0
    ;;
  batch)
    echo "Running $CMD $RAG_MODEL: $DATASET with $WORKERS workers and limit: $LIMIT"
    uv run python scripts/batch_runner.py --dataset "$DATASET" --limit "$LIMIT" --workers "$WORKERS"
    ;;
  *)
    echo "Usage: $0 [index|batch] [dataset] [workers] [limit]"
    ;;
esac
