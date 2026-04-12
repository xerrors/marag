#!/bin/bash
set -e
export $(cat .env | grep -v '^#' | xargs)

CMD=${1:-batch}
DATASET=${2:-demo}
WORKERS=${3:-10}
LIMIT=${4:-0}

export RAG_MODEL=gpt-4o-mini
export EVAL_MODEL=gpt-5-mini

case $CMD in
  index)
    uv run python scripts/build_index.py --dataset "$DATASET" \
      --model Qwen/Qwen3-Embedding-0.6B \
      --device cuda:0
    ;;
  batch)
    uv run python scripts/batch_runner.py --dataset "$DATASET" --limit "$LIMIT" --workers "$WORKERS"
    ;;
  eval)
    uv run python scripts/eval.py --predictions "results/$DATASET/predictions.jsonl" --workers "$WORKERS"
    ;;
  *)
    echo "Usage: $0 [index|batch|eval] [dataset] [workers] [limit]"
    ;;
esac
