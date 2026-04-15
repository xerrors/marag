#!/usr/bin/env python3
"""
Batch Runner for ARAG - Supports concurrent execution and checkpoint resume.

Usage:
    python scripts/batch_runner.py \
        --config configs/local.toml \
        --dataset musique
"""

import json
import argparse
import logging
import os
import re
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from arag import BaseAgent, Config, LLMClient, ToolRegistry, resolve_llm_profile
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool

logging.basicConfig(level=logging.ERROR)

INVALID_CLOSING_TAG_RE = re.compile(r"</[^>]+>")


class BatchRunner:
    """Batch runner with concurrent execution and checkpoint resume."""

    def __init__(
        self,
        config: Config,
        dataset: str,
        limit: int | None = None,
        num_workers: int = 10,
        verbose: bool = False,
    ):
        self.config = config
        self.data = config["data"][dataset]

        output_name = os.environ.get("RAG_MODEL", "default")
        if output_suffix := os.environ.get("ARAG_OUTPUT_SUFFIX", "").strip():
            output_name = f"{output_name}-{output_suffix}"

        output_dir_base = os.environ.get("ARAG_OUTPUT_DIR", "results")
        self.output_dir = Path(output_dir_base) / dataset / output_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.num_workers = num_workers
        self.verbose = verbose

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.write_lock = Lock()

        self.questions = self._load_questions()

        # Pre-initialize shared tools (load embedding model only once)
        self._shared_tools = self._init_shared_tools()

        # Load system prompt once
        prompt_file = Path(__file__).parent.parent / "src/arag/agent/prompts/default.txt"
        if prompt_file.exists():
            self._system_prompt = prompt_file.read_text()
        else:
            self._system_prompt = "You are a helpful assistant."

    def _init_shared_tools(self) -> ToolRegistry:
        """Initialize shared tools (embedding model loaded only once)."""
        chunks_file = self.data["chunks_file"]
        index_dir = self.data["index_dir"]
        index_file = Path(index_dir) / "sentence_index.pkl"

        if not index_file.exists():
            raise FileNotFoundError(
                f"Required semantic index file not found: {index_file}. "
                "Please build the index first (e.g., run scripts/build_index.py)."
            )

        tools = ToolRegistry()
        tools.register(KeywordSearchTool(chunks_file=chunks_file))
        tools.register(ReadChunkTool(chunks_file=chunks_file))

        embedding = self.config["embedding"]
        print(f"Loading embedding model: {embedding['model']}")
        tools.register(
            SemanticSearchTool(
                chunks_file=chunks_file,
                index_dir=index_dir,
                model_name=embedding["model"],
                device=embedding.get("device"),
            )
        )
        print("Embedding model loaded successfully!")

        return tools

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load questions from file."""
        with open(Path(self.data["questions_file"]), encoding="utf-8") as f:
            questions = json.load(f)

        if self.limit:
            questions = questions[: self.limit]

        return questions

    def _load_completed_qids(self) -> set:
        """Load completed question IDs for checkpoint resume."""
        completed_qids = set()
        kept_lines = []
        dirty = False

        if not self.predictions_file.exists():
            return completed_qids

        try:
            with open(self.predictions_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        dirty = True
                        continue
                    try:
                        data = json.loads(line)
                        if not self._is_completed_prediction(data):
                            dirty = True
                            continue

                        kept_lines.append(json.dumps(data, ensure_ascii=False))
                        qid = data.get("qid") or data.get("id")
                        if qid is not None:
                            completed_qids.add(qid)
                    except json.JSONDecodeError:
                        dirty = True
                        continue
        except Exception as e:
            print(f"Warning: Error loading completed data: {e}")

        if dirty:
            content = "\n".join(kept_lines)
            if content:
                content += "\n"
            with open(self.predictions_file, "w", encoding="utf-8") as f:
                f.write(content)

        return completed_qids

    def _is_completed_prediction(self, prediction: dict[str, Any]) -> bool:
        """Return True only for predictions that should be skipped on resume."""
        if "question" not in prediction:
            return False

        pred_answer = prediction.get("pred_answer")
        if not isinstance(pred_answer, str):
            return False

        pred_answer = pred_answer.strip()
        if not pred_answer:
            return False
        if pred_answer.startswith("Error:"):
            return False
        if "tool_call" in pred_answer:
            return False
        if INVALID_CLOSING_TAG_RE.search(pred_answer):
            return False

        # Additional validity checks: ensure pred_answer is a meaningful non-empty string
        if len(pred_answer) < 2:
            return False

        return True

    def _append_prediction(self, prediction: dict[str, Any]):
        """Append prediction to file (thread-safe)."""
        pred_answer = prediction.get("pred_answer", "")
        if isinstance(pred_answer, str) and pred_answer.startswith("Error:"):
            return

        with self.write_lock:
            with open(self.predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    def _create_agent(self) -> BaseAgent:
        """Create agent instance with shared tools."""
        client = LLMClient(**resolve_llm_profile(self.config, role="rag"))

        agent_config = self.config["agent"]

        return BaseAgent(
            llm_client=client,
            tools=self._shared_tools,  # Use shared tools
            system_prompt=self._system_prompt,
            max_loops=agent_config.get("max_loops", 10),
            max_token_budget=agent_config.get("max_token_budget", 128000),
            verbose=self.verbose,
        )

    def _process_one(self, item: dict[str, Any], agent: BaseAgent) -> dict[str, Any]:
        """Process one question."""
        qid = item.get("qid") or item.get("id")
        question = item.get("question", "")
        gold_answer = item.get("answer", item.get("gold_answer", ""))

        try:
            result = agent.run(question)

            return {
                "qid": qid,
                "question": question,
                "trajectory": result["trajectory"],
                "gold_answer": gold_answer,
                "pred_answer": result["answer"],
                "total_cost": result["total_cost"],
                "loops": result["loops"],
                "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
                "retrieval_logs": result.get("retrieval_logs", []),
                "chunks_read_count": result.get("chunks_read_count", 0),
                "chunks_read_ids": result.get("chunks_read_ids", []),
            }
        except Exception as e:
            return {
                "qid": qid,
                "question": question,
                "trajectory": [],
                "gold_answer": gold_answer,
                "pred_answer": f"Error: {str(e)}",
                "total_cost": 0,
                "loops": 0,
                "total_retrieved_tokens": 0,
                "retrieval_logs": [],
                "chunks_read_count": 0,
                "chunks_read_ids": [],
                "error": str(e),
            }

    def run(self):
        """Run batch processing."""
        completed_qids = self._load_completed_qids()

        # Filter pending questions
        pending = [q for q in self.questions if (q.get("qid") or q.get("id")) not in completed_qids]

        print(f"Total questions: {len(self.questions)}")
        print(f"Completed: {len(completed_qids)}")
        print(f"Pending: {len(pending)}")

        if not pending:
            print("All questions completed!")
            return

        print(f"Starting with {self.num_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}

            for item in pending:
                agent = self._create_agent()
                future = executor.submit(self._process_one, item, agent)
                futures[future] = item.get("qid") or item.get("id")

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task_id = progress.add_task("Run", total=len(pending))

                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        result = future.result()
                        self._append_prediction(result)
                    except Exception as e:
                        print(f"Error processing {qid}: {e}")
                    finally:
                        progress.advance(task_id)

        print(f"\nResults saved to: {self.predictions_file}")


def main():
    parser = argparse.ArgumentParser(description="ARAG Batch Runner")
    parser.add_argument("--config", "-c", default="configs/local.toml", help="Config file path")
    parser.add_argument("--dataset", "-d", default="demo", help="Dataset name")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Number of workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = Config.from_file(args.config)

    runner = BatchRunner(
        config=config,
        dataset=args.dataset,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose,
    )

    runner.run()


if __name__ == "__main__":
    main()
