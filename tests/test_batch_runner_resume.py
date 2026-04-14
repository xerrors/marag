"""Tests for batch runner checkpoint resume behavior."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from threading import Lock


def load_batch_runner_class():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "batch_runner.py"
    spec = importlib.util.spec_from_file_location("batch_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.BatchRunner


BatchRunner = load_batch_runner_class()


def make_runner(tmp_path: Path, records: list[dict]) -> BatchRunner:
    predictions_file = tmp_path / "predictions.jsonl"
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    if content:
        content += "\n"
    predictions_file.write_text(content, encoding="utf-8")

    runner = BatchRunner.__new__(BatchRunner)
    runner.predictions_file = predictions_file
    runner.write_lock = Lock()
    return runner


def test_load_completed_qids_only_keeps_valid_answers(tmp_path):
    runner = make_runner(
        tmp_path,
        [
            {"qid": "ok", "question": "q1", "pred_answer": "normal final answer"},
            {"qid": "error", "question": "q2", "pred_answer": "Error: failed to answer"},
            {"qid": "tool", "question": "q3", "pred_answer": "I saw a tool_call in output"},
            {"qid": "tag", "question": "q4", "pred_answer": "answer </think> leaked"},
            {"qid": "blank", "question": "q5", "pred_answer": "   "},
        ],
    )

    assert runner._load_completed_qids() == {"ok"}
    assert runner.predictions_file.read_text(encoding="utf-8") == (
        json.dumps(
            {"qid": "ok", "question": "q1", "pred_answer": "normal final answer"},
            ensure_ascii=False,
        )
        + "\n"
    )


def test_is_completed_prediction_rejects_invalid_pred_answer_patterns(tmp_path):
    runner = make_runner(tmp_path, [])

    assert runner._is_completed_prediction({"question": "q", "pred_answer": "valid"}) is True
    assert runner._is_completed_prediction({"question": "q", "pred_answer": "Error: boom"}) is False
    assert runner._is_completed_prediction({"question": "q", "pred_answer": "contains tool_call"}) is False
    assert runner._is_completed_prediction({"question": "q", "pred_answer": "contains </answer>"}) is False
    assert runner._is_completed_prediction({"question": "q", "pred_answer": ""}) is False


def test_append_prediction_skips_error_records(tmp_path):
    runner = make_runner(tmp_path, [])

    runner._append_prediction({"qid": "ok", "question": "q1", "pred_answer": "final"})
    runner._append_prediction({"qid": "bad", "question": "q2", "pred_answer": "Error: boom"})

    assert runner.predictions_file.read_text(encoding="utf-8") == (
        json.dumps({"qid": "ok", "question": "q1", "pred_answer": "final"}, ensure_ascii=False)
        + "\n"
    )
