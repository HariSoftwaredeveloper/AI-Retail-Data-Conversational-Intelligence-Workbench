"""Tests for batch evaluation entrypoint and output format compliance.
"""
import json
from pathlib import Path
from app.evaluate import run_batch_evaluation


def test_batch_evaluator_end_to_end(tmp_path: Path):
    # Setup test case
    cases_file = tmp_path / "cases.json"
    results_file = tmp_path / "results.json"
    art_dir = tmp_path / "artifacts"
    csv_file = tmp_path / "test_orders.csv"

    csv_file.write_text(
        "order_id,region,total_amount,status\n"
        "O1,West,$100.00,Completed\n"
        "O2,East,$50.00,Returned\n"
        "O1,West,$100.00,Completed\n"
    )

    cases = [{
        "id": "eval_test_01",
        "dataset_name": "test_orders",
        "dataset_path": str(csv_file),
        "chat_questions": [
            {"question": "What is the total revenue?", "expected": {"status": "ok"}}
        ]
    }]

    with open(cases_file, "w") as f:
        json.dump(cases, f)

    run_batch_evaluation(str(cases_file), str(results_file), str(art_dir))

    assert results_file.exists()
    with open(results_file, "r") as f:
        res = json.load(f)

    record = res[0] if isinstance(res, list) else res
    assert "dataset" in record
    assert "profile_before" in record
    assert "cleaning_plan" in record
    assert "profile_after" in record
    assert "validation" in record
    assert "chat_evaluation" in record
    assert record["validation"]["idempotent"] is True
