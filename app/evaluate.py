"""Mandatory batch evaluation CLI entrypoint for hidden datasets and test suites.
Usage:
    python -m app.evaluate --input cases.json --output results.json --artifacts-dir ./evaluation_output
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from app.cleaning.pipeline import run_pipeline
from app.assistant.planner import QueryPlanner
from app.assistant.validator import validate_plan
from app.assistant.executor import execute_plan
from app.assistant.explainer import format_grounded_answer
from app.assistant.session import SessionManager
from app.core.models import ChatTurn


def evaluate_single_case(case: Dict[str, Any], artifacts_dir: Path, case_index: int) -> Dict[str, Any]:
    """Evaluate an individual test case, ensuring errors do not abort the batch."""
    case_id = case.get("id", f"case_{case_index:03d}")
    dataset_path_str = case.get("dataset_path") or case.get("csv_path")
    dataset_name = case.get("dataset_name", f"dataset_{case_index}")
    case_artifacts_dir = artifacts_dir / case_id
    case_artifacts_dir.mkdir(parents=True, exist_ok=True)

    result_payload: Dict[str, Any] = {
        "case_id": case_id,
        "dataset": {},
        "profile_before": {},
        "issues": [],
        "cleaning_plan": [],
        "profile_after": {},
        "validation": {},
        "chat_evaluation": [],
        "artifacts": {},
        "status": "ok",
    }

    try:
        # 1. Ingest Data
        if not dataset_path_str or not Path(dataset_path_str).exists():
            # Check if multi-dataset or inline data
            multi_datasets = case.get("datasets", {})
            if not multi_datasets:
                return {
                    **result_payload,
                    "status": "error",
                    "error": f"Dataset path '{dataset_path_str}' not found and no multi-datasets provided.",
                }
            loaded_dfs: Dict[str, pd.DataFrame] = {}
            for d_name, d_path in multi_datasets.items():
                if Path(d_path).exists():
                    loaded_dfs[d_name] = pd.read_csv(d_path)
            primary_name = list(loaded_dfs.keys())[0]
            df_raw = loaded_dfs[primary_name]
            dataset_name = primary_name
        else:
            df_raw = pd.read_csv(dataset_path_str)
            loaded_dfs = {dataset_name: df_raw}

        # 2. Run Cleaning Pipeline
        pipeline_output = run_pipeline(
            df_raw=df_raw,
            dataset_name=dataset_name,
            run_id=case_id,
            artifacts_dir=case_artifacts_dir,
        )

        result_payload["dataset"] = pipeline_output["dataset"]
        result_payload["profile_before"] = pipeline_output["profile_before"]
        result_payload["issues"] = pipeline_output["issues"]
        result_payload["cleaning_plan"] = pipeline_output["cleaning_plan"]
        result_payload["profile_after"] = pipeline_output["profile_after"]
        result_payload["validation"] = pipeline_output["validation"]
        result_payload["artifacts"] = pipeline_output["artifacts"]

        # Register both raw and cleaned datasets in active workspace
        active_datasets = dict(loaded_dfs)
        active_datasets[f"{dataset_name}_clean"] = pipeline_output["df_clean"]
        active_datasets[f"{dataset_name}_raw"] = df_raw

        # If secondary datasets were given, clean them as well
        for d_name, df_sub in loaded_dfs.items():
            if d_name != dataset_name:
                sub_pipe = run_pipeline(
                    df_raw=df_sub,
                    dataset_name=d_name,
                    run_id=f"{case_id}_{d_name}",
                    artifacts_dir=case_artifacts_dir,
                )
                active_datasets[f"{d_name}_clean"] = sub_pipe["df_clean"]
                active_datasets[f"{d_name}_raw"] = df_sub

        # 3. Chat Assistant Evaluation
        questions = case.get("chat_questions", case.get("questions", []))
        chat_eval_results: List[Dict[str, Any]] = []

        planner = QueryPlanner()
        session_mgr = SessionManager(sessions_dir=case_artifacts_dir / "sessions")
        session = session_mgr.get_or_create_session(f"eval_{case_id}")
        session.active_dataset = f"{dataset_name}_clean"

        for q_item in questions:
            q_text = q_item if isinstance(q_item, str) else q_item.get("question", "")
            expected = q_item.get("expected", {}) if isinstance(q_item, dict) else {}
            t0 = time.time()

            try:
                # Plan
                plan = planner.plan(q_text, active_datasets, session)
                # Validate
                is_valid, val_errors, sanitized_plan = validate_plan(plan, active_datasets)

                evidence = None
                status = "ok"

                if not is_valid:
                    status = "validation_error"
                    answer = {"summary": f"Plan validation failed: {'; '.join(val_errors)}", "grounded": False}
                elif sanitized_plan.intent in ("refuse", "clarify"):
                    status = sanitized_plan.intent
                    answer = format_grounded_answer(sanitized_plan, None, q_text)
                else:
                    # Execute
                    evidence = execute_plan(sanitized_plan, active_datasets)
                    answer = format_grounded_answer(sanitized_plan, evidence, q_text)

                duration_ms = round((time.time() - t0) * 1000, 2)

                trace = {
                    "duration_ms": duration_ms,
                    "validation_errors": val_errors,
                    "rows_considered": evidence.row_count_considered if evidence else 0,
                    "operations": evidence.operations if evidence else [],
                }

                turn = ChatTurn(
                    question=q_text,
                    plan=sanitized_plan.model_dump(),
                    answer=answer,
                    trace=trace,
                    status=status,
                )

                # Update session
                session = session_mgr.append_turn(
                    session_id=session.session_id,
                    turn=turn,
                    new_active_dataset=sanitized_plan.dataset,
                    new_filters=sanitized_plan.filters if sanitized_plan.filters else None,
                )

                eval_record = {
                    "question": q_text,
                    "plan": sanitized_plan.model_dump(),
                    "answer": answer,
                    "trace": trace,
                    "status": status,
                }

                # Evaluate against expected conditions if specified
                if expected:
                    passed = True
                    if "status" in expected and expected["status"] != status:
                        passed = False
                    if "scalar_value" in expected:
                        ans_val = answer.get("scalar_value")
                        if ans_val != expected["scalar_value"]:
                            passed = False
                    eval_record["expected_match"] = passed

                chat_eval_results.append(eval_record)

            except Exception as q_err:
                chat_eval_results.append({
                    "question": q_text,
                    "plan": {},
                    "answer": {"error": str(q_err)},
                    "trace": {"duration_ms": round((time.time() - t0) * 1000, 2)},
                    "status": "error",
                })

        result_payload["chat_evaluation"] = chat_eval_results

    except Exception as e:
        result_payload["status"] = "error"
        result_payload["error"] = str(e)

    return result_payload


def run_batch_evaluation(input_file: str, output_file: str, artifacts_dir: str) -> None:
    """Run batch evaluation on cases.json and write standardized results.json."""
    in_path = Path(input_file).resolve()
    out_path = Path(output_file).resolve()
    art_path = Path(artifacts_dir).resolve()

    art_path.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Error: Input file '{in_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(in_path, "r", encoding="utf-8") as f:
        cases_data = json.load(f)

    cases = cases_data if isinstance(cases_data, list) else [cases_data]
    print(f"Loaded {len(cases)} evaluation case(s) from '{in_path}'.")

    results = []
    for idx, case in enumerate(cases, 1):
        print(f"Running evaluation case {idx}/{len(cases)}: {case.get('id', f'case_{idx}')}...")
        case_res = evaluate_single_case(case, art_path, idx)
        results.append(case_res)

    # If input was a single case object, match schema directly; otherwise return list or wrapped object
    output_content = results[0] if len(results) == 1 and not isinstance(cases_data, list) else results

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_content, f, indent=2)

    print(f"Evaluation complete! Results saved to '{out_path}'. Artifacts saved to '{art_path}'.")
    return results


def main():
    parser = argparse.ArgumentParser(description="AI Retail Data Workbench Batch Evaluator")
    parser.add_argument("--input", required=True, help="Path to input cases.json")
    parser.add_argument("--output", required=True, help="Path to output results.json")
    parser.add_argument("--artifacts-dir", default="./evaluation_output", help="Directory to save artifacts")

    args = parser.parse_args()
    run_batch_evaluation(args.input, args.output, args.artifacts_dir)


if __name__ == "__main__":
    main()
