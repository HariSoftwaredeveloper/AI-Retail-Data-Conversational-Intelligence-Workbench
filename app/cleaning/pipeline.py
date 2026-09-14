"""Cleaning pipeline orchestrator with idempotency validation and artifact persistence.
"""
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
from app.core.config import DATA_DIR, ARTIFACTS_DIR
from app.core.models import CleaningPlan, DatasetProfile, DeltaValidation, Issue
from app.profiling.profiler import profile_dataframe
from app.profiling.issue_detector import detect_issues
from app.cleaning.planner import build_cleaning_plan
from app.cleaning.transformers import (
    strip_whitespace_and_cleanup,
    parse_currency_and_numbers,
    parse_mixed_dates,
    harmonize_category_variants,
    deduplicate_exact_rows,
    deduplicate_by_primary_key,
)


def execute_cleaning_plan(df: pd.DataFrame, plan: CleaningPlan) -> Tuple[pd.DataFrame, CleaningPlan]:
    """Execute each step in the cleaning plan deterministically."""
    df_current = df.copy()

    for step in plan.steps:
        try:
            if step.transformation_type == "strip_whitespace":
                cols = step.parameters.get("columns", [])
                df_current = strip_whitespace_and_cleanup(df_current, cols)
                step.status = "applied"

            elif step.transformation_type == "parse_currency_numeric":
                col = step.parameters.get("column")
                df_current = parse_currency_and_numbers(df_current, col)
                step.status = "applied"

            elif step.transformation_type == "standardize_dates":
                col = step.parameters.get("column")
                fmt = step.parameters.get("output_format", "%Y-%m-%d")
                df_current = parse_mixed_dates(df_current, col, fmt)
                step.status = "applied"

            elif step.transformation_type == "standardize_categories":
                col = step.parameters.get("column")
                df_current = harmonize_category_variants(df_current, col)
                step.status = "applied"

            elif step.transformation_type == "deduplicate_exact":
                df_current = deduplicate_exact_rows(df_current)
                step.status = "applied"

            elif step.transformation_type == "deduplicate_by_pk":
                pk = step.parameters.get("pk_column")
                df_current = deduplicate_by_primary_key(df_current, pk)
                step.status = "applied"

            else:
                step.status = "skipped"

        except Exception as e:
            step.status = "failed"
            step.reason = f"{step.reason} [Error: {str(e)}]"

    return df_current, plan


def run_pipeline(
    df_raw: pd.DataFrame,
    dataset_name: str,
    run_id: str,
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full cleaning pipeline: profile -> detect -> plan -> transform -> validate idempotency -> persist."""
    target_artifacts_dir = artifacts_dir if artifacts_dir else ARTIFACTS_DIR / run_id
    target_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Profile before
    profile_before = profile_dataframe(df_raw, dataset_name)

    # 2. Detect issues
    issues = detect_issues(df_raw, dataset_name)

    # 3. Build cleaning plan
    plan = build_cleaning_plan(df_raw, issues, dataset_name, run_id)

    # 4. Execute cleaning plan
    df_clean, executed_plan = execute_cleaning_plan(df_raw, plan)

    # 5. Profile after
    profile_after = profile_dataframe(df_clean, f"{dataset_name}_clean")

    # 6. Check unresolved issues
    post_issues = detect_issues(df_clean, f"{dataset_name}_clean")
    unresolved = [iss.description for iss in post_issues if iss.severity == "high"]

    # 7. Idempotency test (Re-run cleaner on clean output)
    plan_rerun = build_cleaning_plan(df_clean, post_issues, f"{dataset_name}_clean", f"{run_id}_rerun")
    df_rerun, _ = execute_cleaning_plan(df_clean, plan_rerun)
    idempotent = (len(df_clean) == len(df_rerun)) and (list(df_clean.columns) == list(df_rerun.columns))

    # 8. Schema & Row deltas
    row_delta = len(df_clean) - len(df_raw)
    schema_before = list(df_raw.columns)
    schema_after = list(df_clean.columns)
    schema_mutated = (schema_before != schema_after)

    validation = DeltaValidation(
        rows_before=len(df_raw),
        rows_after=len(df_clean),
        row_delta=row_delta,
        schema_before=schema_before,
        schema_after=schema_after,
        schema_mutated=schema_mutated,
        unresolved_issues=unresolved,
        idempotent=idempotent,
    )

    # 9. Persist artifacts
    raw_path = target_artifacts_dir / f"{dataset_name}_raw.csv"
    clean_path = target_artifacts_dir / f"{dataset_name}_clean.csv"
    metadata_path = target_artifacts_dir / f"{dataset_name}_metadata.json"

    df_raw.to_csv(raw_path, index=False)
    df_clean.to_csv(clean_path, index=False)

    metadata = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "raw_artifact": str(raw_path),
        "clean_artifact": str(clean_path),
        "rows_before": len(df_raw),
        "rows_after": len(df_clean),
        "idempotent": idempotent,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "dataset": {
            "name": dataset_name,
            "raw_path": str(raw_path),
            "clean_path": str(clean_path),
        },
        "profile_before": profile_before.model_dump(),
        "issues": [iss.model_dump() for iss in issues],
        "cleaning_plan": [step.model_dump() for step in executed_plan.steps],
        "profile_after": profile_after.model_dump(),
        "validation": validation.model_dump(),
        "artifacts": {
            "raw_csv": str(raw_path),
            "clean_csv": str(clean_path),
            "metadata_json": str(metadata_path),
        },
        "df_clean": df_clean,
    }
