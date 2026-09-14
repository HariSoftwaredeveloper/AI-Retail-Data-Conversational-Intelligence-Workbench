"""Cleaning planner that constructs an explicit, auditable plan before mutation.
"""
from typing import List
import pandas as pd
from app.core.models import CleaningPlan, CleaningStep, Issue


def build_cleaning_plan(df: pd.DataFrame, issues: List[Issue], dataset_name: str, run_id: str) -> CleaningPlan:
    """Build a deterministic cleaning plan from detected issues."""
    steps: List[CleaningStep] = []
    step_counter = 1

    # 1. Whitespace & Unicode Cleanup Step
    whitespace_fields = [iss.field for iss in issues if iss.issue_type == "whitespace_case" and iss.field]
    # Always include text/string columns for standard strip
    str_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
    target_str_cols = list(set(whitespace_fields + str_cols))

    if target_str_cols:
        steps.append(
            CleaningStep(
                step_id=f"{dataset_name}_step_{step_counter:02d}_trim_whitespace",
                reason="Trim leading/trailing whitespace and normalize invisible characters in string columns",
                affected_fields=sorted(target_str_cols),
                risk="low",
                source="code",
                transformation_type="strip_whitespace",
                parameters={"columns": sorted(target_str_cols)},
            )
        )
        step_counter += 1

    # 2. Currency & Number Parsing Steps
    currency_issues = [iss for iss in issues if iss.issue_type == "currency_formats"]
    for iss in currency_issues:
        if iss.field:
            steps.append(
                CleaningStep(
                    step_id=f"{dataset_name}_step_{step_counter:02d}_parse_numeric_{iss.field}",
                    reason=f"Convert formatted currency or comma-separated string in '{iss.field}' to numeric float",
                    affected_fields=[iss.field],
                    risk="low",
                    source="code",
                    transformation_type="parse_currency_numeric",
                    parameters={"column": iss.field},
                )
            )
            step_counter += 1

    # 3. Inconsistent Date Standardization
    date_issues = [iss for iss in issues if iss.issue_type == "inconsistent_dates"]
    for iss in date_issues:
        if iss.field:
            steps.append(
                CleaningStep(
                    step_id=f"{dataset_name}_step_{step_counter:02d}_standardize_dates_{iss.field}",
                    reason=f"Standardize mixed date formats in '{iss.field}' to ISO-8601 (YYYY-MM-DD)",
                    affected_fields=[iss.field],
                    risk="medium",
                    source="code",
                    transformation_type="standardize_dates",
                    parameters={"column": iss.field, "output_format": "%Y-%m-%d"},
                )
            )
            step_counter += 1

    # 4. Inconsistent Category Casing & Spelling Variants
    cat_issues = [iss for iss in issues if iss.issue_type in ("suspicious_categories", "whitespace_case") and iss.field]
    seen_cat_cols = set()
    for iss in cat_issues:
        if iss.field and iss.field not in seen_cat_cols:
            col_lower = iss.field.lower()
            if any(term in col_lower for term in ["category", "type", "department", "segment", "region", "status"]):
                seen_cat_cols.add(iss.field)
                steps.append(
                    CleaningStep(
                        step_id=f"{dataset_name}_step_{step_counter:02d}_standardize_category_{iss.field}",
                        reason=f"Standardize casing and harmonize spelling variations in '{iss.field}'",
                        affected_fields=[iss.field],
                        risk="medium",
                        source="code",
                        transformation_type="standardize_categories",
                        parameters={"column": iss.field},
                    )
                )
                step_counter += 1

    # 5. Deduplication Step
    dup_issues = [iss for iss in issues if iss.issue_type == "duplicate_records"]
    if dup_issues:
        # Check if primary key duplicate or whole row duplicate
        pk_issues = [iss for iss in dup_issues if iss.field is not None]
        if pk_issues:
            pk_col = pk_issues[0].field
            steps.append(
                CleaningStep(
                    step_id=f"{dataset_name}_step_{step_counter:02d}_deduplicate_by_pk",
                    reason=f"Resolve duplicate records on primary key '{pk_col}' preserving most complete record",
                    affected_fields=[pk_col],
                    risk="high",
                    source="code",
                    transformation_type="deduplicate_by_pk",
                    parameters={"pk_column": pk_col},
                )
            )
        else:
            steps.append(
                CleaningStep(
                    step_id=f"{dataset_name}_step_{step_counter:02d}_deduplicate_exact_rows",
                    reason="Remove exact duplicate rows across all columns",
                    affected_fields=list(df.columns),
                    risk="medium",
                    source="code",
                    transformation_type="deduplicate_exact",
                    parameters={},
                )
            )
        step_counter += 1

    return CleaningPlan(
        run_id=run_id,
        dataset_name=dataset_name,
        steps=steps,
    )
