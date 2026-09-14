"""Strict allow-list validator for generated query plans.
"""
from typing import Dict, List, Tuple
import pandas as pd
from app.core.models import QueryPlan
from app.assistant.dsl import ALLOWED_OPERATORS, ALLOWED_AGGREGATIONS, ALLOWED_INTENTS
from app.core.config import MAX_QUERY_LIMIT, DEFAULT_QUERY_LIMIT


def validate_plan(
    plan: QueryPlan,
    loaded_datasets: Dict[str, pd.DataFrame]
) -> Tuple[bool, List[str], QueryPlan]:
    """Validate query plan against allow-lists and active schema.
    Returns: (is_valid, errors, sanitized_plan)
    """
    errors: List[str] = []
    sanitized = plan.model_copy(deep=True)

    # 1. Intent validation
    if sanitized.intent not in ALLOWED_INTENTS:
        errors.append(f"Disallowed plan intent: '{sanitized.intent}'")

    if sanitized.intent in ("clarify", "refuse"):
        return (len(errors) == 0, errors, sanitized)

    # 2. Target dataset validation
    if sanitized.dataset not in loaded_datasets:
        errors.append(f"Target dataset '{sanitized.dataset}' is not currently loaded in workbench.")
        return False, errors, sanitized

    base_df = loaded_datasets[sanitized.dataset]
    available_columns = set(base_df.columns)

    # 3. Joins validation
    for join in sanitized.joins:
        if join.dataset not in loaded_datasets:
            errors.append(f"Join dataset '{join.dataset}' is not loaded.")
            continue
        if join.left_on not in available_columns:
            errors.append(f"Join left_on key '{join.left_on}' missing from '{sanitized.dataset}'")
        right_df = loaded_datasets[join.dataset]
        if join.right_on not in right_df.columns:
            errors.append(f"Join right_on key '{join.right_on}' missing from '{join.dataset}'")
        # Include joined columns in available columns
        available_columns.update(right_df.columns)

    # 4. Filters validation
    for flt in sanitized.filters:
        if flt.field not in available_columns:
            errors.append(f"Filter field '{flt.field}' does not exist in schema.")
        if flt.op not in ALLOWED_OPERATORS:
            errors.append(f"Filter operator '{flt.op}' is not allowed.")

    # 5. Group By validation
    for gb in sanitized.group_by:
        if gb not in available_columns:
            errors.append(f"Group by field '{gb}' does not exist in schema.")

    # 6. Metrics validation
    for metric in sanitized.metrics:
        if metric.agg not in ALLOWED_AGGREGATIONS:
            errors.append(f"Aggregation '{metric.agg}' is not allowed.")
        if metric.field not in available_columns and metric.field != "*":
            errors.append(f"Metric field '{metric.field}' does not exist in schema.")

    # 7. Sort validation
    for s in sanitized.sort:
        metric_aliases = {m.alias for m in sanitized.metrics if m.alias}
        metric_fields = {m.field for m in sanitized.metrics}
        valid_sort_targets = available_columns | metric_aliases | metric_fields
        if s.field not in valid_sort_targets:
            errors.append(f"Sort field '{s.field}' is neither a dataset column nor computed metric.")
        if s.dir.lower() not in ("asc", "desc"):
            s.dir = "desc"

    # 8. Limit clamping
    if sanitized.limit <= 0:
        sanitized.limit = DEFAULT_QUERY_LIMIT
    elif sanitized.limit > MAX_QUERY_LIMIT:
        sanitized.limit = MAX_QUERY_LIMIT

    is_valid = (len(errors) == 0)
    return is_valid, errors, sanitized
