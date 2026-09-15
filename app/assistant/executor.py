"""Safe, deterministic execution engine for query AST plans.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from app.core.models import QueryPlan, ExecutionEvidence
from app.retail.relationships import safe_join


def _to_clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(r"[$,€£\s]", "", regex=True)
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def execute_plan(
    plan: QueryPlan,
    loaded_datasets: Dict[str, pd.DataFrame]
) -> ExecutionEvidence:
    """Execute validated query plan and return computational evidence."""
    dataset_name = plan.dataset
    df = loaded_datasets[dataset_name].copy()
    initial_rows = len(df)

    operations: List[str] = [f"LOAD dataset '{dataset_name}' ({initial_rows} rows)"]
    columns_used = set()
    joins_performed = []

    # 1. Execute Joins
    for j in plan.joins:
        columns_used.add(j.left_on)
        columns_used.add(j.right_on)
        right_df = loaded_datasets[j.dataset]
        df, diag = safe_join(
            df_left=df,
            left_name=dataset_name,
            df_right=right_df,
            right_name=j.dataset,
            left_on=j.left_on,
            right_on=j.right_on,
            how=j.how,
        )
        joins_performed.append(diag)
        operations.append(
            f"JOIN '{j.dataset}' ON {j.left_on} = {j.right_on} (output {diag['rows_joined']} rows, fan-out {diag['fanout_ratio']}x)"
        )

    # 2. Execute Filters
    filters_applied = []
    for flt in plan.filters:
        field = flt.field
        op = flt.op
        val = flt.value
        columns_used.add(field)
        filters_applied.append({"field": field, "op": op, "value": val})

        series = df[field]
        if op == "eq":
            if isinstance(val, str):
                mask = series.astype(str).str.lower() == str(val).lower()
            else:
                mask = series == val
        elif op == "neq":
            if isinstance(val, str):
                mask = series.astype(str).str.lower() != str(val).lower()
            else:
                mask = series != val
        elif op == "gt":
            mask = _to_clean_numeric(series) > float(val)
        elif op == "gte":
            mask = _to_clean_numeric(series) >= float(val)
        elif op == "lt":
            mask = _to_clean_numeric(series) < float(val)
        elif op == "lte":
            mask = _to_clean_numeric(series) <= float(val)
        elif op == "in":
            if isinstance(val, list):
                val_set = {str(v).lower() for v in val}
                mask = series.astype(str).str.lower().isin(val_set)
            else:
                mask = series == val
        elif op == "contains":
            mask = series.astype(str).str.contains(str(val), case=False, na=False)
        elif op == "is_null":
            mask = series.isna()
        elif op == "not_null":
            mask = series.notna()
        else:
            mask = pd.Series(True, index=df.index)

        df = df[mask]
        operations.append(f"FILTER WHERE {field} {op} {val} (remaining rows: {len(df)})")

    rows_considered = len(df)

    # 3. Group By & Aggregations
    scalar_answer = None
    result_df = df

    if plan.group_by:
        for gb in plan.group_by:
            columns_used.add(gb)

        agg_dict = {}
        alias_map = {}
        for m in plan.metrics:
            field = m.field
            agg = m.agg
            alias = m.alias or f"{agg}_{field}"
            columns_used.add(field)

            # Map aggregation string to pandas operation
            if agg == "sum":
                result_df[field] = _to_clean_numeric(result_df[field]).fillna(0.0)
                agg_dict[alias] = pd.NamedAgg(column=field, aggfunc="sum")
            elif agg in ("avg", "mean"):
                result_df[field] = _to_clean_numeric(result_df[field])
                agg_dict[alias] = pd.NamedAgg(column=field, aggfunc="mean")
            elif agg == "min":
                result_df[field] = _to_clean_numeric(result_df[field])
                agg_dict[alias] = pd.NamedAgg(column=field, aggfunc="min")
            elif agg == "max":
                result_df[field] = _to_clean_numeric(result_df[field])
                agg_dict[alias] = pd.NamedAgg(column=field, aggfunc="max")
            elif agg == "count":
                agg_dict[alias] = pd.NamedAgg(column=field, aggfunc="count")
            elif agg == "count_distinct":
                agg_dict[alias] = pd.NamedAgg(column=field, aggfunc="nunique")

        if agg_dict:
            result_df = result_df.groupby(plan.group_by, as_index=False).agg(**agg_dict)
            operations.append(
                f"GROUP_BY [{', '.join(plan.group_by)}] WITH AGGREGATIONS [{', '.join(agg_dict.keys())}]"
            )
        else:
            result_df = result_df[plan.group_by].drop_duplicates()
            operations.append(f"GROUP_BY [{', '.join(plan.group_by)}]")

    elif plan.metrics:
        # Global aggregations
        agg_results = {}
        for m in plan.metrics:
            field = m.field
            agg = m.agg
            alias = m.alias or f"{agg}_{field}"
            columns_used.add(field)

            if agg == "count":
                val = len(result_df) if field == "*" else int(result_df[field].count())
            elif agg == "count_distinct":
                val = int(result_df[field].nunique())
            else:
                num_s = _to_clean_numeric(result_df[field]).dropna()
                if agg == "sum":
                    val = round(float(num_s.sum()), 2)
                elif agg in ("avg", "mean"):
                    val = round(float(num_s.mean()), 2) if len(num_s) > 0 else 0.0
                elif agg == "min":
                    val = round(float(num_s.min()), 2) if len(num_s) > 0 else None
                elif agg == "max":
                    val = round(float(num_s.max()), 2) if len(num_s) > 0 else None
                else:
                    val = 0.0

            agg_results[alias] = val

        result_df = pd.DataFrame([agg_results])
        operations.append(f"AGGREGATE [{', '.join(agg_results.keys())}]")
        if len(agg_results) == 1:
            scalar_answer = list(agg_results.values())[0]

    # 4. Sorting
    if plan.sort:
        sort_fields = []
        sort_dirs = []
        for s in plan.sort:
            if s.field in result_df.columns:
                sort_fields.append(s.field)
                sort_dirs.append(True if s.dir.lower() == "asc" else False)
        if sort_fields:
            result_df = result_df.sort_values(by=sort_fields, ascending=sort_dirs)
            operations.append(f"SORT BY {sort_fields} ascending={sort_dirs}")

    # 5. Limit
    bounded_df = result_df.head(plan.limit)
    operations.append(f"LIMIT {plan.limit} (returned {len(bounded_df)} rows)")

    # Format preview records
    preview_records = bounded_df.to_dict(orient="records")
    # Clean NaN / NaT for valid JSON serialization
    for rec in preview_records:
        for k, v in rec.items():
            if pd.isna(v):
                rec[k] = None
            elif isinstance(v, (np.floating, float)):
                rec[k] = round(float(v), 2)
            elif isinstance(v, (np.integer, int)):
                rec[k] = int(v)

    return ExecutionEvidence(
        dataset_version=dataset_name,
        columns_used=sorted(list(columns_used)),
        filters_applied=filters_applied,
        joins_performed=joins_performed,
        operations=operations,
        row_count_considered=rows_considered,
        rows_returned=len(preview_records),
        preview=preview_records,
        scalar_answer=scalar_answer,
    )
