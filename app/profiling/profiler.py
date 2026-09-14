"""Data profiler for retail datasets before and after cleaning.
"""
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from app.core.models import ColumnProfile, DatasetProfile
from app.core.security import mask_pii_in_records


def infer_column_type(series: pd.Series) -> str:
    """Infer high-level semantic type for a pandas Series."""
    clean_s = series.dropna()
    if clean_s.empty:
        return "empty"

    col_name = str(series.name).lower()
    if col_name.endswith(("_id", "id", "sku", "code")):
        return "identifier"

    # Check boolean
    unique_vals = set(clean_s.astype(str).str.strip().str.lower().unique())
    if unique_vals.issubset({"true", "false", "0", "1", "yes", "no", "t", "f", "y", "n"}):
        return "boolean"

    # Check numeric
    try:
        pd.to_numeric(clean_s.astype(str).str.replace(r"[\$,%]", "", regex=True).str.strip(), errors="raise")
        if any(term in col_name for term in ["price", "cost", "revenue", "amount", "sales", "discount", "margin", "total", "tax"]):
            return "currency"
        return "numeric"
    except Exception:
        pass

    # Check datetime
    date_terms = ["date", "time", "created_at", "updated_at", "timestamp", "day"]
    if any(term in col_name for term in date_terms):
        try:
            pd.to_datetime(clean_s.head(50), errors="raise", format="mixed")
            return "datetime"
        except Exception:
            pass

    # Check categorical vs free text
    distinct_ratio = len(clean_s.unique()) / len(clean_s)
    if distinct_ratio < 0.2 or len(clean_s.unique()) <= 30:
        return "categorical"

    return "text"


def profile_dataframe(df: pd.DataFrame, dataset_name: str) -> DatasetProfile:
    """Generate a comprehensive profile of a dataset."""
    row_count = int(len(df))
    col_count = int(len(df.columns))
    duplicate_rows = int(df.duplicated().sum()) if row_count > 0 else 0

    col_profiles: Dict[str, ColumnProfile] = {}

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round(float((null_count / row_count) * 100), 2) if row_count > 0 else 0.0
        clean_s = series.dropna()
        distinct_count = int(clean_s.nunique())
        cardinality_ratio = round(float(distinct_count / row_count), 4) if row_count > 0 else 0.0
        is_unique = (distinct_count == row_count and null_count == 0)

        inferred_type = infer_column_type(series)
        stats: Dict[str, Any] = {}
        parse_failures = 0

        # Sample values
        sample_vals = clean_s.head(5).tolist()

        if inferred_type in ("numeric", "currency"):
            try:
                num_s = pd.to_numeric(
                    clean_s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip(),
                    errors="coerce"
                )
                parse_failures = int(num_s.isna().sum())
                valid_num = num_s.dropna()
                if not valid_num.empty:
                    stats = {
                        "min": round(float(valid_num.min()), 2),
                        "max": round(float(valid_num.max()), 2),
                        "mean": round(float(valid_num.mean()), 2),
                        "median": round(float(valid_num.median()), 2),
                        "std": round(float(valid_num.std()), 2) if len(valid_num) > 1 else 0.0,
                        "q25": round(float(valid_num.quantile(0.25)), 2),
                        "q75": round(float(valid_num.quantile(0.75)), 2),
                    }
            except Exception:
                pass
        elif inferred_type == "datetime":
            try:
                dt_s = pd.to_datetime(clean_s, errors="coerce", format="mixed")
                parse_failures = int(dt_s.isna().sum())
                valid_dt = dt_s.dropna()
                if not valid_dt.empty:
                    stats = {
                        "min_date": str(valid_dt.min()),
                        "max_date": str(valid_dt.max()),
                    }
            except Exception:
                pass
        elif inferred_type == "categorical":
            top_freq = clean_s.astype(str).value_counts().head(5).to_dict()
            stats = {
                "top_frequencies": {str(k): int(v) for k, v in top_freq.items()}
            }

        col_profiles[col] = ColumnProfile(
            name=col,
            inferred_type=inferred_type,
            null_count=null_count,
            null_percentage=null_pct,
            distinct_count=distinct_count,
            cardinality_ratio=cardinality_ratio,
            is_unique=is_unique,
            sample_values=sample_vals,
            stats=stats,
            parse_failures=parse_failures,
        )

    return DatasetProfile(
        dataset_name=dataset_name,
        row_count=row_count,
        column_count=col_count,
        duplicate_row_count=duplicate_rows,
        columns=col_profiles,
    )
