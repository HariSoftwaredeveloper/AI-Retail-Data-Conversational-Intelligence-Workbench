"""Deterministic data transformation functions.
"""
from typing import List, Optional
import re
import difflib
import pandas as pd
import numpy as np


def strip_whitespace_and_cleanup(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Trim leading/trailing whitespace, collapse internal spaces, and replace null strings."""
    df_out = df.copy()
    if columns:
        target_cols = columns
    else:
        target_cols = [
            c for c in df_out.columns
            if df_out[c].dtype == object or "str" in str(df_out[c].dtype).lower()
        ]

    for col in target_cols:
        if col in df_out.columns:
            s = df_out[col]
            if s.dtype == object or "str" in str(s.dtype).lower():
                s = s.astype(str).str.replace(r"[\u00a0\u200b\ufeff]", " ", regex=True)
                s = s.str.strip()
                s = s.replace({"nan": None, "None": None, "null": None, "NULL": None, "": None, "N/A": None})
                df_out[col] = s
    return df_out


def parse_currency_and_numbers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Safely convert formatted currency or messy numbers to float."""
    df_out = df.copy()
    if column not in df_out.columns:
        return df_out

    s = df_out[column].astype(str)

    def _clean_num_val(v):
        if pd.isna(v) or str(v).strip().lower() in ("nan", "none", "null", "", "n/a"):
            return np.nan
        val = str(v).strip()
        # Handle parenthesis negatives like (45.20)
        is_neg = False
        if val.startswith("(") and val.endswith(")"):
            is_neg = True
            val = val[1:-1]
        elif val.endswith("-"):
            is_neg = True
            val = val[:-1]
        # Remove currency symbols, commas, percent
        val = re.sub(r"[\$,€£¥% ]", "", val)
        try:
            num = float(val)
            return -num if is_neg else num
        except ValueError:
            return np.nan

    df_out[column] = s.apply(_clean_num_val)
    return df_out


def parse_mixed_dates(df: pd.DataFrame, column: str, output_format: str = "%Y-%m-%d") -> pd.DataFrame:
    """Robust multi-format date parser converting dates to ISO YYYY-MM-DD."""
    df_out = df.copy()
    if column not in df_out.columns:
        return df_out

    s = df_out[column]

    def _parse_dt(val):
        if pd.isna(val) or str(val).strip().lower() in ("nan", "none", "null", "", "n/a"):
            return None
        v = str(val).strip()
        # If timestamp epoch
        if re.match(r"^\d{10,13}$", v):
            try:
                unit = "ms" if len(v) > 10 else "s"
                dt = pd.to_datetime(int(v), unit=unit)
                return dt.strftime(output_format)
            except Exception:
                pass

        try:
            # Let pandas infer mixed date formats with dayfirst heuristics
            dt = pd.to_datetime(v, format="mixed", errors="coerce")
            if pd.notna(dt):
                return dt.strftime(output_format)
        except Exception:
            pass
        return v  # Retain original if unparseable to avoid silent loss

    df_out[column] = s.apply(_parse_dt)
    return df_out


def harmonize_category_variants(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Standardize casing and cluster minor spelling typos to the canonical form."""
    df_out = df.copy()
    if column not in df_out.columns:
        return df_out

    s = df_out[column]
    non_nulls = s.dropna().astype(str).str.strip()
    if non_nulls.empty:
        return df_out

    # Step 1: Standardize title casing
    def _title_case(val):
        if pd.isna(val) or str(val).strip().lower() in ("nan", "none", "null", ""):
            return None
        words = str(val).strip().split()
        return " ".join([w.capitalize() if not w.isupper() or len(w) > 4 else w for w in words])

    standardized = s.apply(_title_case)
    df_out[column] = standardized

    # Step 2: Cluster close spelling variations (Levenshtein / SequenceMatcher)
    counts = df_out[column].value_counts().to_dict()
    canonical_map = {}
    sorted_cats = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)

    for cat in sorted_cats:
        cat_lower = cat.lower()
        if cat in canonical_map:
            continue
        canonical_map[cat] = cat
        for other in sorted_cats:
            if other in canonical_map or other == cat:
                continue
            other_lower = other.lower()
            # High similarity and small length difference implies typo of popular category
            sim = difflib.SequenceMatcher(None, cat_lower, other_lower).ratio()
            if sim >= 0.85 and abs(len(cat_lower) - len(other_lower)) <= 2:
                # Map less frequent variant to the more frequent canonical form
                canonical_map[other] = cat

    df_out[column] = df_out[column].map(lambda x: canonical_map.get(x, x) if pd.notna(x) else None)
    return df_out


def deduplicate_exact_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows while preserving index order."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def deduplicate_by_primary_key(df: pd.DataFrame, pk_column: str) -> pd.DataFrame:
    """Resolve duplicate records on a primary key by retaining the record with fewest nulls."""
    if pk_column not in df.columns:
        return df

    # Count non-nulls per row to pick best record
    null_counts = df.isna().sum(axis=1)
    df_temp = df.assign(_null_score=null_counts)
    df_sorted = df_temp.sort_values(by="_null_score", ascending=True)
    df_dedup = df_sorted.drop_duplicates(subset=[pk_column], keep="first")
    return df_dedup.drop(columns=["_null_score"]).sort_index().reset_index(drop=True)
