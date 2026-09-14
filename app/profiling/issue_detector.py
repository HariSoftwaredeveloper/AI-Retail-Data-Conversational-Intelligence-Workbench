"""Issue detector for identifying data quality anomalies in retail CSVs.
"""
from typing import Any, Dict, List
import difflib
import re
import pandas as pd
from app.core.models import Issue


def detect_issues(df: pd.DataFrame, dataset_name: str) -> List[Issue]:
    """Inspect dataset and return list of structured issues with stable IDs."""
    issues: List[Issue] = []

    # 1. Exact Duplicate Records
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        sample_dup = df[dup_mask].head(3).to_dict(orient="records")
        issues.append(
            Issue(
                id=f"{dataset_name}_issue_duplicate_rows",
                field=None,
                issue_type="duplicate_records",
                severity="medium",
                description=f"Found {dup_count} exact duplicate rows.",
                sample_values=sample_dup,
                affected_count=dup_count,
            )
        )

    # 2. Check each column
    clean_ds_name = dataset_name.lower().replace("_raw", "").replace("_clean", "")

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        col_lower = col.lower()
        is_id_col = col_lower.endswith(("_id", "id", "sku", "code"))

        if non_null.empty:
            issues.append(
                Issue(
                    id=f"{dataset_name}_{col}_all_null",
                    field=col,
                    issue_type="missing_values",
                    severity="high",
                    description=f"Column '{col}' is entirely empty/null.",
                    sample_values=[],
                    affected_count=len(df),
                )
            )
            continue

        str_series = non_null.astype(str)

        # 2a. Whitespace and invisible characters
        has_leading_trailing = str_series.str.match(r"^\s+|\s+$").any()
        if has_leading_trailing:
            dirty_samples = str_series[str_series.str.match(r"^\s+|\s+$")].head(3).tolist()
            count = int(str_series.str.match(r"^\s+|\s+$").sum())
            issues.append(
                Issue(
                    id=f"{dataset_name}_{col}_whitespace",
                    field=col,
                    issue_type="whitespace_case",
                    severity="low",
                    description=f"Column '{col}' has {count} values with leading or trailing whitespace.",
                    sample_values=dirty_samples,
                    affected_count=count,
                )
            )

        # 2b. Currency & Formatted Numbers in String Columns
        has_currency = str_series.str.contains(r"[\$,€£¥]", regex=True).any()
        has_commas_in_num = str_series.str.contains(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$", regex=True).any()
        has_parens_neg = str_series.str.contains(r"^\(\d+(?:\.\d+)?\)$", regex=True).any()
        if has_currency or has_commas_in_num or has_parens_neg:
            affected = int(
                str_series.str.contains(r"[\$,€£¥]|\(\d+(?:\.\d+)?\)|\d{1,3}(?:,\d{3})+", regex=True).sum()
            )
            samples = str_series[
                str_series.str.contains(r"[\$,€£¥]|\(\d+(?:\.\d+)?\)|\d{1,3}(?:,\d{3})+", regex=True)
            ].head(3).tolist()
            issues.append(
                Issue(
                    id=f"{dataset_name}_{col}_currency_formats",
                    field=col,
                    issue_type="currency_formats",
                    severity="medium",
                    description=f"Column '{col}' contains formatted currency or comma-separated numbers as strings.",
                    sample_values=samples,
                    affected_count=affected,
                )
            )

        # 2c. Inconsistent Dates
        if any(term in col_lower for term in ["date", "time", "created_at", "timestamp"]):
            has_dash = str_series.str.contains(r"^\d{4}-\d{1,2}-\d{1,2}", regex=True).any()
            has_slash = str_series.str.contains(r"^\d{1,2}/\d{1,2}/\d{2,4}", regex=True).any()
            if has_dash and has_slash:
                count = int((str_series.str.contains(r"^\d{1,2}/\d{1,2}/\d{2,4}", regex=True)).sum())
                samples = str_series.head(5).tolist()
                issues.append(
                    Issue(
                        id=f"{dataset_name}_{col}_inconsistent_dates",
                        field=col,
                        issue_type="inconsistent_dates",
                        severity="high",
                        description=f"Column '{col}' has mixed date representations (YYYY-MM-DD vs DD/MM/YYYY).",
                        sample_values=samples,
                        affected_count=count,
                    )
                )

        # 2d. Inconsistent casing & typos in true categorical columns (exclude ID columns)
        if not is_id_col and str_series.nunique() <= 50 and len(str_series) >= 2:
            uniques = [u.strip() for u in str_series.unique() if u.strip()]
            lower_map: Dict[str, List[str]] = {}
            for u in uniques:
                lower_map.setdefault(u.lower(), []).append(u)
            casing_variants = [variants for variants in lower_map.values() if len(variants) > 1]
            if casing_variants:
                issues.append(
                    Issue(
                        id=f"{dataset_name}_{col}_inconsistent_case",
                        field=col,
                        issue_type="whitespace_case",
                        severity="low",
                        description=f"Column '{col}' has conflicting case variants: {casing_variants[:3]}",
                        sample_values=[v for sub in casing_variants for v in sub][:5],
                        affected_count=len(casing_variants),
                    )
                )

            # Suspicious category spelling / typos
            cleaned_uniques = list(lower_map.keys())
            suspicious_pairs = []
            for i, u1 in enumerate(cleaned_uniques):
                for u2 in cleaned_uniques[i + 1:]:
                    ratio = difflib.SequenceMatcher(None, u1, u2).ratio()
                    if 0.82 <= ratio < 1.0 and abs(len(u1) - len(u2)) <= 2:
                        suspicious_pairs.append((u1, u2, round(ratio, 2)))
            if suspicious_pairs:
                issues.append(
                    Issue(
                        id=f"{dataset_name}_{col}_suspicious_categories",
                        field=col,
                        issue_type="suspicious_categories",
                        severity="medium",
                        description=f"Column '{col}' has suspicious near-duplicate categories: {suspicious_pairs[:3]}",
                        sample_values=[f"{p[0]} ~ {p[1]} (sim: {p[2]})" for p in suspicious_pairs[:3]],
                        affected_count=len(suspicious_pairs),
                    )
                )

        # 2e. Primary key duplicates: only check if column matches table primary entity name
        # e.g. order_id in orders, product_id in products, customer_id in customers
        is_primary_key = (
            (clean_ds_name.startswith("order") and col_lower == "order_id") or
            (clean_ds_name.startswith("product") and col_lower in ("product_id", "sku")) or
            (clean_ds_name.startswith("customer") and col_lower == "customer_id")
        )
        if is_primary_key:
            id_dups = int(series.duplicated().sum())
            if id_dups > 0:
                dup_ids = series[series.duplicated()].head(3).tolist()
                issues.append(
                    Issue(
                        id=f"{dataset_name}_{col}_duplicate_keys",
                        field=col,
                        issue_type="duplicate_records",
                        severity="high",
                        description=f"Primary identifier column '{col}' contains {id_dups} duplicate keys.",
                        sample_values=dup_ids,
                        affected_count=id_dups,
                    )
                )

    return issues
