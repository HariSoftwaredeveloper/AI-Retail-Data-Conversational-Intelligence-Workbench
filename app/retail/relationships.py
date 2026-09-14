"""Cross-dataset relationship validation and safe joins with fan-out protection.
"""
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# Known retail relationship patterns: (parent, child, parent_key, child_key, expected_cardinality)
RETAIL_RELATIONSHIPS = [
    ("customers", "orders", "customer_id", "customer_id", "1:N"),
    ("orders", "order_items", "order_id", "order_id", "1:N"),
    ("products", "order_items", "product_id", "product_id", "1:N"),
    ("products", "inventory", "product_id", "product_id", "1:1"),
    ("stores", "orders", "store_id", "store_id", "1:N"),
    ("stores", "inventory", "store_id", "store_id", "1:N"),
]


def check_referential_integrity(
    df_parent: pd.DataFrame,
    parent_key: str,
    df_child: pd.DataFrame,
    child_key: str
) -> Dict[str, Any]:
    """Check for orphaned keys and cardinality."""
    if parent_key not in df_parent.columns:
        return {"valid": False, "error": f"Parent key '{parent_key}' not found in parent table"}
    if child_key not in df_child.columns:
        return {"valid": False, "error": f"Child key '{child_key}' not found in child table"}

    parent_keys = set(df_parent[parent_key].dropna().unique())
    child_keys = set(df_child[child_key].dropna().unique())

    orphaned_keys = child_keys - parent_keys
    orphaned_count = len(orphaned_keys)
    orphaned_rows = int(df_child[df_child[child_key].isin(orphaned_keys)].shape[0])

    return {
        "valid": True,
        "parent_key": parent_key,
        "child_key": child_key,
        "parent_distinct_keys": len(parent_keys),
        "child_distinct_keys": len(child_keys),
        "orphaned_keys_count": orphaned_count,
        "orphaned_child_rows": orphaned_rows,
        "sample_orphaned_keys": list(orphaned_keys)[:5],
    }


def safe_join(
    df_left: pd.DataFrame,
    left_name: str,
    df_right: pd.DataFrame,
    right_name: str,
    left_on: str,
    right_on: str,
    how: str = "inner",
    max_fanout_ratio: float = 3.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute join between two retail datasets with explicit join keys and fan-out protection."""
    if left_on not in df_left.columns:
        raise ValueError(f"Join key '{left_on}' missing from '{left_name}'")
    if right_on not in df_right.columns:
        raise ValueError(f"Join key '{right_on}' missing from '{right_name}'")

    rows_left = len(df_left)
    rows_right = len(df_right)

    # Perform safe join
    # Suffix collision handling
    df_joined = pd.merge(
        df_left,
        df_right,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffixes=("", f"_{right_name}")
    )
    rows_joined = len(df_joined)

    # Fan-out calculation
    base_rows = rows_left if how in ("left", "inner") else max(rows_left, rows_right)
    fanout_ratio = round(rows_joined / base_rows, 2) if base_rows > 0 else 1.0

    fanout_warning = False
    if fanout_ratio > max_fanout_ratio:
        fanout_warning = True

    diagnostics = {
        "left_table": left_name,
        "right_table": right_name,
        "left_on": left_on,
        "right_on": right_on,
        "how": how,
        "rows_left": rows_left,
        "rows_right": rows_right,
        "rows_joined": rows_joined,
        "fanout_ratio": fanout_ratio,
        "fanout_warning": fanout_warning,
        "columns_in_result": list(df_joined.columns),
    }

    return df_joined, diagnostics
