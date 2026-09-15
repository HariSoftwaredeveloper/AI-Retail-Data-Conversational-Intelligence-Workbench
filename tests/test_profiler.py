"""Tests for dataset profiling and issue detection.
"""
import pandas as pd
from app.profiling.profiler import profile_dataframe, infer_column_type
from app.profiling.issue_detector import detect_issues


def test_infer_column_type():
    s_id = pd.Series(["ORD-1", "ORD-2"], name="order_id")
    assert infer_column_type(s_id) == "identifier"

    s_curr = pd.Series(["$19.99", "$25.50"], name="price")
    assert infer_column_type(s_curr) == "currency"

    s_cat = pd.Series(["West", "West", "East", "West"], name="region")
    assert infer_column_type(s_cat) == "categorical"


def test_profile_dataframe_stats():
    df = pd.DataFrame({
        "order_id": ["O1", "O2", "O3"],
        "amount": [10.0, 20.0, 30.0],
        "category": ["A", "B", "A"]
    })
    profile = profile_dataframe(df, "test_orders")
    assert profile.row_count == 3
    assert profile.column_count == 3
    assert profile.duplicate_row_count == 0
    assert "amount" in profile.columns
    assert profile.columns["amount"].stats["mean"] == 20.0


def test_issue_detector_identifies_all_anomalies():
    df = pd.DataFrame({
        "order_id": ["O1", "O2", "O1"],  # Duplicate key and duplicate row
        "price": [" $10.00 ", "$20.50", " $10.00 "],  # Currency & whitespace
        "order_date": ["2024-01-01", "02/01/2024", "2024-01-01"],  # Mixed dates
        "category": ["Electronics", "Elecronics", "Electronics"],  # Category typo
    })
    issues = detect_issues(df, "orders")
    issue_types = {iss.issue_type for iss in issues}

    assert "duplicate_records" in issue_types
    assert "whitespace_case" in issue_types
    assert "currency_formats" in issue_types
    assert "inconsistent_dates" in issue_types
    assert "suspicious_categories" in issue_types
