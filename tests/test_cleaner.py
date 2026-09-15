"""Tests for data cleaning pipeline and idempotency invariants.
"""
from pathlib import Path
import pandas as pd
from app.cleaning.transformers import (
    strip_whitespace_and_cleanup,
    parse_currency_and_numbers,
    parse_mixed_dates,
    harmonize_category_variants,
    deduplicate_exact_rows,
)
from app.cleaning.pipeline import run_pipeline


def test_strip_whitespace():
    df = pd.DataFrame({"name": ["  T-Shirt  ", "Shoes \t", "Hat"]})
    clean_df = strip_whitespace_and_cleanup(df)
    assert clean_df["name"].tolist() == ["T-Shirt", "Shoes", "Hat"]


def test_parse_currency_and_numbers():
    df = pd.DataFrame({"price": ["$1,200.50", "(45.00)", "€25.99", "100", "invalid"]})
    clean_df = parse_currency_and_numbers(df, "price")
    assert clean_df["price"].iloc[0] == 1200.50
    assert clean_df["price"].iloc[1] == -45.00
    assert clean_df["price"].iloc[2] == 25.99
    assert clean_df["price"].iloc[3] == 100.0
    assert pd.isna(clean_df["price"].iloc[4])


def test_parse_mixed_dates():
    df = pd.DataFrame({"date": ["2024-01-15", "16/01/2024", "2024-02-01"]})
    clean_df = parse_mixed_dates(df, "date")
    assert clean_df["date"].iloc[0] == "2024-01-15"
    assert clean_df["date"].iloc[1] == "2024-01-16"
    assert clean_df["date"].iloc[2] == "2024-02-01"


def test_harmonize_categories():
    df = pd.DataFrame({"category": ["Electronics", "Elecronics", "electronics", "Electronics", "Apparel"]})
    clean_df = harmonize_category_variants(df, "category")
    # 'Elecronics' should cluster into dominant 'Electronics'
    assert clean_df["category"].iloc[1] == "Electronics"
    assert clean_df["category"].iloc[2] == "Electronics"


def test_pipeline_idempotency(tmp_path: Path):
    """Invariant: Running the cleaner twice on clean output must produce zero changes."""
    df_raw = pd.DataFrame({
        "order_id": ["O1", "O2", "O1"],
        "price": ["$10.00", "$20.50", "$10.00"],
        "order_date": ["2024-01-01", "02/01/2024", "2024-01-01"],
        "category": ["Electronics", "Elecronics", "Electronics"],
    })

    res = run_pipeline(df_raw, "orders_test", "run_test_01", artifacts_dir=tmp_path)
    assert res["validation"]["idempotent"] is True
    assert res["validation"]["row_delta"] == -1  # Exactly 1 duplicate row removed
    assert res["validation"]["schema_mutated"] is False
