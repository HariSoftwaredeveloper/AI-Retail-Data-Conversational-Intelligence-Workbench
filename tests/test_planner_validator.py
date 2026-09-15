"""Tests for AST plan validation and defense against malicious/expensive plans.
"""
import pandas as pd
from app.core.models import QueryPlan, QueryFilter, QueryMetric
from app.assistant.validator import validate_plan


def test_validator_accepts_valid_plan():
    df = pd.DataFrame({"region": ["West", "East"], "sales": [100.0, 200.0]})
    loaded = {"orders_clean": df}

    plan = QueryPlan(
        intent="aggregate",
        dataset="orders_clean",
        filters=[QueryFilter(field="region", op="eq", value="West")],
        metrics=[QueryMetric(agg="sum", field="sales", alias="total_sales")],
        limit=10,
    )

    is_valid, errors, sanitized = validate_plan(plan, loaded)
    assert is_valid is True
    assert len(errors) == 0


def test_validator_rejects_unknown_field():
    df = pd.DataFrame({"region": ["West", "East"], "sales": [100.0, 200.0]})
    loaded = {"orders_clean": df}

    plan = QueryPlan(
        intent="aggregate",
        dataset="orders_clean",
        filters=[QueryFilter(field="non_existent_column", op="eq", value="X")],
        metrics=[QueryMetric(agg="sum", field="sales")],
    )

    is_valid, errors, _ = validate_plan(plan, loaded)
    assert is_valid is False
    assert any("does not exist in schema" in err for err in errors)


def test_validator_clamps_excessive_limit():
    df = pd.DataFrame({"region": ["West", "East"], "sales": [100.0, 200.0]})
    loaded = {"orders_clean": df}

    plan = QueryPlan(
        intent="filter",
        dataset="orders_clean",
        limit=1000000,  # Excessive unbounded limit
    )

    is_valid, errors, sanitized = validate_plan(plan, loaded)
    assert is_valid is True
    assert sanitized.limit == 100  # Clamped to MAX_QUERY_LIMIT
