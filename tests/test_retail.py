"""Tests for retail domain intelligence: normalization, safe joins, metrics, and scoring.
"""
import pandas as pd
from app.retail.normalizer import normalize_category_or_product, normalize_brand, normalize_sku
from app.retail.relationships import safe_join, check_referential_integrity
from app.retail.metrics import compute_sales_metrics, compute_inventory_metrics
from app.retail.scoring import score_product_health, score_customer_rfm


def test_product_category_normalization_with_uncertainty():
    # Exact canonical synonym
    cat, conf, note = normalize_category_or_product("tee")
    assert cat == "Apparel - Tops"
    assert conf == 1.0
    assert note is None

    # Substring match
    cat2, conf2, note2 = normalize_category_or_product("casual blue t-shirt")
    assert cat2 == "Apparel - Tops"
    assert conf2 >= 0.9

    # Uncertain value
    cat3, conf3, note3 = normalize_category_or_product("vintage artisanal widgets")
    assert conf3 < 0.8
    assert note3 is not None


def test_safe_join_and_fanout_detection():
    df_orders = pd.DataFrame({
        "order_id": ["O1", "O2"],
        "product_id": ["P1", "P2"]
    })
    # Intentionally duplicate product_id to simulate Cartesian fan-out
    df_products_fanout = pd.DataFrame({
        "product_id": ["P1", "P1", "P1", "P1", "P2"],
        "category": ["A", "B", "C", "D", "E"]
    })

    joined, diag = safe_join(
        df_orders, "orders", df_products_fanout, "products",
        left_on="product_id", right_on="product_id", how="inner", max_fanout_ratio=2.0
    )
    assert diag["rows_joined"] == 5
    assert diag["fanout_warning"] is True
    assert diag["fanout_ratio"] == 2.5


def test_retail_sales_metrics():
    df_orders = pd.DataFrame({
        "order_id": ["O1", "O2", "O3"],
        "total_amount": [100.0, 50.0, 50.0],
        "status": ["Completed", "Returned", "Completed"]
    })
    metrics = compute_sales_metrics(df_orders)
    assert metrics["total_orders"]["value"] == 3
    assert metrics["gross_revenue"]["value"] == 200.0
    assert metrics["aov"]["value"] == 66.67
    assert metrics["return_rate_pct"]["value"] == 33.33
    assert metrics["net_revenue"]["value"] == 150.0


def test_product_health_scoring_explainability():
    product = {"product_id": "P1", "name": "Premium Shoe", "price": 100.0, "cost": 40.0}
    # Strong margin (60%), solid velocity, low returns, healthy stock
    score_data = score_product_health(
        product_row=product,
        sales_units=150,
        return_rate=0.02,
        stock_qty=50
    )
    assert score_data["health_score"] > 80
    assert "Promote" in score_data["recommendation"]
    assert len(score_data["contributing_factors"]) == 4
    # Missing evidence must be empty when all signals are supplied
    assert len(score_data["missing_evidence"]) == 0
