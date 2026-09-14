"""Explainable retail scoring and recommendation engine based on deterministic signals.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


def score_product_health(
    product_row: Dict[str, Any],
    sales_units: Optional[int] = None,
    return_rate: Optional[float] = None,
    stock_qty: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute an explainable Product Health Score (0-100) from explicit retail signals.
    Never invents signals; explicitly enumerates missing evidence.
    """
    score = 50  # baseline neutral
    contributing_factors: List[Dict[str, Any]] = []
    missing_evidence: List[str] = []

    # 1. Margin Signal
    price = product_row.get("price") or product_row.get("unit_price")
    cost = product_row.get("cost") or product_row.get("unit_cost")

    if price is not None and cost is not None:
        try:
            p_val = float(price)
            c_val = float(cost)
            if p_val > 0:
                margin_pct = (p_val - c_val) / p_val
                if margin_pct >= 0.50:
                    pts = 20
                    desc = f"Strong gross margin ({round(margin_pct*100, 1)}%)"
                elif margin_pct >= 0.30:
                    pts = 10
                    desc = f"Healthy gross margin ({round(margin_pct*100, 1)}%)"
                elif margin_pct >= 0.15:
                    pts = 0
                    desc = f"Moderate gross margin ({round(margin_pct*100, 1)}%)"
                else:
                    pts = -15
                    desc = f"Low gross margin ({round(margin_pct*100, 1)}%)"
                score += pts
                contributing_factors.append({"factor": "Gross Margin", "points": pts, "reason": desc})
        except (ValueError, TypeError):
            missing_evidence.append("Cost or price could not be parsed as numeric")
    else:
        missing_evidence.append("Unit cost missing; margin contribution omitted")

    # 2. Sales Velocity Signal
    if sales_units is not None:
        if sales_units > 100:
            pts = 20
            desc = f"High sales volume ({sales_units} units)"
        elif sales_units >= 25:
            pts = 10
            desc = f"Moderate sales volume ({sales_units} units)"
        elif sales_units > 0:
            pts = 0
            desc = f"Slow sales volume ({sales_units} units)"
        else:
            pts = -15
            desc = "Zero sales recorded"
        score += pts
        contributing_factors.append({"factor": "Sales Velocity", "points": pts, "reason": desc})
    else:
        missing_evidence.append("Sales transaction history not provided")

    # 3. Return Rate Signal
    if return_rate is not None:
        if return_rate <= 0.03:
            pts = 10
            desc = f"Very low return rate ({round(return_rate*100, 1)}%)"
        elif return_rate <= 0.08:
            pts = 0
            desc = f"Acceptable return rate ({round(return_rate*100, 1)}%)"
        elif return_rate <= 0.15:
            pts = -15
            desc = f"Elevated return rate ({round(return_rate*100, 1)}%)"
        else:
            pts = -25
            desc = f"Critical return rate ({round(return_rate*100, 1)}%)"
        score += pts
        contributing_factors.append({"factor": "Return Rate", "points": pts, "reason": desc})
    else:
        missing_evidence.append("Return data not linked for this product")

    # 4. Stock Availability Signal
    if stock_qty is not None:
        if stock_qty == 0:
            pts = -20
            desc = "Out of stock (lost sales risk)"
        elif stock_qty < 10:
            pts = -5
            desc = f"Critically low stock ({stock_qty} units left)"
        elif stock_qty > 500:
            pts = -5
            desc = f"Potential overstocking ({stock_qty} units on hand)"
        else:
            pts = 10
            desc = f"Optimal stock coverage ({stock_qty} units on hand)"
        score += pts
        contributing_factors.append({"factor": "Stock Level", "points": pts, "reason": desc})
    else:
        missing_evidence.append("Inventory level not provided")

    # Clamp score to [0, 100]
    final_score = max(0, min(100, score))

    # Determine recommendation
    if final_score >= 80:
        recommendation = "Promote / Feature: High performer with solid fundamentals"
    elif final_score >= 60:
        recommendation = "Maintain: Stable performer"
    elif final_score >= 40:
        recommendation = "Review: Investigate low margins or return rate issues"
    else:
        recommendation = "Intervene: High risk of loss, stockout, or customer dissatisfaction"

    return {
        "product_id": product_row.get("product_id", "Unknown"),
        "product_name": product_row.get("name") or product_row.get("title") or "Unnamed Product",
        "health_score": final_score,
        "recommendation": recommendation,
        "contributing_factors": contributing_factors,
        "missing_evidence": missing_evidence,
    }


def score_customer_rfm(df_orders: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute auditable RFM (Recency, Frequency, Monetary) segmentation for customers."""
    cust_col = next((c for c in df_orders.columns if "customer" in c.lower()), None)
    rev_col = next((c for c in df_orders.columns if c.lower() in ("total_amount", "revenue", "sales", "amount")), None)
    date_col = next((c for c in df_orders.columns if "date" in c.lower()), None)

    if not cust_col or not rev_col:
        return []

    df = df_orders.copy()
    df["_rev"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0.0)

    # Group by customer
    grouped = df.groupby(cust_col)
    records = []

    for cust_id, grp in grouped:
        freq = len(grp)
        monetary = round(float(grp["_rev"].sum()), 2)

        # Segment classification
        if freq >= 5 and monetary >= 500:
            segment = "Champion"
        elif freq >= 3 or monetary >= 250:
            segment = "Loyal Customer"
        elif freq == 1 and monetary < 100:
            segment = "Occasional / Low Value"
        else:
            segment = "Regular Customer"

        records.append({
            "customer_id": str(cust_id),
            "frequency_orders": freq,
            "monetary_spend": monetary,
            "segment": segment,
            "audit_evidence": f"Calculated from {freq} orders totaling ${monetary}",
        })

    records.sort(key=lambda x: x["monetary_spend"], reverse=True)
    return records
