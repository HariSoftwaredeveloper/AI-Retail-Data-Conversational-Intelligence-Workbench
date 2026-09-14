"""Auditable retail metrics calculation engine with formula citations.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


def compute_sales_metrics(df_orders: pd.DataFrame) -> Dict[str, Any]:
    """Compute core retail sales metrics with auditable formulas."""
    if df_orders.empty:
        return {"error": "Orders dataset is empty"}

    total_orders = int(len(df_orders))

    # Identify revenue / amount column
    rev_col = next((c for c in df_orders.columns if c.lower() in ("total_amount", "revenue", "sales", "total_price", "amount", "total")), None)
    if not rev_col:
        return {"error": "Revenue/amount column not found in orders"}

    revenue_series = pd.to_numeric(df_orders[rev_col], errors="coerce").fillna(0.0)
    gross_revenue = round(float(revenue_series.sum()), 2)
    aov = round(float(gross_revenue / total_orders), 2) if total_orders > 0 else 0.0

    # Status / Return Rate
    status_col = next((c for c in df_orders.columns if c.lower() in ("status", "order_status")), None)
    returned_orders = 0
    returned_revenue = 0.0
    return_rate_pct = 0.0

    if status_col:
        return_mask = df_orders[status_col].astype(str).str.lower().isin(["returned", "refunded", "cancelled"])
        returned_orders = int(return_mask.sum())
        returned_revenue = round(float(revenue_series[return_mask].sum()), 2)
        return_rate_pct = round(float((returned_orders / total_orders) * 100), 2) if total_orders > 0 else 0.0

    net_revenue = round(gross_revenue - returned_revenue, 2)

    return {
        "total_orders": {
            "value": total_orders,
            "formula": "COUNT(order_id)",
            "citation": f"Counted {total_orders} order rows",
        },
        "gross_revenue": {
            "value": gross_revenue,
            "formula": f"SUM({rev_col})",
            "citation": f"Summed {rev_col} across all orders",
        },
        "net_revenue": {
            "value": net_revenue,
            "formula": "gross_revenue - returned_revenue",
            "citation": f"${gross_revenue} - ${returned_revenue}",
        },
        "aov": {
            "value": aov,
            "formula": f"gross_revenue / total_orders",
            "citation": f"${gross_revenue} / {total_orders} orders = ${aov}",
        },
        "return_rate_pct": {
            "value": return_rate_pct,
            "formula": "(returned_orders / total_orders) * 100",
            "citation": f"({returned_orders} returned / {total_orders} total) * 100 = {return_rate_pct}%",
        },
        "returned_orders_count": returned_orders,
        "returned_revenue": returned_revenue,
    }


def compute_inventory_metrics(
    df_inventory: pd.DataFrame,
    df_sales: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Compute inventory KPIs: stockout frequency, total inventory, and sell-through."""
    if df_inventory.empty:
        return {"error": "Inventory dataset is empty"}

    stock_col = next((c for c in df_inventory.columns if c.lower() in ("stock", "stock_level", "quantity", "inventory_level")), None)
    if not stock_col:
        return {"error": "Stock column not found in inventory dataset"}

    stock_s = pd.to_numeric(df_inventory[stock_col], errors="coerce").fillna(0)
    total_stock = int(stock_s.sum())
    total_skus = int(len(df_inventory))

    # Stock-out frequency
    stockout_mask = stock_s <= 0
    stockout_skus = int(stockout_mask.sum())
    stockout_pct = round(float((stockout_skus / total_skus) * 100), 2) if total_skus > 0 else 0.0

    result = {
        "total_stock": {
            "value": total_stock,
            "formula": f"SUM({stock_col})",
            "citation": f"Summed stock across {total_skus} SKU records",
        },
        "total_skus": total_skus,
        "stockout_skus": stockout_skus,
        "stockout_frequency_pct": {
            "value": stockout_pct,
            "formula": "(count(stock <= 0) / total_skus) * 100",
            "citation": f"({stockout_skus} out-of-stock / {total_skus} SKUs) * 100 = {stockout_pct}%",
        },
    }

    # If sales items are available, calculate sell-through rate
    if df_sales is not None and not df_sales.empty:
        qty_col = next((c for c in df_sales.columns if c.lower() in ("quantity", "units_sold", "qty")), None)
        if qty_col:
            units_sold = int(pd.to_numeric(df_sales[qty_col], errors="coerce").fillna(0).sum())
            denominator = units_sold + total_stock
            sell_through_pct = round(float((units_sold / denominator) * 100), 2) if denominator > 0 else 0.0
            result["units_sold"] = units_sold
            result["sell_through_rate_pct"] = {
                "value": sell_through_pct,
                "formula": "(units_sold / (units_sold + ending_stock)) * 100",
                "citation": f"({units_sold} sold / ({units_sold} + {total_stock} stock)) * 100 = {sell_through_pct}%",
            }

    return result


def compute_category_analytics(df_merged: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute breakdown by category for sales, units, and return rate."""
    cat_col = next((c for c in df_merged.columns if "category" in c.lower()), None)
    rev_col = next((c for c in df_merged.columns if c.lower() in ("total_amount", "revenue", "sales", "price")), None)

    if not cat_col or not rev_col:
        return []

    # Safe numeric conversion
    df = df_merged.copy()
    df["_rev_clean"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0.0)

    grouped = df.groupby(cat_col)
    analytics = []

    for cat, grp in grouped:
        total_rev = round(float(grp["_rev_clean"].sum()), 2)
        count_orders = int(len(grp))
        aov = round(float(total_rev / count_orders), 2) if count_orders > 0 else 0.0

        analytics.append({
            "category": str(cat),
            "revenue": total_rev,
            "order_count": count_orders,
            "aov": aov,
        })

    # Sort descending by revenue
    analytics.sort(key=lambda x: x["revenue"], reverse=True)
    return analytics
