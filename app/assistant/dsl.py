"""DSL specification for constrained query plans.
"""
from typing import Any, Dict, List, Optional
from app.core.models import QueryFilter, QueryMetric, QuerySort, QueryJoin, QueryPlan


ALLOWED_OPERATORS = {"eq", "neq", "gt", "gte", "lt", "lte", "in", "contains", "is_null", "not_null"}
ALLOWED_AGGREGATIONS = {"sum", "avg", "min", "max", "count", "count_distinct"}
ALLOWED_INTENTS = {
    "aggregate",
    "filter",
    "compare",
    "top_n",
    "distinct_values",
    "stats",
    "retail_diagnostic",
    "clarify",
    "refuse",
}
