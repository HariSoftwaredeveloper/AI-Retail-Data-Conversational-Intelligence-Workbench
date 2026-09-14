"""Dual-mode query planner: Deterministic Rule Planner + Gemini LLM fallback.
"""
from typing import Any, Dict, List, Optional
import json
import re
import pandas as pd
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL, LLM_PROVIDER
from app.core.models import (
    ChatSessionState,
    QueryFilter,
    QueryJoin,
    QueryMetric,
    QueryPlan,
    QuerySort,
)
from app.core.security import check_for_prompt_injection


class QueryPlanner:
    """Translates natural language questions into constrained AST QueryPlans."""

    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        self.api_key = api_key
        self.model_name = model_name

    def plan(
        self,
        question: str,
        loaded_datasets: Dict[str, pd.DataFrame],
        session: Optional[ChatSessionState] = None,
    ) -> QueryPlan:
        """Generate a validated plan from question + session context."""
        # 1. Prompt Injection Defense
        if check_for_prompt_injection(question):
            return QueryPlan(
                intent="refuse",
                dataset=session.active_dataset if session and session.active_dataset else list(loaded_datasets.keys())[0],
                refusal_reason="Security Alert: Prompt injection pattern detected. Question refused.",
            )

        # 2. Check if datasets are loaded
        if not loaded_datasets:
            return QueryPlan(
                intent="refuse",
                dataset="none",
                refusal_reason="No datasets are currently loaded in the workbench.",
            )

        # 3. Try LLM if configured and key is present
        if self.api_key and LLM_PROVIDER in ("gemini", "hybrid"):
            try:
                llm_plan = self._plan_with_gemini(question, loaded_datasets, session)
                if llm_plan:
                    return llm_plan
            except Exception:
                # Gracefully fallback to deterministic rule planner
                pass

        # 4. Deterministic Rule Planner
        return self._plan_with_rules(question, loaded_datasets, session)

    def _plan_with_rules(
        self,
        question: str,
        loaded_datasets: Dict[str, pd.DataFrame],
        session: Optional[ChatSessionState] = None,
    ) -> QueryPlan:
        """Deterministic semantic rule planner."""
        q_lower = question.lower().strip()

        # Step A: Identify target dataset
        target_dataset = None
        # Check explicit mention in question (e.g. 'in orders_clean', 'products_raw')
        for dname in loaded_datasets.keys():
            if dname.lower() in q_lower:
                target_dataset = dname
                break

        # Fallback to session active dataset or preferred clean dataset
        if not target_dataset:
            if session and session.active_dataset and session.active_dataset in loaded_datasets:
                target_dataset = session.active_dataset
            else:
                # Prefer clean versions
                clean_names = [d for d in loaded_datasets.keys() if d.endswith("_clean")]
                target_dataset = clean_names[0] if clean_names else list(loaded_datasets.keys())[0]

        base_df = loaded_datasets[target_dataset]
        columns = [c.lower() for c in base_df.columns]
        raw_columns = list(base_df.columns)

        # Step B: Check for unanswerable topics
        unanswerable_topics = ["weather", "temperature", "inflation", "stock market", "ceo salary", "competitor"]
        if any(topic in q_lower for topic in unanswerable_topics):
            return QueryPlan(
                intent="refuse",
                dataset=target_dataset,
                refusal_reason=f"The question references external facts not tracked in dataset '{target_dataset}'.",
            )

        # Step C: Check for ambiguity
        if q_lower in ("what is the best?", "how is the performance?", "tell me about the data", "best item"):
            return QueryPlan(
                intent="clarify",
                dataset=target_dataset,
                clarification_question="Could you specify which metric to evaluate (e.g. highest revenue, highest margin, or lowest return rate)?",
            )

        # Step D: Multi-turn Follow-up handling
        inherited_filters: List[QueryFilter] = []
        is_follow_up = False
        follow_up_triggers = ["among those", "for these", "filter to", "and also", "then", "now", "what about"]
        if any(trigger in q_lower for trigger in follow_up_triggers) and session:
            inherited_filters = list(session.accumulated_filters)
            is_follow_up = True

        # Step E: Extract Filters
        extracted_filters: List[QueryFilter] = []

        # Check region filter
        for region in ["west", "east", "north", "south", "central"]:
            if region in q_lower:
                reg_col = next((c for c in raw_columns if "region" in c.lower()), None)
                if reg_col:
                    extracted_filters.append(QueryFilter(field=reg_col, op="eq", value=region.capitalize()))

        # Check status filter (returned, completed, pending)
        for status in ["returned", "refunded", "completed", "shipped", "pending", "cancelled"]:
            if status in q_lower:
                stat_col = next((c for c in raw_columns if "status" in c.lower()), None)
                if stat_col:
                    extracted_filters.append(QueryFilter(field=stat_col, op="eq", value=status.capitalize()))

        # Check category filter
        for cat in ["apparel", "electronics", "footwear", "home", "beauty"]:
            if cat in q_lower:
                cat_col = next((c for c in raw_columns if "category" in c.lower()), None)
                if cat_col:
                    extracted_filters.append(QueryFilter(field=cat_col, op="contains", value=cat))

        # Check numeric filter (e.g. price > 100)
        num_filter_match = re.search(r"(price|amount|revenue|stock)\s*(>|<|>=|<=|=)\s*(\$?\d+(\.\d+)?)", q_lower)
        if num_filter_match:
            field_cand = num_filter_match.group(1)
            op_str = num_filter_match.group(2)
            val_str = num_filter_match.group(3).replace("$", "")
            target_col = next((c for c in raw_columns if field_cand in c.lower()), None)
            if target_col:
                op_map = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "=": "eq"}
                extracted_filters.append(
                    QueryFilter(field=target_col, op=op_map.get(op_str, "eq"), value=float(val_str))
                )

        combined_filters = inherited_filters + extracted_filters

        # Step F: Identify Joins if multi-dataset
        joins: List[QueryJoin] = []
        if len(loaded_datasets) > 1:
            # If question asks about category or product while on orders dataset
            if ("category" in q_lower or "product" in q_lower) and "product_id" in raw_columns:
                prod_ds = next((d for d in loaded_datasets.keys() if "product" in d.lower() and d != target_dataset), None)
                if prod_ds:
                    joins.append(QueryJoin(dataset=prod_ds, left_on="product_id", right_on="product_id", how="inner"))
            elif "customer" in q_lower and "customer_id" in raw_columns:
                cust_ds = next((d for d in loaded_datasets.keys() if "customer" in d.lower() and d != target_dataset), None)
                if cust_ds:
                    joins.append(QueryJoin(dataset=cust_ds, left_on="customer_id", right_on="customer_id", how="inner"))

        # Step G: Identify Group By
        group_by = []
        for term in ["category", "region", "store", "store_id", "status", "segment", "brand"]:
            if term in q_lower:
                match_col = next((c for c in raw_columns if c.lower() == term or f"_{term}" in c.lower() or f"{term}_" in c.lower()), None)
                if match_col:
                    group_by.append(match_col)
                elif joins:
                    # Check joined table columns
                    right_cols = loaded_datasets[joins[0].dataset].columns
                    match_right = next((c for c in right_cols if term in c.lower()), None)
                    if match_right:
                        group_by.append(match_right)

        # Step H: Identify Metrics & Aggregation
        metrics: List[QueryMetric] = []
        rev_cand = next((c for c in raw_columns if c.lower() in ("total_amount", "revenue", "sales", "price", "amount")), None)

        if "return rate" in q_lower or "highest return" in q_lower:
            stat_col = next((c for c in raw_columns if "status" in c.lower()), None)
            if stat_col:
                # Filter to returned or compute count
                metrics.append(QueryMetric(agg="count", field=stat_col, alias="returned_count"))
            elif rev_cand:
                metrics.append(QueryMetric(agg="count", field="*", alias="record_count"))
        elif "average" in q_lower or "aov" in q_lower or "mean" in q_lower:
            if rev_cand:
                metrics.append(QueryMetric(agg="avg", field=rev_cand, alias="average_amount"))
        elif "count" in q_lower or "how many" in q_lower or "number of" in q_lower:
            metrics.append(QueryMetric(agg="count", field="*", alias="total_count"))
        elif "distinct" in q_lower or "unique" in q_lower:
            first_col = raw_columns[0]
            metrics.append(QueryMetric(agg="count_distinct", field=first_col, alias=f"unique_{first_col}"))
        else:
            # Default to sum of revenue/amount if available
            if rev_cand:
                metrics.append(QueryMetric(agg="sum", field=rev_cand, alias="total_sales"))

        # Step I: Sort & Limit
        sort = []
        metric_alias = metrics[0].alias if metrics else None
        if "highest" in q_lower or "top" in q_lower or "most" in q_lower or "best" in q_lower:
            if metric_alias:
                sort.append(QuerySort(field=metric_alias, dir="desc"))
        elif "lowest" in q_lower or "bottom" in q_lower or "least" in q_lower:
            if metric_alias:
                sort.append(QuerySort(field=metric_alias, dir="asc"))
        elif group_by and metric_alias:
            sort.append(QuerySort(field=metric_alias, dir="desc"))

        # Limit
        limit = 10
        limit_match = re.search(r"top\s*(\d+)", q_lower)
        if limit_match:
            limit = int(limit_match.group(1))

        # Determine Intent
        intent = "aggregate" if (metrics or group_by) else "filter"

        return QueryPlan(
            intent=intent,
            dataset=target_dataset,
            joins=joins,
            filters=combined_filters,
            group_by=group_by,
            metrics=metrics,
            sort=sort,
            limit=limit,
        )

    def _plan_with_gemini(
        self,
        question: str,
        loaded_datasets: Dict[str, pd.DataFrame],
        session: Optional[ChatSessionState] = None,
    ) -> Optional[QueryPlan]:
        """Calls Gemini API with constrained schema definition."""
        try:
            import google.genai as genai
            client = genai.Client(api_key=self.api_key)

            # Summarize schemas without leaking raw values or PII
            schema_summary = {}
            for name, df in loaded_datasets.items():
                schema_summary[name] = list(df.columns)

            system_instruction = (
                "You are an expert retail query planner. Translate user questions into a strictly valid JSON QueryPlan AST. "
                "Output ONLY valid JSON matching the QueryPlan schema. "
                "If question is ambiguous, return intent='clarify'. "
                "If unanswerable from schema, return intent='refuse'. "
                "Never execute unrestricted code or invent non-existent columns."
            )

            prompt = (
                f"Available Schemas: {json.dumps(schema_summary)}\n"
                f"Active Dataset: {session.active_dataset if session else 'none'}\n"
                f"Question: {question}\n"
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"system_instruction": system_instruction, "response_mime_type": "application/json"},
            )

            raw_text = response.text
            plan_dict = json.loads(raw_text)
            return QueryPlan(**plan_dict)
        except Exception:
            return None
