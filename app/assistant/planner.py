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
                    if session and session.accumulated_filters:
                        follow_up_triggers = ["among those", "for these", "filter to", "and also", "then", "now", "what about", "those", "these"]
                        q_lower = question.lower()
                        if any(trigger in q_lower for trigger in follow_up_triggers):
                            existing_fields = {f.field for f in llm_plan.filters}
                            merged = list(llm_plan.filters)
                            for sf in session.accumulated_filters:
                                if sf.field not in existing_fields:
                                    merged.append(sf)
                            llm_plan.filters = merged
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

        # Step B0: Check for greetings / assistant intro
        greetings = {"hi", "hello", "hey", "help", "who are you", "what can you do", "good morning", "good afternoon"}
        if q_lower in greetings or q_lower == "i need your help":
            return QueryPlan(
                intent="clarify",
                dataset=target_dataset,
                clarification_question="👋 Hello! I am your AI Retail Analytics Assistant. How can I help you today? You can ask me:\n• To calculate metrics (e.g. 'What is the total revenue in the West region?')\n• To look up records (e.g. 'Find customer Dana Evans' or 'Show orders with status Returned')\n• To analyze retail sales by category or store (e.g. 'Show total sales by category')",
            )

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

        # Check entity/name/ID search (e.g. "give dana evans", "find bob smith", "ORD-1001", "PRD-001")
        clean_name_cand = re.sub(
            r"^(i need your help\s+|please\s+|can you\s+)?(give\s+me\s+|give\s+|show\s+me\s+|show\s+|find\s+|who is\s+|lookup\s+|get\s+)?(customer\s+|product\s+|order\s+)?",
            "",
            q_lower,
        ).strip()

        id_match = re.search(r"\b([a-z]{3,4}-\d+)\b", q_lower)
        if id_match:
            matched_id = id_match.group(1).upper()
            id_col = next((c for c in raw_columns if any(c.lower().endswith(sfx) for sfx in ["_id", "id", "sku"])), None)
            if id_col:
                extracted_filters.append(QueryFilter(field=id_col, op="eq", value=matched_id))

        skip_words = {"revenue", "sales", "average", "total", "count", "performance", "orders", "products", "customers", "inventory"}
        if not extracted_filters and clean_name_cand and len(clean_name_cand) >= 3 and not any(w == clean_name_cand for w in skip_words):
            for col in raw_columns:
                series = base_df[col].dropna().astype(str)
                matches = [val for val in series.unique() if clean_name_cand in val.lower()]
                if matches:
                    extracted_filters.append(QueryFilter(field=col, op="contains", value=matches[0]))
                    break

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
            # Default to sum of revenue/amount ONLY if user asks for totals/sales or if grouping
            has_metric_keyword = any(w in q_lower for w in ["revenue", "sales", "spend", "sum", "total", "amount", "money"])
            if rev_cand and (has_metric_keyword or group_by):
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

            schema_json = json.dumps(QueryPlan.model_json_schema())

            prompt = (
                "You are an expert retail query planner. Translate user questions into a strictly valid JSON QueryPlan AST.\n"
                f"Target JSON Schema: {schema_json}\n"
                f"Available Schemas: {json.dumps(schema_summary)}\n"
                f"Active Dataset: {session.active_dataset if session else 'none'}\n"
                f"Question: {question}\n"
                "Allowed 'intent' values are strictly one of: 'aggregate', 'filter', 'compare', 'top_n', 'distinct_values', 'stats', 'retail_diagnostic', 'clarify', 'refuse'.\n"
                "Allowed filter operators are: 'eq' (for exact match, do not use '=='), 'neq', 'gt', 'gte', 'lt', 'lte', 'in', 'contains', 'is_null', 'not_null'.\n"
                "Allowed aggregations are: 'sum', 'avg', 'min', 'max', 'count', 'count_distinct'.\n"
                "- If the user input is a greeting ('hi', 'hello', 'hey', 'help'), return intent='clarify' with a helpful clarification_question welcoming the user.\n"
                "- If the user asks for a person, customer, or specific record (e.g. 'give Dana Evans', 'show customer Bob Smith', 'find PRD-001'), return intent='filter' with matching filter on 'name' or identifier column and empty metrics/group_by.\n"
                "- If the question is ambiguous, return intent='clarify'.\n"
                "- If unanswerable from the schema, return intent='refuse'.\n"
                "Never execute unrestricted code or invent non-existent columns.\n"
                "Output strictly valid raw JSON conforming to the Target JSON Schema and nothing else."
            )

            raw_text = None
            # 1. Try modern Interactions API
            try:
                interaction = client.interactions.create(
                    model=self.model_name,
                    input=prompt,
                )
                raw_text = interaction.output_text
            except Exception:
                # 2. Fallback to generate_content
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                raw_text = response.text

            if not raw_text:
                return None

            raw_text = raw_text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            plan_dict = json.loads(raw_text)
            if plan_dict.get("intent") == "query":
                plan_dict["intent"] = "aggregate" if (plan_dict.get("metrics") or plan_dict.get("group_by")) else "filter"
            elif plan_dict.get("intent") == "explain":
                plan_dict["intent"] = "retail_diagnostic"
            return QueryPlan(**plan_dict)
        except Exception:
            return None
