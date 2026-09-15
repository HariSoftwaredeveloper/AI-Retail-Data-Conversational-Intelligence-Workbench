"""Grounded response and retail conversational explanation builder.
"""
from typing import Any, Dict, List, Optional
from app.core.models import ExecutionEvidence, QueryPlan


def format_grounded_answer(
    plan: QueryPlan,
    evidence: Optional[ExecutionEvidence],
    raw_question: str
) -> Dict[str, Any]:
    """Generate a verifiable answer grounded strictly in computational evidence."""
    # 1. Handle Refusal
    if plan.intent == "refuse":
        return {
            "summary": plan.refusal_reason or "I cannot answer this question from the available dataset.",
            "type": "refusal",
            "grounded": True,
            "evidence": None,
        }

    # 2. Handle Clarification
    if plan.intent == "clarify":
        return {
            "summary": plan.clarification_question or "Could you clarify which metric or field you would like to analyze?",
            "type": "clarification",
            "grounded": True,
            "evidence": None,
        }

    if not evidence:
        return {
            "summary": "Execution produced no evidence.",
            "type": "error",
            "grounded": False,
            "evidence": None,
        }

    # 3. Handle Scalar Result (Single metric calculation)
    if evidence.scalar_answer is not None:
        val = evidence.scalar_answer
        metric_name = plan.metrics[0].alias or plan.metrics[0].field if plan.metrics else "Result"
        formatted_val = f"${val:,.2f}" if isinstance(val, (int, float)) and any(t in metric_name.lower() for t in ["rev", "sale", "price", "amount", "cost"]) else f"{val:,}" if isinstance(val, int) else f"{val}"
        summary = f"Based on dataset **{evidence.dataset_version}** ({evidence.row_count_considered} rows evaluated), the computed **{metric_name}** is **{formatted_val}**."
        return {
            "summary": summary,
            "scalar_value": val,
            "type": "scalar",
            "grounded": True,
            "evidence": evidence.model_dump(),
        }

    # 4. Handle Grouped / Top-N Results
    preview = evidence.preview
    if not preview:
        return {
            "summary": f"No records matched the query criteria in dataset **{evidence.dataset_version}**.",
            "type": "empty_result",
            "grounded": True,
            "evidence": evidence.model_dump(),
        }

    top_item = preview[0]
    primary_group = plan.group_by[0] if plan.group_by else list(top_item.keys())[0]
    top_label = top_item.get(primary_group, "Record")

    # If it is a pure filter or record retrieval without group_by/aggregations
    if plan.intent == "filter" or (not plan.group_by and not plan.metrics):
        if len(preview) == 1:
            details = [f"**{k}**: {v}" for k, v in top_item.items() if k != primary_group and v is not None]
            summary = f"Found matching record in **{evidence.dataset_version}**: **{top_label}** ({', '.join(details[:4])})."
        else:
            summary = f"Found {len(preview)} matching record(s) in **{evidence.dataset_version}** ({evidence.row_count_considered} rows evaluated). Showing records below."
    else:
        metric_keys = [k for k in top_item.keys() if k not in plan.group_by]
        primary_metric = metric_keys[0] if metric_keys else None
        top_metric_val = top_item.get(primary_metric, "") if primary_metric else ""

        summary = (
            f"In **{evidence.dataset_version}** across {evidence.row_count_considered} records, "
            f"**{top_label}** ranked highest with **{top_metric_val}** in {primary_metric or 'the metric'}. "
            f"Showing top {len(preview)} records below."
        )

    return {
        "summary": summary,
        "type": "table",
        "grounded": True,
        "top_item": top_item,
        "evidence": evidence.model_dump(),
    }
