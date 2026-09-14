"""Domain models and Pydantic schemas for the Workbench.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class Issue(BaseModel):
    id: str
    field: Optional[str] = None
    issue_type: str  # mixed_types, inconsistent_dates, whitespace_case, duplicate_records, suspicious_categories, currency_formats, missing_values
    severity: str  # low, medium, high
    description: str
    sample_values: List[Any] = Field(default_factory=list)
    affected_count: int = 0


class CleaningStep(BaseModel):
    step_id: str
    reason: str
    affected_fields: List[str]
    risk: str = "low"  # low, medium, high
    source: str = "code"  # code | ai
    transformation_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # pending, applied, skipped, failed


class CleaningPlan(BaseModel):
    run_id: str
    dataset_name: str
    steps: List[CleaningStep]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ColumnProfile(BaseModel):
    name: str
    inferred_type: str
    null_count: int
    null_percentage: float
    distinct_count: int
    cardinality_ratio: float
    is_unique: bool
    sample_values: List[Any] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)  # min, max, mean, median, std, top_frequencies
    parse_failures: int = 0


class DatasetProfile(BaseModel):
    dataset_name: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: Dict[str, ColumnProfile]
    profiled_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DeltaValidation(BaseModel):
    rows_before: int
    rows_after: int
    row_delta: int
    schema_before: List[str]
    schema_after: List[str]
    schema_mutated: bool
    unresolved_issues: List[str] = Field(default_factory=list)
    idempotent: bool = True


class QueryFilter(BaseModel):
    field: str
    op: str  # eq, neq, gt, gte, lt, lte, in, contains, is_null, not_null
    value: Any


class QueryMetric(BaseModel):
    agg: str  # sum, avg, min, max, count, count_distinct
    field: str
    alias: Optional[str] = None


class QuerySort(BaseModel):
    field: str
    dir: str = "desc"  # asc | desc


class QueryJoin(BaseModel):
    dataset: str
    left_on: str
    right_on: str
    how: str = "inner"  # inner, left


class QueryPlan(BaseModel):
    intent: str  # aggregate, filter, compare, top_n, distinct_values, stats, explain, clarify, refuse
    dataset: str
    joins: List[QueryJoin] = Field(default_factory=list)
    filters: List[QueryFilter] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    metrics: List[QueryMetric] = Field(default_factory=list)
    sort: List[QuerySort] = Field(default_factory=list)
    limit: int = 10
    explanation_topic: Optional[str] = None
    clarification_question: Optional[str] = None
    refusal_reason: Optional[str] = None


class ExecutionEvidence(BaseModel):
    dataset_version: str
    columns_used: List[str]
    filters_applied: List[Dict[str, Any]]
    joins_performed: List[Dict[str, Any]]
    operations: List[str]
    row_count_considered: int
    rows_returned: int
    preview: List[Dict[str, Any]] = Field(default_factory=list)
    scalar_answer: Optional[Union[float, int, str]] = None


class ChatTurn(BaseModel):
    question: str
    plan: Dict[str, Any]
    answer: Dict[str, Any]
    trace: Dict[str, Any]
    status: str = "ok"  # ok, clarified, refused, error
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChatSessionState(BaseModel):
    session_id: str
    active_dataset: Optional[str] = None
    accumulated_filters: List[QueryFilter] = Field(default_factory=list)
    history: List[ChatTurn] = Field(default_factory=list)
