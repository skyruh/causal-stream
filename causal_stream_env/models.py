from enum import Enum
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field

class RootCauseEnum(str, Enum):
    LATENCY_SPIKE = "latency_spike"
    JOIN_FAILURE = "join_failure"
    DUPLICATE_FLOOD = "duplicate_flood"
    SCHEMA_DRIFT = "schema_drift"
    OUT_OF_ORDER = "out_of_order"
    EXPECTED_MAINTENANCE = "expected_maintenance"

class DashboardMetrics(BaseModel):
    revenue: float
    error_rate: float
    avg_latency: float
    active_users: int

class EventSnippet(BaseModel):
    event_id: str
    event_time: float
    arrival_time: float
    provider: str
    status: str

class StreamSample(BaseModel):
    events: List[EventSnippet]
    tick: int

class SQLModel(BaseModel):
    model_id: str
    sql: str
    dependencies: List[str]

class Observation(BaseModel):
    current_tick: int
    dashboard: DashboardMetrics
    last_sample: Optional[StreamSample] = None
    inspected_lineage: Optional[SQLModel] = None
    alert_feed: List[str]

# --- Actions (Discriminated Union) ---

class ReadDashboardAction(BaseModel):
    type: Literal["read_dashboard"] = "read_dashboard"

class SampleStreamAction(BaseModel):
    type: Literal["sample_stream"] = "sample_stream"
    sample_size: int = Field(default=10, ge=1, le=100)

class InspectLineageAction(BaseModel):
    type: Literal["inspect_lineage"] = "inspect_lineage"
    model_id: str

class SubmitTheoryAction(BaseModel):
    type: Literal["submit_theory"] = "submit_theory"
    cause: RootCauseEnum
    evidence: List[str] = Field(description="List of IDs or timestamps proving the cause")

class PatchAggregatorAction(BaseModel):
    type: Literal["patch_aggregator"] = "patch_aggregator"
    model_id: str
    new_sql: str

class AskCounterfactualAction(BaseModel):
    type: Literal["ask_counterfactual"] = "ask_counterfactual"
    window_offset: int = Field(description="Seconds to add to the aggregation window")

class QueryMetadataAction(BaseModel):
    type: Literal["query_metadata"] = "query_metadata"
    table_name: str

class CheckSlaAction(BaseModel):
    type: Literal["check_sla"] = "check_sla"
    provider_id: str

Action = Union[
    ReadDashboardAction,
    SampleStreamAction,
    InspectLineageAction,
    SubmitTheoryAction,
    PatchAggregatorAction,
    AskCounterfactualAction,
    QueryMetadataAction,
    CheckSlaAction
]
