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

class PreventionEnum(str, Enum):
    INCREASE_TIMEOUT = "increase_timeout"
    ADD_INDEX = "add_index"
    BLOCK_DUPLICATES = "block_duplicates"
    UPDATE_SCHEMA = "update_schema"
    SCHEDULED_MAINTENANCE_SYNC = "scheduled_maintenance_sync"

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
    sla_p99_latency_ms: float
    actual_p99_latency_ms: float
    sla_breach: bool
    evidence_tokens: List[str] = []

class IncidentEvent(BaseModel):
    tick: int
    description: str

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

class SimulateConfigChangeAction(BaseModel):
    type: Literal["simulate_config_change"] = "simulate_config_change"
    config_param: str = Field(description="The configuration parameter to change (e.g. 'aggregation_window')")
    value: int = Field(description="The new value to simulate")

class QuerySystemLogsAction(BaseModel):
    type: Literal["query_system_logs"] = "query_system_logs"
    log_name: str

class QueryProviderContractAction(BaseModel):
    type: Literal["query_provider_contract"] = "query_provider_contract"
    provider_id: str

class SubmitPostmortemAction(BaseModel):
    type: Literal["submit_postmortem"] = "submit_postmortem"
    timeline: List[IncidentEvent]
    impact_duration_ticks: int
    prevention_action: PreventionEnum

Action = Union[
    ReadDashboardAction,
    SampleStreamAction,
    InspectLineageAction,
    SubmitTheoryAction,
    PatchAggregatorAction,
    SimulateConfigChangeAction,
    QuerySystemLogsAction,
    QueryProviderContractAction,
    SubmitPostmortemAction
]
