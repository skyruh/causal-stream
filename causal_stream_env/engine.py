import random
import time
import copy
from typing import List, Dict, Any, Optional
from .models import (
    Observation, DashboardMetrics, StreamSample, EventSnippet, 
    SQLModel, RootCauseEnum, Action, ReadDashboardAction,
    SampleStreamAction, InspectLineageAction, AskCounterfactualAction
)

class CausalStreamEngine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random = random.Random(seed)
        self.current_tick = 0
        self.max_ticks = 100
        self.aggregation_window = 300  # 5 minutes
        self.events_buffer: List[EventSnippet] = []
        self.sql_models: Dict[str, SQLModel] = {
            "aggregator": SQLModel(
                model_id="aggregator",
                sql="SELECT SUM(value) FROM events WHERE arrival_time - event_time < 300",
                dependencies=["raw_stream"]
            )
        }
        self.active_incident: Optional[RootCauseEnum] = None
        self._initialize_buffer()

    def _initialize_buffer(self):
        """Pre-fills the buffer with stochastic base events."""
        for i in range(100):
            self._generate_event()

    def _generate_event(self):
        """Generates a single event with stochastic latency and SLA metadata."""
        base_latency = 0.5
        jitter = self.random.uniform(0, 1.0)
        
        # Inject noise randomly
        is_phantom = self.random.random() < 0.05  # 5% chance of phantom/corrupted buffer event
        
        # Inject incident-specific latency
        if self.active_incident == RootCauseEnum.LATENCY_SPIKE and not is_phantom:
            base_latency += 5.0  # Force events outside the 300s window
        
        event_time = time.time() - (100 - len(self.events_buffer))
        arrival_time = event_time + base_latency + jitter
        
        actual_latency_ms = (arrival_time - event_time) * 1000.0
        sla_ms = 1000.0
        
        event = EventSnippet(
            event_id=f"evt_{self.random.getrandbits(32)}",
            event_time=event_time,
            arrival_time=arrival_time,
            provider="Stripe-Sim" if not is_phantom else "corrupted_buffer",
            status="success",
            sla_p99_latency_ms=sla_ms,
            actual_p99_latency_ms=actual_latency_ms,
            sla_breach=actual_latency_ms > sla_ms
        )
        self.events_buffer.append(event)
        
        # Duplicate occasionally
        if self.random.random() < 0.02:
            dup_event = copy.deepcopy(event)
            dup_event.arrival_time += self.random.uniform(0, 0.5)
            self.events_buffer.append(dup_event)
            
        if len(self.events_buffer) > 500:
            self.events_buffer = self.events_buffer[-500:]

    def tick(self, count: int = 1):
        """Increments the world clock and updates the stream."""
        for _ in range(count):
            self.current_tick += 1
            self._generate_event()

    def get_observation(self) -> Observation:
        # Calculate mock metrics based on buffer and active incident
        revenue = 1000.0
        if self.active_incident:
            revenue *= 0.9  # 10% drop
            
        metrics = DashboardMetrics(
            revenue=revenue,
            error_rate=0.02,
            avg_latency=1.2,
            active_users=5000
        )
        
        return Observation(
            current_tick=self.current_tick,
            dashboard=metrics,
            alert_feed=["Critical: 10% Revenue Drop Detected" if self.active_incident else "System Nominal"]
        )

    def step(self, action: Action) -> Observation:
        """Executes an action and returns the new observation."""
        obs = self.get_observation()
        
        if action.type == "read_dashboard":
            pass
        elif action.type == "sample_stream":
            self.tick(1)
            obs = self.get_observation()
            obs.last_sample = StreamSample(
                events=self.events_buffer[-action.sample_size:],
                tick=self.current_tick
            )
        elif action.type == "inspect_lineage":
            self.tick(1)
            obs = self.get_observation()
            if action.model_id in self.sql_models:
                obs.inspected_lineage = self.sql_models[action.model_id]
        elif action.type == "ask_counterfactual":
            self.tick(2) # Counterfactuals are expensive
            obs = self.get_observation()
            obs.alert_feed.append(f"Counterfactual: With window offset {action.window_offset}s, Revenue would be {obs.dashboard.revenue * 1.05:.2f}")
        elif action.type == "query_metadata":
            self.tick(1)
            obs = self.get_observation()
            if self.active_incident == RootCauseEnum.EXPECTED_MAINTENANCE:
                obs.alert_feed.append(f"Metadata Query [{action.table_name}]: MAINT_WINDOW_0800_1000 matched. SYSTEM_EVENTS_METADATA shows maintenance occurred.")
            else:
                obs.alert_feed.append(f"Metadata Query [{action.table_name}]: Normal. No maintenance logs found.")
        elif action.type == "check_sla":
            self.tick(1)
            obs = self.get_observation()
            if self.active_incident == RootCauseEnum.LATENCY_SPIKE:
                obs.alert_feed.append(f"SLA Check [{action.provider_id}]: SLA_BREACH_STRIPE true, P99 is > 3000ms.")
            else:
                obs.alert_feed.append(f"SLA Check [{action.provider_id}]: SLA met. P99 is nominal.")
            
        return obs

    def set_incident(self, incident: RootCauseEnum):
        self.active_incident = incident
