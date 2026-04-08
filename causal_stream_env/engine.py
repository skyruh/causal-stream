import random
import time
import copy
from typing import List, Dict, Any, Optional
from .models import (
    Observation, DashboardMetrics, StreamSample, EventSnippet, 
    SQLModel, RootCauseEnum, Action, ReadDashboardAction,
    SampleStreamAction, InspectLineageAction, SimulateConfigChangeAction
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
        """Generates a single event with stochastic latency and incident evidence."""
        base_latency = 0.5
        jitter = self.random.uniform(0, 1.0)
        status = "success"
        evidence_tokens = []
        is_phantom = self.random.random() < 0.05
        
        # Incident logic
        if self.active_incident == RootCauseEnum.LATENCY_SPIKE and not is_phantom:
            base_latency += 5.0
            if self.random.random() < 0.2:
                evidence_tokens.append("STRIPE_WEBHOOK_DELAY")
                evidence_tokens.append("P99_LATENCY_3000MS")
        
        elif self.active_incident == RootCauseEnum.JOIN_FAILURE and not is_phantom:
            if self.random.random() < 0.3:
                status = "error"
                evidence_tokens.append("NULL_KEY_ERR")
                evidence_tokens.append("JOIN_MISMATCH_404")
        
        elif self.active_incident == RootCauseEnum.OUT_OF_ORDER and not is_phantom:
            jitter += 350.0  # Force it past the 300s window
            if self.random.random() < 0.2:
                evidence_tokens.append("ARRIVAL_GT_EVENT_TIME")
                evidence_tokens.append("WINDOW_TIMEOUT")

        # Use a true deterministic clock
        base_epoch = 1700000000.0
        event_time = base_epoch + self.current_tick - (100 - len(self.events_buffer))
        arrival_time = event_time + base_latency + jitter
        
        actual_latency_ms = (arrival_time - event_time) * 1000.0
        sla_ms = 1000.0
        
        event = EventSnippet(
            event_id=f"evt_{self.random.getrandbits(32)}",
            event_time=event_time,
            arrival_time=arrival_time,
            provider="Stripe-Sim" if not is_phantom else "corrupted_buffer",
            status=status,
            sla_p99_latency_ms=sla_ms,
            actual_p99_latency_ms=actual_latency_ms,
            sla_breach=actual_latency_ms > sla_ms,
            evidence_tokens=evidence_tokens
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
        elif action.type == "simulate_config_change":
            self.tick(2) # Simulations are expensive
            obs = self.get_observation()
            obs.alert_feed.append(f"Simulation: Changed {action.config_param} to {action.value}. Revenue would be {obs.dashboard.revenue * 1.05:.2f}")
        elif action.type == "query_system_logs":
            self.tick(1)
            obs = self.get_observation()
            if self.active_incident == RootCauseEnum.EXPECTED_MAINTENANCE:
                obs.alert_feed.append(f"System Logs [{action.log_name}]: MAINT_WINDOW_0800_1000 matched. SYSTEM_EVENTS shows maintenance occurred.")
            else:
                obs.alert_feed.append(f"System Logs [{action.log_name}]: Normal. No maintenance logs found.")
        elif action.type == "query_provider_contract":
            self.tick(1)
            obs = self.get_observation()
            if self.active_incident == RootCauseEnum.LATENCY_SPIKE:
                obs.alert_feed.append(f"Contract Check [{action.provider_id}]: SLA_BREACH_STRIPE true, P99 is > 3000ms.")
            else:
                obs.alert_feed.append(f"Contract Check [{action.provider_id}]: SLA met. P99 is nominal.")
            
        return obs

    def set_incident(self, incident: RootCauseEnum):
        self.active_incident = incident
