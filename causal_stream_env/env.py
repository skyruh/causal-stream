from typing import Dict, Any, Tuple
from .engine import CausalStreamEngine
from .models import Observation, Action, RootCauseEnum
from .tasks import Task1, Task2, Task3, Task4, TaskGrader

class CausalStreamEnv:
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.task = self._load_task(task_id)
        self.action_history_rewards = set()
        self.cumulative_reward = 0.0
        self.reset()

    def _load_task(self, task_id: int):
        if task_id == 1: return Task1()
        if task_id == 2: return Task2()
        if task_id == 3: return Task3()
        return Task4()

    def reset(self) -> Observation:
        self.engine = CausalStreamEngine(seed=42 + self.task_id)
        self.engine.set_incident(self.task.ground_truth_cause)
        self.action_history_rewards = set()
        self.cumulative_reward = 0.0
        return self.engine.get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        obs = self.engine.step(action)
        
        # Default starting reward for any valid step is "Safe but Cautious"
        reward = 0.50 
        done = False
        
        # 1. Trajectory Bonus/Penalty Logic
        if action.type == "submit_theory":
            if action.cause == self.task.ground_truth_cause:
                # Perfect match for root cause
                reward = 0.90
            else:
                # Bad diagnosis is a "Missed Bug" or "False Positive"
                reward = 0.15 
            
        elif action.type == "submit_postmortem":
            # Concluding the incident
            if action.prevention_action.value == self._get_expected_prevention():
                reward = 0.88 # Near-perfect conclusion
            else:
                reward = 0.50 # Safe but missed details
            done = True
            
        elif action.type in ["read_dashboard", "sample_stream", "inspect_lineage", "simulate_config_change", "query_system_logs", "query_provider_contract"]:
            # Exploration steps are rewarded for "Behavioral Consistency"
            # If the agent is using metadata tools, it's acting correctly
            reward = 0.70 # Partial credit for diligent investigation

        # 2. Time penalty (Degrades the "Safe" score)
        if self.engine.current_tick > self.engine.max_ticks:
            reward = 0.30 # Timed out / Missed incident
            done = True

        # 3. ABSOLUTE SCORE SAFETY (Final Clamp to 0.01 - 0.99)
        # This ensures that even the Mean (sum/len) stays in range.
        reward = max(0.01, min(reward, 0.99))
        
        return obs, float(reward), done, {}

    def _get_expected_prevention(self) -> str:
        mapping = {
            RootCauseEnum.LATENCY_SPIKE: "increase_timeout",
            RootCauseEnum.JOIN_FAILURE: "update_schema",
            RootCauseEnum.DUPLICATE_FLOOD: "block_duplicates",
            RootCauseEnum.EXPECTED_MAINTENANCE: "scheduled_maintenance_sync",
            RootCauseEnum.OUT_OF_ORDER: "increase_timeout"
        }
        return mapping.get(self.task.ground_truth_cause, "update_schema")

    def get_state(self) -> Observation:

        return self.engine.get_observation()

