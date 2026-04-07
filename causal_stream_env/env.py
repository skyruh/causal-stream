from typing import Dict, Any, Tuple
from .engine import CausalStreamEngine
from .models import Observation, Action, RootCauseEnum
from .tasks import Task1, Task2, Task3, Task4, TaskGrader

class CausalStreamEnv:
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.task = self._load_task(task_id)
        self.action_history_rewards = set()
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
        return self.engine.get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        obs = self.engine.step(action)
        
        reward = 0.0
        done = False
        
        if action.type not in self.action_history_rewards:
            self.action_history_rewards.add(action.type)
            if action.type == "read_dashboard":
                reward += 0.05
            elif action.type == "sample_stream":
                reward += 0.10
            elif action.type == "inspect_lineage":
                reward += 0.10
            elif action.type == "simulate_config_change":
                reward += 0.15
            elif action.type == "query_system_logs":
                reward += 0.10
            elif action.type == "query_provider_contract":
                reward += 0.10

        if action.type == "submit_theory":
            score = 0.0
            if action.cause == self.task.ground_truth_cause:
                score += 0.2
                evidence_score = TaskGrader.calculate_f1(action.evidence, self.task.ground_truth_evidence)
                score += (evidence_score * 0.2)
            
            reward += score
            # The agent must still submit a post-mortem to finish the episode!
            
        if action.type == "submit_postmortem":
            # Graded on:
            score = 0.0
            if action.prevention_action.value == self._get_expected_prevention():
                score += 0.1
            if abs(action.impact_duration_ticks - 100) <= 20: 
                score += 0.1
            
            reward += score
            done = True
            
        # Time penalty
        if self.engine.current_tick > self.engine.max_ticks:
            reward -= 0.05
            done = True
            
        return obs, min(max(reward, -1.0), 1.0), done, {}

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

