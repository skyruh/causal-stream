from typing import Dict, Any, Tuple
from .engine import CausalStreamEngine
from .models import Observation, Action, RootCauseEnum
from .tasks import Task1, Task2, Task3, TaskGrader

class CausalStreamEnv:
    def __init__(self, task_id: int = 1):
        self.engine = CausalStreamEngine()
        self.task_id = task_id
        self.task = self._load_task(task_id)
        self.reset()

    def _load_task(self, task_id: int):
        if task_id == 1: return Task1()
        if task_id == 2: return Task2()
        return Task3()

    def reset(self) -> Observation:
        self.engine = CausalStreamEngine(seed=42 + self.task_id)
        self.engine.set_incident(self.task.ground_truth_cause)
        return self.engine.get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        # Validate action via Pydantic (usually handled by the server wrapper)
        obs = self.engine.step(action)
        
        reward = 0.0
        done = False
        
        if action.type == "submit_theory":
            score = 0.0
            if action.cause == self.task.ground_truth_cause:
                score += 0.5
                evidence_score = TaskGrader.calculate_f1(action.evidence, self.task.ground_truth_evidence)
                score += (evidence_score * 0.5)
            
            reward = score
            done = True
            
        # Time penalty
        if self.engine.current_tick > self.engine.max_ticks:
            done = True
            
        return obs, reward, done, {}

    def get_state(self) -> Observation:
        return self.engine.get_observation()
