from typing import List, Set
from .models import RootCauseEnum

class TaskGrader:
    @staticmethod
    def calculate_f1(submitted: List[str], ground_truth: Set[str]) -> float:
        if not submitted:
            return 0.0
        
        sub_set = set(submitted)
        intersection = sub_set.intersection(ground_truth)
        
        precision = len(intersection) / len(sub_set) if sub_set else 0.0
        recall = len(intersection) / len(ground_truth) if ground_truth else 0.0
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)

class CausalTask:
    def __init__(self, name: str, difficulty: str, goal: str):
        self.name = name
        self.difficulty = difficulty
        self.goal = goal
        self.ground_truth_cause: RootCauseEnum = RootCauseEnum.LATENCY_SPIKE
        self.ground_truth_evidence: Set[str] = set()

class Task1(CausalTask):
    def __init__(self):
        super().__init__(
            "Structural Join Mismatch", 
            "Easy", 
            "Identify why 10% of revenue events are missing from the aggregator."
        )
        self.ground_truth_cause = RootCauseEnum.JOIN_FAILURE
        self.ground_truth_evidence = {"NULL_KEY_ERR", "JOIN_MISMATCH_404"}

class Task2(CausalTask):
    def __init__(self):
        super().__init__(
            "Temporal Jitter Drift", 
            "Medium", 
            "Investigate why revenue dropped during a period of high network jitter."
        )
        self.ground_truth_cause = RootCauseEnum.OUT_OF_ORDER
        self.ground_truth_evidence = {"ARRIVAL_GT_EVENT_TIME", "WINDOW_TIMEOUT"}

class Task3(CausalTask):
    def __init__(self):
        super().__init__(
            "The Multi-Variable Ghost", 
            "Hard", 
            "Diagnose an intermittent 5% drop using counterfactuals and precise evidence."
        )
        self.ground_truth_cause = RootCauseEnum.LATENCY_SPIKE
        self.ground_truth_evidence = {"STRIPE_WEBHOOK_DELAY", "P99_LATENCY_3000MS"}

class Task4(CausalTask):
    def __init__(self):
        super().__init__(
            "The Red Herring Storm", 
            "Adversarial", 
            "Diagnose a 15% revenue drop hidden among red herrings like high latency and join mismatches."
        )
        self.ground_truth_cause = RootCauseEnum.EXPECTED_MAINTENANCE
        self.ground_truth_evidence = {"MAINT_WINDOW_0800_1000", "SYSTEM_EVENTS_METADATA"}
