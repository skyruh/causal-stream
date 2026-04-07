import os
import requests
import random
from typing import Dict, Any

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = "RandomAgent"
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

def reset_env(task_id: int):
    return requests.post(f"{ENV_URL}/reset?task_id={task_id}", json={}).json()

def step_env(task_id: int, action: Dict[str, Any]):
    return requests.post(f"{ENV_URL}/step?task_id={task_id}", json=action).json()

def run_agent(task_id: int):
    task_name = f"task-{task_id}"
    print(f"[START] task={task_name} env=causal-stream-v3 model={MODEL_NAME}", flush=True)
    
    obs = reset_env(task_id)
    rewards = []
    steps_taken = 0
    done = False

    actions = [
        {"type": "read_dashboard"},
        {"type": "sample_stream", "sample_size": 20},
        {"type": "inspect_lineage", "model_id": "aggregator"},
        {"type": "query_system_logs", "log_name": "system_events"},
        {"type": "query_provider_contract", "provider_id": "Stripe-Sim"},
        {"type": "submit_theory", "cause": "latency_spike", "evidence": []},
        {"type": "submit_postmortem", "timeline": [{"tick": 0, "description": "Random guess"}], "impact_duration_ticks": 100, "prevention_action": "update_schema"}
    ]
    
    while not done and steps_taken < 10:
        action = random.choice(actions)
        resp = step_env(task_id, action)
        reward = resp['reward']
        done = resp['done']
        rewards.append(reward)
        steps_taken += 1
        print(f"[STEP] step={steps_taken} action={action['type']} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        if action["type"] == "submit_postmortem":
            break

    success = sum(rewards) > 0.5
    score = sum(rewards)
    print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}", flush=True)

if __name__ == "__main__":
    for tid in [1, 2, 3, 4]:
        run_agent(tid)
