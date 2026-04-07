import os
import requests
import json
from openai import OpenAI

# LLM Config
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN", ""))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

TOOLS = [
    {"type": "function", "function": {"name": "read_dashboard", "description": "Read the top-level metrics dashboard.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "sample_stream", "description": "Pull raw events from the stream.", "parameters": {"type": "object", "properties": {"sample_size": {"type": "integer", "description": "Number of events to sample (1-100)"}}, "required": ["sample_size"]}}},
    {"type": "function", "function": {"name": "inspect_lineage", "description": "Inspect the SQL definition of a model.", "parameters": {"type": "object", "properties": {"model_id": {"type": "string"}}, "required": ["model_id"]}}},
    {"type": "function", "function": {"name": "query_system_logs", "description": "Query system metadata or maintenance logs.", "parameters": {"type": "object", "properties": {"log_name": {"type": "string"}}, "required": ["log_name"]}}},
    {"type": "function", "function": {"name": "query_provider_contract", "description": "Check the SLA contract for a specific provider.", "parameters": {"type": "object", "properties": {"provider_id": {"type": "string"}}, "required": ["provider_id"]}}},
    {"type": "function", "function": {"name": "simulate_config_change", "description": "Simulate what metrics would be if a config parameter changed.", "parameters": {"type": "object", "properties": {"config_param": {"type": "string"}, "value": {"type": "integer"}}, "required": ["config_param", "value"]}}},
    {"type": "function", "function": {"name": "submit_theory", "description": "Submit a diagnosis for the root cause. This must be done before the postmortem.", "parameters": {"type": "object", "properties": {"cause": {"type": "string", "enum": ["latency_spike", "join_failure", "duplicate_flood", "schema_drift", "out_of_order", "expected_maintenance"]}, "evidence": {"type": "array", "items": {"type": "string"}}}, "required": ["cause", "evidence"]}}},
    {"type": "function", "function": {"name": "submit_postmortem", "description": "Conclude the incident with a final postmortem. This permanently ends the episode.", "parameters": {"type": "object", "properties": {"timeline": {"type": "array", "items": {"type": "object", "properties": {"tick": {"type": "integer"}, "description": {"type": "string"}}}}, "impact_duration_ticks": {"type": "integer"}, "prevention_action": {"type": "string", "enum": ["increase_timeout", "add_index", "block_duplicates", "update_schema", "scheduled_maintenance_sync"]}}, "required": ["timeline", "impact_duration_ticks", "prevention_action"]}}}
]

def reset_env(task_id: int):
    return requests.post(f"{ENV_URL}/reset?task_id={task_id}", json={}).json()

def step_env(task_id: int, action: dict):
    return requests.post(f"{ENV_URL}/step?task_id={task_id}", json=action).json()

def run_agent(task_id: int):
    task_name = f"task-{task_id}"
    print(f"[START] task={task_name} env=causal-stream-v3 model={MODEL_NAME}-LLM", flush=True)
    obs = reset_env(task_id)
    
    if not client:
        print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
        return

    messages = [
        {"role": "system", "content": "You are a Senior SRE agent diagnosing data pipeline issues. You must use tools to investigate state (e.g. read_dashboard). Then use submit_theory to log your hypothesis, followed by submit_postmortem to end the episode."},
        {"role": "user", "content": f"Please diagnose the issue for {task_name}. You start with no context, call read_dashboard immediately."}
    ]
    
    steps_taken = 0
    done = False
    rewards = []
    
    try:
        while not done and steps_taken < 15:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
            msg = response.choices[0].message
            # Append assistant message properly
            assistant_msg = {"role": "assistant"}
            if msg.content: assistant_msg["content"] = msg.content
            if msg.tool_calls: assistant_msg["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in msg.tool_calls]
            messages.append(assistant_msg)
            
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    
                    action_payload = args.copy()
                    action_payload["type"] = func_name
                    
                    res = step_env(task_id, action_payload)
                    rewards.append(res['reward'])
                    done = res['done']
                    steps_taken += 1
                    
                    print(f"[STEP] step={steps_taken} action={func_name} reward={res['reward']:.2f} done={str(done).lower()} error=null", flush=True)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": func_name,
                        "content": json.dumps(res['observation'])
                    })
                    
                    if done:
                        break
            else:
                messages.append({"role": "user", "content": "Please invoke a tool to continue your investigation."})
                steps_taken += 1
    except Exception as e:
        print(f"Agent failed with error: {e}")

    success = sum(rewards) > 0.5
    score = sum(rewards)
    print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}", flush=True)

if __name__ == "__main__":
    for tid in [1, 2, 3, 4]:
        run_agent(tid)

