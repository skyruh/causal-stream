from fastapi import FastAPI, HTTPException
from typing import Optional
from causal_stream_env.env import CausalStreamEnv
from causal_stream_env.models import Action
import uvicorn

app = FastAPI(title="CausalStream OpenEnv Server")
envs = {}

@app.post("/reset")
def reset(task_id: Optional[int] = 1):
    # Validator pings /reset with {} and no params
    tid = task_id or 1
    env = CausalStreamEnv(task_id=tid)
    envs[tid] = env
    return env.reset()

@app.post("/step")
def step(task_id: int, payload: dict):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Task not initialized.")
    
    # Handle the difference between `{"type": ...}` and `{"action": {"type": ...}}`
    raw_action = payload.get("action", payload)
    
    # Parse into Pydantic model natively
    try:
        from pydantic import TypeAdapter
        action_obj = TypeAdapter(Action).validate_python(raw_action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action payload: {e}")

    obs, reward, done, info = envs[task_id].step(action_obj)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state(task_id: int):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Task not initialized.")
    return envs[task_id].get_state()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
