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
def step(task_id: int, action: Action):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Task not initialized. Call /reset first.")
    obs, reward, done, info = envs[task_id].step(action)
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
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
