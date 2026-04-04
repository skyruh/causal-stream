---
title: CausalStream
emoji: 🌊
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# CausalStream: High-Fidelity SRE Logic Environment

CausalStream is a specialized Reinforcement Learning (RL) environment designed to benchmark and train agents in **temporal causal reasoning** and **resource allocation under uncertainty**. It simulates a complex streaming data infrastructure (Kafka/Flink pattern) where agents must diagnose production incidents.

## �️ Core Engine Architecture

### 1. The Stochastic World Clock (Ticked Execution)
Unlike traditional "static" debugging environments, CausalStream operates on a discrete **Tick** system. 
- **Temporal Drift**: Every action that consumes compute or network resources (e.g., sampling a raw stream) advances the world clock.
- **Event Physics**: Events are generated with a stochastic latency model: `arrival_time = event_time + base_latency + jitter`. 
- **Incident Injection**: Incidents are not just boolean flags; they modify the distribution of the stream. For example, a `LATENCY_SPIKE` incident increases the `base_latency` variance, causing "late arrival" data loss in aggregation windows.

### 2. Tiered Observation Model
Information is not free. Agents interact with a hierarchical observation space:
- **Dashboard (L0)**: Free to read. Provides aggregated metrics (Revenue, Error Rate). High signal but low granularity.
- **Stream Samples (L1)**: Costs **1 Tick**. Provides raw JSON event snippets. Necessary for detecting jitter and schema mismatches.
- **Lineage Graph (L2)**: Costs **1 Tick**. Provides the SQL logic and dependency chain of the data pipeline.

### 3. Counterfactual Query Engine (Causal Discovery)
To prove a hypothesis, agents can execute **Counterfactual Actions**. 
- **Action**: `ask_counterfactual(window_offset=X)`
- **Mechanism**: The engine forks the internal state and simulates what the metrics *would have been* if a variable (like the late-arrival window) was modified. 
- **Significance**: This forces agents to move beyond pattern matching and perform true scientific experimentation.

## ⚖️ Deterministic Grading & F1 Evidence

CausalStream solves the "Subjective Reasoning" problem by requiring agents to submit a structured **Theory**.
- **The Theory**: Consists of a `RootCauseEnum` (e.g., `OUT_OF_ORDER`) and a `List[Evidence]`.
- **F1 Scoring**: The grader calculates the **Precision and Recall** of the submitted evidence tokens. 
  - To get a 1.0 (Full Credit), an agent must identify the correct cause AND provide exactly the evidence tokens (specific IDs or timestamps) that prove that cause without including "noise" tokens.

## 🚀 Technical Setup

### Local Development
```bash
pip install -r requirements.txt
python server.py --port 7860
```

### Docker Deployment
The environment is optimized for **2 vCPU / 8GB RAM** constraints and exposes a REST API via FastAPI.
```bash
docker build -t causal-stream .
docker run -p 7860:7860 causal-stream
```

---
*Developed for the Meta PyTorch OpenEnv Hackathon 2026. Focus: Causal Intelligence.*
