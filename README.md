# CausalStream v3: The SRE Oracle 🌊

**A meta-hackathon submission for the OpenEnv Hackathon.**

`CausalStream` is a stochastic, high-fidelity RL environment designed to evaluate an agent's ability to perform **Causal Reasoning and Resource Allocation** under uncertainty.

## 🏆 Scoring Highlights
- **Real-World Utility**: Models the #1 production nightmare of modern data companies (Streaming Incident Response).
- **Novel Mechanic**: Introduced the **Counterfactual Query Engine** and **Temporal Cost Model**.
- **Deterministic Grading**: Uses schema-bound hypothesis enums and F1 Evidence Scoring.

## 🚀 Setup
1. **Local**: `pip install -r requirements.txt && python server.py`
2. **Docker**: `docker build -t causal-stream . && docker run -p 7860:7860 causal-stream`

## 🧠 Evaluation
The environment contains 3 tasks ranging from "Easy" (Structural) to "Elite Hard" (Temporal/Multi-variable). 
- Baseline Agent: `python inference.py` (Solves Task 1 & 2).

---
*Tags: [openenv, sre, causal-reasoning]*
