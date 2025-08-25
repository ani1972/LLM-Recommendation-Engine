
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from llm_rl_recommender.bandits import LinUCB
from llm_rl_recommender.registry import ModelRegistry
from llm_rl_recommender.contexts import build_context_vector
from llm_rl_recommender.persist import load_json, save_json

POLICY_PATH = "artifacts/policy.json"
MODELS_PATH = "data/models.yaml"

app = FastAPI(title="LLM RL Recommender API", version="0.1.0")
registry = ModelRegistry(MODELS_PATH)
bandit = LinUCB.load(load_json(POLICY_PATH))

class Context(BaseModel):
    task: str = "chat"
    domain: str = "general"
    dataset_size: str = "small"
    latency_budget: str = "medium"
    cost_budget: str = "medium"
    multilingual: bool = False
    needs_coding: bool = False
    needs_reasoning: bool = True
    safety_sensitive: bool = False

class RecRequest(BaseModel):
    context: Context
    top_k: int = 3

class FeedbackRequest(BaseModel):
    context: Context
    chosen_model_id: str
    reward: float

def score_all(x: np.ndarray):
    d = bandit.config.d
    x = x.reshape(-1,1)
    scores = []
    for a in range(bandit.config.n_actions):
        A_inv = np.linalg.inv(bandit.As[a])
        theta = A_inv @ bandit.bs[a].reshape(d,1)
        mean = float(theta.T @ x)
        bonus = bandit.config.alpha * float(np.sqrt(x.T @ A_inv @ x))
        scores.append({"model_id": registry.get(a).id, "score": mean + bonus, "mean": mean, "bonus": bonus})
    scores.sort(key=lambda s: s["score"], reverse=True)
    return scores

@app.post("/recommend")
def recommend(req: RecRequest):
    x = build_context_vector(req.context.dict())
    scores = score_all(x)
    return {"context": req.context, "recommendations": scores[:req.top_k]}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    x = build_context_vector(req.context.dict())
    idx = registry.index_of(req.chosen_model_id)
    bandit.update(x, idx, req.reward)
    # Persist online updates
    save_json(POLICY_PATH, bandit.export())
    return {"status": "ok"}
