
import json
import streamlit as st
import numpy as np
from llm_rl_recommender.registry import ModelRegistry
from llm_rl_recommender.contexts import build_context_vector
from llm_rl_recommender.bandits import LinUCB
from llm_rl_recommender.persist import load_json, save_json

st.set_page_config(page_title="LLM RL Recommender", page_icon="🤖")


POLICY_PATH = "artifacts/policy.json"
MODELS_PATH = "data/models.yaml"

registry = ModelRegistry(MODELS_PATH)
state = load_json(POLICY_PATH)
bandit = LinUCB.load(state)

st.title("LLM Model Recommender (Contextual Bandits)")
st.caption("RL-driven agent that recommends the best LLM given your task/dataset constraints.")

with st.expander("Model Catalog"):
    st.json({"models": [m.__dict__ for m in registry.models]})

st.subheader("1) Describe your context")


def pick(label, options, default):
    return st.selectbox(label, options, index=options.index(default))

ctx = {}
ctx["task"] = pick("Task", ["chat","summarization","code","qa","classification"], "chat")
ctx["domain"] = pick("Domain", ["general","legal","medical","finance"], "general")
ctx["dataset_size"] = pick("Dataset Size", ["tiny","small","medium","large"], "small")
ctx["latency_budget"] = pick("Latency Budget", ["very_low","low","medium","high"], "medium")
ctx["cost_budget"] = pick("Cost Budget", ["very_low","low","medium","high"], "medium")
ctx["multilingual"] = st.checkbox("Multilingual", value=False)
ctx["needs_coding"] = st.checkbox("Needs Coding", value=False)
ctx["needs_reasoning"] = st.checkbox("Needs Reasoning", value=True)
ctx["safety_sensitive"] = st.checkbox("Safety Sensitive", value=False)

st.subheader("2) Recommendations")
x = build_context_vector(ctx)

# scoring
import numpy as np
d = bandit.config.d
xv = x.reshape(-1,1)
rows = []
for a in range(bandit.config.n_actions):
    A_inv = np.linalg.inv(bandit.As[a])
    theta = A_inv @ bandit.bs[a].reshape(d,1)
    mean = float(theta.T @ xv)
    bonus = bandit.config.alpha * float(np.sqrt(xv.T @ A_inv @ xv))
    rows.append({"model_id": registry.get(a).id, "score": mean + bonus, "mean": mean, "bonus": bonus})
rows = sorted(rows, key=lambda r: r["score"], reverse=True)

st.table(rows[:3])

st.subheader("3) Feedback (Online Learning)")
chosen = st.selectbox("Which model did you deploy?", [r["model_id"] for r in rows])
reward = st.slider("Observed reward (0 = bad, 1 = good)", 0.0, 1.0, 0.7, 0.05)
if st.button("Submit feedback & update policy"):
    idx = registry.index_of(chosen)
    bandit.update(x, idx, reward)
    save_json(POLICY_PATH, bandit.export())
    st.success("Policy updated and saved.")
