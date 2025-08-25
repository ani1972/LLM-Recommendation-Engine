
import argparse, json, os, random
import numpy as np
from llm_rl_recommender.bandits import LinUCB, LinUCBConfig
from llm_rl_recommender.registry import ModelRegistry
from llm_rl_recommender.contexts import build_context_vector
from llm_rl_recommender.rewarders import HeuristicRewarder
from llm_rl_recommender.persist import save_json

def random_ctx():
    return {
        "task": random.choice(["chat","summarization","code","qa","classification"]),
        "domain": random.choice(["general","legal","medical","finance"]),
        "dataset_size": random.choice(["tiny","small","medium","large"]),
        "latency_budget": random.choice(["very_low","low","medium","high"]),
        "cost_budget": random.choice(["very_low","low","medium","high"]),
        "multilingual": random.choice([True, False]),
        "needs_coding": random.choice([True, False]),
        "needs_reasoning": random.choice([True, False]),
        "safety_sensitive": random.choice([True, False]),
    }

def main(args):
    registry = ModelRegistry(args.models_yaml)
    d = len(build_context_vector(random_ctx()))
    cfg = LinUCBConfig(alpha=args.alpha, d=d, n_actions=len(registry))
    bandit = LinUCB(cfg)
    rewarder = HeuristicRewarder(registry, noise=args.noise)

    for t in range(args.steps):
        ctx = random_ctx()
        x = build_context_vector(ctx)
        a = bandit.select(x)
        r = rewarder.sample_reward(ctx, a)
        bandit.update(x, a, r)
        if (t+1) % 1000 == 0:
            print(f"step {t+1}: updated policy")

    state = bandit.export()
    save_json(args.out, state)
    print(f"Saved trained policy to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models-yaml", default="data/models.yaml")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--out", default="artifacts/policy.json")
    args = p.parse_args()
    os.makedirs("artifacts", exist_ok=True)
    main(args)
