
from __future__ import annotations
import numpy as np
from typing import Dict
from .registry import ModelRegistry

class HeuristicRewarder:
    """A simple synthetic reward function for offline simulation/training.

    It encodes assumptions like:
      - High cost_budget + reasoning -> high-end closed models score higher.
      - Low cost/low latency -> small open models score higher.
      - Multilingual or coding flips weight to specific models.
    Replace with real evals/metrics in production.
    """
    def __init__(self, registry: ModelRegistry, noise: float = 0.05):
        self.registry = registry
        self.noise = noise

    def expected_reward(self, ctx: Dict, model_idx: int) -> float:
        model = self.registry.get(model_idx)
        score = 0.0

        # Cost fit
        cb = ctx.get("cost_budget", "medium")
        if cb in ["high"] and not model.open_source:
            score += 0.6
        if cb in ["very_low", "low"] and model.open_source:
            score += 0.6

        # Latency fit
        lb = ctx.get("latency_budget", "medium")
        tiers = {"very_low": 0.6, "low": 0.4, "medium": 0.2, "high": 0.0}
        score += tiers.get(lb, 0.2) if model.latency_tier in ["very_low", "low"] else 0.1

        # Task/domain preferences
        task = ctx.get("task", "chat")
        domain = ctx.get("domain", "general")
        if task == "code" and ("coding" in model.strengths or "speed" in model.strengths):
            score += 0.5
        if task == "summarization" and ("general" in model.strengths or "reasoning" in model.strengths):
            score += 0.3
        if task == "qa" and ("reasoning" in model.strengths):
            score += 0.4

        if domain in ["legal", "medical", "finance"] and ("reasoning" in model.strengths or not model.open_source):
            score += 0.3

        # Flags
        if ctx.get("multilingual") and ("multilingual" in model.strengths):
            score += 0.3
        if ctx.get("needs_reasoning") and ("reasoning" in model.strengths):
            score += 0.4
        if ctx.get("needs_coding") and ("coding" in model.strengths):
            score += 0.4
        if ctx.get("safety_sensitive") and ("safety" in model.strengths):
            score += 0.3

        # Slight preference for fine-tuneable open models when dataset is large
        if ctx.get("dataset_size") in ["medium", "large"] and ("fine_tuning" in model.strengths or model.open_source):
            score += 0.2

        # Normalize-ish and add noise
        score = np.clip(score, 0.0, 2.5)
        score += np.random.normal(0, self.noise)
        return float(score)

    def sample_reward(self, ctx: Dict, model_idx: int) -> float:
        return self.expected_reward(ctx, model_idx)
