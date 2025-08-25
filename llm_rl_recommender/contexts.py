
from __future__ import annotations
from typing import Dict, List
import numpy as np

# One-hot utilities
def one_hot(value: str, choices: List[str]) -> List[int]:
    return [1 if value == c else 0 for c in choices]

COST_CHOICES = ["very_low", "low", "medium", "high"]
LAT_CHOICES  = ["very_low", "low", "medium", "high"]
TASK_CHOICES = ["chat", "summarization", "code", "qa", "classification"]
DOMAIN_CHOICES = ["general", "legal", "medical", "finance"]
SIZE_CHOICES = ["tiny", "small", "medium", "large"]
BOOL = lambda b: [1] if b else [0]

def build_context_vector(ctx: Dict) -> np.ndarray:
    """Encode a user/application/dataset context into a numeric feature vector.

    Expected keys in ctx:
      - task: str in TASK_CHOICES
      - domain: str in DOMAIN_CHOICES
      - dataset_size: str in SIZE_CHOICES
      - latency_budget: str in LAT_CHOICES
      - cost_budget: str in COST_CHOICES
      - multilingual: bool
      - needs_coding: bool
      - needs_reasoning: bool
      - safety_sensitive: bool
    """
    x = []
    x += one_hot(ctx.get("task", "chat"), TASK_CHOICES)
    x += one_hot(ctx.get("domain", "general"), DOMAIN_CHOICES)
    x += one_hot(ctx.get("dataset_size", "small"), SIZE_CHOICES)
    x += one_hot(ctx.get("latency_budget", "medium"), LAT_CHOICES)
    x += one_hot(ctx.get("cost_budget", "medium"), COST_CHOICES)
    x += BOOL(ctx.get("multilingual", False))
    x += BOOL(ctx.get("needs_coding", False))
    x += BOOL(ctx.get("needs_reasoning", True))
    x += BOOL(ctx.get("safety_sensitive", False))
    return np.array(x, dtype=float)
