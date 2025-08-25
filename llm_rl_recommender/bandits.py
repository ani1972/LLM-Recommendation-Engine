
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class LinUCBConfig:
    alpha: float = 1.0  # exploration
    d: int = 16         # feature dimension
    n_actions: int = 6  # number of models/actions

class LinUCB:
    """LinUCB contextual bandit with per-action linear models.

    Each action a has A_a (d x d) and b_a (d x 1):
      A_a = D_a^T D_a + I_d
      b_a = sum_{t: a_t=a} r_t x_t
    Estimate: theta_a = A_a^{-1} b_a
    UCB score: p_a = theta_a^T x_t + alpha * sqrt(x_t^T A_a^{-1} x_t)
    """
    def __init__(self, config: LinUCBConfig):
        self.config = config
        d, k = config.d, config.n_actions
        self.As = np.array([np.eye(d) for _ in range(k)])   # (k, d, d)
        self.bs = np.zeros((k, d))                           # (k, d)

    def select(self, x: np.ndarray) -> int:
        x = x.reshape(-1, 1)  # (d,1)
        d = self.config.d
        p = np.zeros(self.config.n_actions)
        for a in range(self.config.n_actions):
            A_inv = np.linalg.inv(self.As[a])
            theta = A_inv @ self.bs[a].reshape(d, 1)
            mean = float(theta.T @ x)
            bonus = self.config.alpha * float(np.sqrt(x.T @ A_inv @ x))
            p[a] = mean + bonus
        return int(np.argmax(p))

    def update(self, x: np.ndarray, a: int, r: float):
        x = x.reshape(-1, 1)
        self.As[a] += x @ x.T
        self.bs[a] += (r * x.flatten())

    def export(self) -> dict:
        return {
            "config": self.config.__dict__,
            "As": self.As.tolist(),
            "bs": self.bs.tolist(),
        }

    @classmethod
    def load(cls, state: dict) -> "LinUCB":
        cfg = LinUCBConfig(**state["config"])
        inst = cls(cfg)
        inst.As = np.array(state["As"], dtype=float)
        inst.bs = np.array(state["bs"], dtype=float)
        return inst
