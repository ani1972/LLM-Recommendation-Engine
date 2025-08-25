
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import yaml
import os

@dataclass
class ModelCard:
    id: str
    provider: str
    open_source: bool
    cost_tier: str
    latency_tier: str
    strengths: list

class ModelRegistry:
    def __init__(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.models = [ModelCard(**m) for m in data["models"]]

    def list_ids(self) -> List[str]:
        return [m.id for m in self.models]

    def __len__(self):
        return len(self.models)

    def get(self, idx: int) -> ModelCard:
        return self.models[idx]

    def index_of(self, model_id: str) -> int:
        for i, m in enumerate(self.models):
            if m.id == model_id:
                return i
        raise KeyError(model_id)

    def to_table(self) -> List[Dict[str, Any]]:
        return [m.__dict__ for m in self.models]
