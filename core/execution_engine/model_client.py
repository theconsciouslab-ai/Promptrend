# core/execution_engine/model_client.py
from dataclasses import dataclass
from typing import Any

@dataclass
class ModelClient:
    name: str
    client: Any
    timeout: float = 60.0
