# src/agent_config.py
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import yaml


DEFAULT_WRITER_CONFIG_PATH = os.getenv("WRITER_CONFIG_PATH", "configs/writer.yaml")


@lru_cache(maxsize=8)
def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config shape in {path}: expected dict at root")
    return data


def get_writer_config(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or DEFAULT_WRITER_CONFIG_PATH
    return load_yaml_config(path)
