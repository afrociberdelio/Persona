"""Carrega config/default.yaml, com override opcional de config/local.yaml.

local.yaml e gitignored e serve para diferencas de maquina (ex: dev num
notebook com GPU menor rodando um LLM reduzido, enquanto default.yaml
reflete o alvo de producao).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "default.yaml"
_LOCAL_CONFIG_PATH = _PROJECT_ROOT / "config" / "local.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    """Acesso somente-leitura, por atributo, a secoes da config (ex: cfg.llm.model_name)."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        for key, value in data.items():
            setattr(self, key, _Section(value) if isinstance(value, dict) else value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def raw(self) -> dict[str, Any]:
        return self._data


class _Section:
    def __init__(self, data: dict[str, Any]):
        for key, value in data.items():
            setattr(self, key, _Section(value) if isinstance(value, dict) else value)
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        return f"_Section({self._data!r})"


def load_config(
    default_path: Path = _DEFAULT_CONFIG_PATH,
    local_path: Path = _LOCAL_CONFIG_PATH,
) -> Config:
    with open(default_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)

    return Config(data)


def project_root() -> Path:
    return _PROJECT_ROOT
