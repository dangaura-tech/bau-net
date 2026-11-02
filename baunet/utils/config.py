"""YAML configuration with ``_base_`` inheritance and command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _deep_update(base: Dict, new: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _deep_update(load_config(str(path.parent / base)), cfg)
    return cfg


def merge_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``key.sub=value`` overrides, parsing values as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def _parse_value(raw: str) -> Any:
    """Parse an override value as YAML, also accepting floats like ``5e-4`` (YAML 1.1 reads them as strings)."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value
