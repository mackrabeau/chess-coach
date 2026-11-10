"""Config loading and path helpers."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Project root = parent of the configs/ folder.
    cfg["_root"] = path.parent.parent
    return cfg


def resolve(cfg: dict, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else cfg["_root"] / p


def data_dir(cfg: dict, *parts: str) -> Path:
    d = resolve(cfg, cfg["paths"]["data_dir"]).joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
