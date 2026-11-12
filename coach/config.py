"""Config loading and path helpers.

Data layout (several players side by side; nothing is deleted when you switch):
  data/players/<username>/player/     that player's games, engine results, reports (data_dir(cfg, "player"))
  data/players/<username>/openings/   their opening tree, book, opening engine cache
  data/players/<username>/findings/   their findings (the report's input)
  data/players/<username>/imported/   their imported PGN games
  data/population_<lo>_<hi>/          peers of one Lichess rating band (shared by every player using it)
  data/population_<lo>_<hi>/ideas/    idea model learned on those peers
  data/band_<lo>/                     multi-band samples (shared)
  data/compare/<a>_vs_<b>/            head-to-head reports
Configs can inherit: `extends: v0.yaml` loads that file first and overrides it key by key, so a
second player's config only needs the player block (and the peer band).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

PLAYER_SCOPED = ("player", "openings", "findings", "imported")
POPULATION_SCOPED = ("population", "ideas")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "extends" in cfg:
        base = load_config(path.parent / cfg.pop("extends"))
        base.pop("_maia_fitted", None)
        cfg = _merge({k: v for k, v in base.items() if not k.startswith("_")}, cfg)
    # Project root = parent of the configs/ folder.
    cfg["_root"] = path.parent.parent
    cfg["_config_path"] = str(path)
    _migrate_legacy(cfg)
    _use_fitted_maia(cfg)
    return cfg


def resolve(cfg: dict, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else cfg["_root"] / p


def data_root(cfg: dict) -> Path:
    return resolve(cfg, cfg["paths"]["data_dir"])


def player_root(cfg: dict) -> Path:
    return data_root(cfg) / "players" / cfg["player"]["username"].lower()


def population_name(cfg: dict) -> str:
    p = cfg["population"]
    return f"population_{p['rating_center'] - p['rating_halfwidth']}_{p['rating_center'] + p['rating_halfwidth']}"


def scope_path(cfg: dict, first: str) -> Path:
    """Where the top-level data folder `first` lives for this config (no mkdir)."""
    if first in PLAYER_SCOPED:
        return player_root(cfg) / first
    if first == "population":
        return data_root(cfg) / population_name(cfg)
    if first == "ideas":
        return data_root(cfg) / population_name(cfg) / "ideas"
    return data_root(cfg) / first


def data_dir(cfg: dict, *parts: str) -> Path:
    d = scope_path(cfg, parts[0]).joinpath(*parts[1:]) if parts else data_root(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _archives_user(arch: Path) -> set[str]:
    for p in sorted(arch.glob("*.json"))[:3]:
        try:
            games = json.loads(p.read_text(encoding="utf-8")).get("games", [])
        except Exception:
            continue
        names = [g[c]["username"].lower() for g in games[:50] for c in ("white", "black") if c in g]
        if names:
            return {max(set(names), key=names.count)}
    return set()


def _migrate_legacy(cfg: dict) -> None:
    """One-time move from the old single-player layout (data/player, data/population, ...)."""
    root = data_root(cfg)
    legacy = root / "player"
    if legacy.is_dir() and not player_root(cfg).exists():
        if cfg["player"]["username"].lower() in _archives_user(legacy / "archives"):
            player_root(cfg).mkdir(parents=True, exist_ok=True)
            for name in PLAYER_SCOPED:
                if (root / name).is_dir():
                    shutil.move(str(root / name), str(player_root(cfg) / name))
            if (root / "refine_state.json").exists():
                shutil.move(str(root / "refine_state.json"), str(player_root(cfg) / "refine_state.json"))
            print(f"moved the existing player data to {player_root(cfg)}")
    pop_old, pop_new = root / "population", root / population_name(cfg)
    if pop_old.is_dir() and not pop_new.exists() and (pop_old / "positions_engine.parquet").exists():
        try:
            import pandas as pd
            r = pd.read_parquet(pop_old / "positions_engine.parquet", columns=["mover_rating"]).mover_rating
            p = cfg["population"]
            ok = r.between(p["rating_center"] - p["rating_halfwidth"], p["rating_center"] + p["rating_halfwidth"]).mean() > 0.9
        except Exception:
            ok = False
        if ok:
            shutil.move(str(pop_old), str(pop_new))
            if (root / "ideas").is_dir() and not (pop_new / "ideas").exists():
                shutil.move(str(root / "ideas"), str(pop_new / "ideas"))
            print(f"moved the existing peer data to {pop_new}")


def _use_fitted_maia(cfg: dict) -> None:
    """09b writes the fitted chess.com -> Maia offset per player; it overrides the config values."""
    f = player_root(cfg) / "player" / "maia_offset.json"
    if f.exists():
        try:
            fit = json.loads(f.read_text(encoding="utf-8"))
            cfg.setdefault("maia", {})["player_offset"] = fit["offset"]
            cfg["maia"]["player_offset_slope"] = fit.get("slope", 0.0)
            cfg["_maia_fitted"] = True
        except Exception:
            pass
