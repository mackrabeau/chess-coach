"""Peer rating bands (shared by 17_multiband and 24_weakness_model).

Every band is a Lichess rating range with its own analysed positions: the player's own peer set
(`population`), a full peer set mined for another player (`population_<lo>_<hi>`), or a small
multi-band sample (`band_<lo>`, mined by 17).
"""
from __future__ import annotations

from .config import data_root

MB = {"bands": [[1600, 1800], [1800, 2000], [2200, 2400], [2400, 3000]], "per_game": 3, "target": 4000,
      "boot": 300, "min_moves": 150}


def band_name(lo):
    return f"band_{lo}"


def own_range(cfg):
    pc = cfg["population"]
    return pc["rating_center"] - pc["rating_halfwidth"], pc["rating_center"] + pc["rating_halfwidth"]


def settings(cfg) -> dict:
    return {**MB, **cfg.get("multiband", {})}


def resolve_sets(cfg, mb):
    """[(dataset name, (lo, hi))] for every band + your own peer band. A range that has a full peer set
    (data/population_<lo>_<hi>, e.g. another player's peers) uses it instead of mining a band sample."""
    ranges = {tuple(b) for b in mb["bands"]} | {own_range(cfg)}
    out = []
    for lo, hi in sorted(ranges):
        pop = f"population_{lo}_{hi}"
        if (lo, hi) == own_range(cfg) or any((data_root(cfg) / pop / "engine").glob("part-*.parquet")):
            out.append(("population" if (lo, hi) == own_range(cfg) else pop, (lo, hi)))
        else:
            out.append((band_name(lo), (lo, hi)))
    return out


def band_ranges(cfg, mb):
    """Bands that need their own mined sample (no full peer set covers them)."""
    return [r for n, r in resolve_sets(cfg, mb) if n.startswith("band_")]
