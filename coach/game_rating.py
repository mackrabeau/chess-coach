"""Per-game (and per-phase) rating estimates from the multi-band fits (17).

A game's loss per move in a phase is placed on the line fitted through the rating bands' loss per
move for that phase (Lichess rapid scale). One game is a small sample (a phase is often 10-20 moves),
so treat it as a rough reading, and compare games with each other rather than with your rating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import data_dir

PHASE_CAT = {"all": "all moves", "opening": "phase: opening (moves 10-15)",
             "middlegame": "phase: middlegame", "endgame": "phase: endgame"}


def lines(cfg) -> dict:
    """phase -> (a, s): loss per move = a + s (rating - 2100), fitted through the bands."""
    p = data_dir(cfg, "player") / "multiband.parquet"
    if not p.exists():
        return {}
    mb = pd.read_parquet(p)
    mb = mb[mb.metric == "loss per 100 moves"]
    bcols = sorted(c for c in mb.columns if c.startswith("band_"))
    rcols = sorted(c for c in mb.columns if c.startswith("rating_"))
    out = {}
    for ph, cat in PHASE_CAT.items():
        r = mb[mb.category == cat]
        if not len(r) or not rcols:
            continue
        x = np.array([float(r[c].iloc[0]) for c in rcols]) - 2100
        y = np.array([float(r[c].iloc[0]) for c in bcols])
        s, a = np.polyfit(x, y, 1)
        if s < 0:                                  # stronger players lose less; otherwise no reading
            out[ph] = (a, s)
    return out


def level(loss_per_move: float, fit, lo: float = 1000, hi: float = 2900) -> float:
    if fit is None or not np.isfinite(loss_per_move):
        return np.nan
    a, s = fit
    return float(np.clip(2100 + (loss_per_move - a) / s, lo, hi))
