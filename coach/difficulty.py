"""Maia-based difficulty per position, read from the 09b curves.

09b stores, per position, Maia's P(engine best move) and P(played move) at every self-Elo
on maia.grid. Here we read each curve at the mover's own Maia-scale Elo:
  peers : their Lichess rating
  you   : your chess.com rating r at the time of the game, mapped onto Maia's scale as
          r + player_offset + player_offset_slope * (r - 1800)
          (the chess.com/Lichess gap isn't the same at 1000 as at 1900; 09b fits both)
p_best_own = how often a player at that level finds the engine's best move here, per Maia.
"""
from __future__ import annotations

import numpy as np

from .config import data_dir
from .parts import read_parts


def grid_of(mcfg: dict) -> np.ndarray:
    lo, hi, step = mcfg.get("grid", [1100, 2500, 100])
    return np.arange(lo, hi + 1, step)


def maia_dir(cfg: dict, which: str):
    """09b output folder for the configured grid (a new grid starts a fresh folder)."""
    lo, hi, step = cfg["maia"].get("grid", [1100, 2500, 100])
    return data_dir(cfg, which, f"maia_{lo}_{hi}_{step}")


def player_maia_elo(rating, mcfg: dict, df=None):
    """chess.com rating(s) -> Maia-scale Elo, from the fitted offset (+ slope)."""
    r = np.asarray(rating, float)
    off = mcfg.get("player_offset")
    if off is None:
        off = mcfg["player_elo"] - (np.median(r) if df is None else df.mover_rating.median())
    return r + float(off) + float(mcfg.get("player_offset_slope") or 0) * (r - 1800)


def at_elo(curves: np.ndarray, elo, grid: np.ndarray) -> np.ndarray:
    """Read each row's grid curve at its own Elo (linear interpolation, clamped to the grid)."""
    x = np.clip((np.asarray(elo, float) - grid[0]) / (grid[1] - grid[0]), 0, len(grid) - 1)
    i = np.minimum(np.floor(x).astype(int), len(grid) - 2)
    w = x - i
    r = np.arange(len(curves))
    return (1 - w) * curves[r, i] + w * curves[r, i + 1]


def own_elo(df, which: str, mcfg: dict) -> np.ndarray:
    if which == "population":
        return df.mover_rating.to_numpy(float)
    return player_maia_elo(df.mover_rating.to_numpy(float), mcfg, df)


def p_best_own(cfg: dict, df, which: str):
    """P(best move) at the mover's own Maia Elo, aligned to df rows (NaN where 09b hasn't run).
    Returns None if there are no 09b results for this set."""
    q = read_parts(maia_dir(cfg, which))
    if q is None:
        return None
    mcfg = cfg["maia"]
    grid = grid_of(mcfg)
    curves = dict(zip(q.pos_id, q.p_best))
    have = df.pos_id.isin(curves.keys()).to_numpy()
    out = np.full(len(df), np.nan)
    if have.any():
        C = np.stack([np.asarray(curves[p], float) for p in df.pos_id[have]])
        out[have] = at_elo(C, own_elo(df[have], which, mcfg), grid)
    return out
