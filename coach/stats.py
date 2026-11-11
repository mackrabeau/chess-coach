"""Shared statistics helpers (game-level bootstrap)."""
from __future__ import annotations

import numpy as np


def game_bootstrap(df, rows, values: np.ndarray, n_boot: int, seed: int = 0) -> np.ndarray:
    """Resample whole games. values: (len(rows), m) per-move contributions.
    Returns (n_boot, m) of 100 * sum(values) / moves."""
    games, g_idx = np.unique(df.game_id.to_numpy(), return_inverse=True)
    n_moves = np.bincount(g_idx, minlength=len(games))
    contrib = np.zeros((len(games), values.shape[1]))
    if len(rows):
        np.add.at(contrib, g_idx[rows], values)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, values.shape[1]))
    for b in range(n_boot):
        pick = rng.integers(0, len(games), len(games))
        boots[b] = 100 * contrib[pick].sum(0) / max(n_moves[pick].sum(), 1)
    return boots


def ci(boots: np.ndarray):
    return np.percentile(boots, 5, axis=0), np.percentile(boots, 95, axis=0)
