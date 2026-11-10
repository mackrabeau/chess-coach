"""Ideas = nonnegative dictionary atoms over feature-difference vectors.

For a position x with best move a* and comparison move b (played move, or 2nd best):
    d(x) = Δf(x, a*) - Δf(x, b) = f(end of best line) - f(end of b's line)
Each feature is divided by its population std of Δ, the vector is normalized to unit
length (direction = the idea, norm = how big the difference is), and split into
[max(u, 0), max(-u, 0)] so atoms read as "the best line has more X / less Y".
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

from .handcrafted import readable


def delta_scale(*deltas: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Per-feature std of Δ over all valid rows (pooled over best/played/alt lines)."""
    stacked = np.concatenate([d[np.isfinite(d).all(1)] for d in deltas])
    return np.maximum(stacked.std(0), floor).astype(np.float32)


def make_inputs(d: np.ndarray, scale: np.ndarray, min_norm: float):
    """Return (X, keep_mask, norms). Rows with NaN or tiny norm are dropped."""
    u = d / scale
    finite = np.isfinite(u).all(1)
    norms = np.where(finite, np.linalg.norm(np.nan_to_num(u), axis=1), 0.0)
    keep = finite & (norms >= min_norm)
    u = u[keep] / norms[keep, None]
    X = np.hstack([np.maximum(u, 0), np.maximum(-u, 0)]).astype(np.float64)
    return X, keep, norms


def fit_nmf(X: np.ndarray, k: int, seed: int, alpha_w: float = 0.0) -> NMF:
    m = NMF(n_components=k, init="nndsvda", max_iter=1000, random_state=seed,
            alpha_W=alpha_w, l1_ratio=1.0)
    m.fit(X)
    return m


def match_atoms(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each row of A, cosine similarity to its Hungarian-matched row of B."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = An @ Bn.T
    r, c = linear_sum_assignment(-S)
    out = np.zeros(len(A))
    out[r] = S[r, c]
    return out


def split_by_game(game_ids: np.ndarray, seed: int) -> np.ndarray:
    """Boolean mask: True = half A. Split by game so positions from one game stay together."""
    games = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    half_a = set(rng.permutation(games)[: len(games) // 2])
    return np.array([g in half_a for g in game_ids])


def shares(W: np.ndarray) -> np.ndarray:
    return W / (W.sum(1, keepdims=True) + 1e-12)


@dataclass
class IdeaModel:
    nmf: NMF
    scale: np.ndarray
    feature_names: list[str]
    min_norm: float
    stability: np.ndarray            # per atom: min(cos to half-A match, cos to half-B match)
    stable_cos: float
    meta: dict = field(default_factory=dict)

    @property
    def components(self) -> np.ndarray:
        return self.nmf.components_

    @property
    def k(self) -> int:
        return self.components.shape[0]

    def encode(self, d: np.ndarray):
        X, keep, norms = make_inputs(d, self.scale, self.min_norm)
        W = self.nmf.transform(X) if len(X) else np.zeros((0, self.k))
        return W, keep, norms

    def describe(self, atom: int, top: int = 6) -> list[tuple[str, str, float]]:
        """[(arrow, readable feature, weight)], strongest first."""
        comp = self.components[atom]
        comp = comp / (comp.sum() + 1e-12)
        D = len(self.feature_names)
        out = []
        for i in np.argsort(-comp)[:top]:
            if comp[i] < 0.03:
                break
            arrow = "↑" if i < D else "↓"
            out.append((arrow, readable(self.feature_names[i % D]), float(comp[i])))
        return out

    def is_stable(self, atom: int) -> bool:
        return bool(self.stability[atom] >= self.stable_cos)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "IdeaModel":
        with open(path, "rb") as f:
            return pickle.load(f)
