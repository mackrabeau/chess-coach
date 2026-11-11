"""Hand-defined concept groups over the handcrafted features (replaces NMF for v0 features).

Why: on real data, NMF on these 58 features found about as many split-half-stable atoms on
column-shuffled null data as on real data, so the "ideas" were feature groups, not
discovered structure. Here the groups are explicit. Each concept is a feature group for
one side ("your king safety" and "opponent's king safety" are different ideas).

For a difference vector d = f(end of best line) - f(end of other line), scaled by the
population std of Δ (u = d / scale), concept c's share is

    s_c = ||u_c||^2 / ||u||^2

so shares sum to 1 per position. Same interface as IdeaModel (k, encode, describe,
is_stable, meta), so 07 and 08 work with either.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .handcrafted import SIDE_FEATURES, readable

GROUPS: dict[str, list[str]] = {
    "pawn structure": ["pawns", "doubled_pawns", "isolated_pawns", "backward_pawns", "pawn_islands"],
    "passed pawns": ["passed_pawns", "passed_pawn_advance"],
    "king safety": ["king_shield", "king_zone_attacks", "king_open_files", "king_center_dist"],
    "piece activity": ["knight_mobility", "bishop_mobility", "rook_mobility", "queen_mobility",
                       "rooks_open_file", "rooks_halfopen_file", "rooks_7th", "outposts"],
    "trades": ["knights", "bishops", "rooks", "queens", "bishop_pair"],
    "center & space": ["center_control", "space"],
    "development": ["undeveloped_minors"],
    "loose pieces": ["hanging_pieces", "attacked_by_lower"],
}
assert sorted(f for fs in GROUPS.values() for f in fs) == sorted(SIDE_FEATURES), "groups must cover every feature once"


def concept_index(feature_names: list[str]) -> tuple[list[str], list[np.ndarray]]:
    """(concept names, column indices per concept), one concept per group x side."""
    pos = {n: i for i, n in enumerate(feature_names)}
    names, cols = [], []
    for side, who in (("us", "your"), ("them", "opponent's")):
        for g, feats in GROUPS.items():
            names.append(f"{who} {g}")
            cols.append(np.array([pos[f"{side}_{f}"] for f in feats]))
    return names, cols


@dataclass
class ConceptModel:
    scale: np.ndarray
    feature_names: list[str]
    min_norm: float
    concept_names: list[str]
    concept_cols: list[np.ndarray]
    direction: np.ndarray = None      # (k, n_features): mean signed u where the concept dominates
    meta: dict = field(default_factory=dict)
    stable_cos: float = 0.0

    @property
    def k(self) -> int:
        return len(self.concept_names)

    @property
    def stability(self) -> np.ndarray:
        return np.ones(self.k)       # hand-defined: nothing to be unstable

    def is_stable(self, k: int) -> bool:
        return True

    def name(self, k: int) -> str:
        return self.concept_names[k]

    def encode(self, d: np.ndarray):
        """Returns (S, keep, norms): S = concept shares (rows sum to 1) for kept rows."""
        u = d / self.scale
        finite = np.isfinite(u).all(1)
        norms = np.where(finite, np.linalg.norm(np.nan_to_num(u), axis=1), 0.0)
        keep = finite & (norms >= self.min_norm)
        uk = u[keep]
        sq = np.stack([(uk[:, c] ** 2).sum(1) for c in self.concept_cols], 1)
        return sq / np.maximum(sq.sum(1, keepdims=True), 1e-12), keep, norms

    def fit_directions(self, d: np.ndarray) -> None:
        """Typical direction of each concept on the population: share-weighted mean of u."""
        S, keep, norms = self.encode(d)
        u = d[keep] / self.scale / norms[keep, None]
        self.direction = (S.T @ u) / np.maximum(S.sum(0)[:, None], 1e-12)

    def describe(self, k: int, top: int = 6) -> list[tuple[str, str, float]]:
        """[(arrow, readable feature, weight)] for the concept's features, strongest first.
        ↑ = the best line typically ends with more of it than the other line."""
        cols = self.concept_cols[k]
        v = self.direction[k, cols] if self.direction is not None else np.ones(len(cols))
        w = np.abs(v) / max(np.abs(v).sum(), 1e-12)
        out = []
        for j in np.argsort(-w)[:top]:
            if w[j] < 0.05:
                break
            out.append(("↑" if v[j] >= 0 else "↓", readable(self.feature_names[cols[j]]), float(w[j])))
        return out

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
