"""Plain-language 'why' for a position: what the best line ends with that yours doesn't.

Uses the feature deltas from 05 (d = features at the end of the best line - at the end of the played
line) scaled like the idea model (06), and names the largest differences:
  "the best line ends with more of your piece activity and less of the opponent's king safety"
Returns {} when 05/06 haven't run.
"""
from __future__ import annotations

import pickle

import numpy as np

from .config import data_dir
from .handcrafted import readable


def why_texts(cfg, which: str = "player", pos_ids=None, top: int = 2) -> dict[str, str]:
    from .results import load_with_deltas
    mp = data_dir(cfg, "ideas") / "model.pkl"
    try:
        df, D, names = load_with_deltas(cfg, which)
        with open(mp, "rb") as f:
            model = pickle.load(f)
    except Exception:
        return {}
    keep = np.ones(len(df), bool) if pos_ids is None else df.pos_id.isin(set(pos_ids)).to_numpy()
    d = (D["dB"] - D["dP"])[keep] / np.asarray(model.scale)
    out = {}
    for pid, u in zip(df.pos_id[keep], d):
        if not np.isfinite(u).all():
            continue
        idx = [j for j in np.argsort(-np.abs(u))[:top] if abs(u[j]) >= 0.5]
        if not idx:
            continue
        parts = []
        for j in idx:
            who, _, feat = readable(names[j]).partition(" ")
            subj = "you end" if who == "your" else "your opponent ends"
            parts.append(f"{subj} with {'more' if u[j] > 0 else 'less'} {feat}")
        out[pid] = "after the best line " + "; ".join(parts) + " than after yours"
    return out
