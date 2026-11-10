"""Compute feature deltas along the engine lines.

For each analyzed position x (us = side that moved):
    F0   = f(x)
    dB   = f(end of best line)   - f(x)
    dP   = f(end of played line) - f(x)
    dA   = f(end of 2nd-best line) - f(x)
Rows are NaN where the line has fewer than 2 plies (e.g. immediate mate).

    python scripts/05_build_deltas.py --set population
    python scripts/05_build_deltas.py --set player

Output: data/<set>/deltas.npz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np

from coach.config import data_dir, load_config
from coach.handcrafted import FEATURE_NAMES, N_FEATURES, features, phase
from coach.results import load_analyzed

MIN_PLIES = 2


def _end_delta(root: chess.Board, us: chess.Color, f0: np.ndarray, line) -> np.ndarray:
    if line is None or len(line) < MIN_PLIES:
        return np.full(N_FEATURES, np.nan, dtype=np.float32)
    b = root.copy(stack=False)
    for m in line:
        b.push_uci(m)
    return features(b, us) - f0


def _work(row):
    fen, best, played, alt = row
    root = chess.Board(fen)
    us = root.turn
    f0 = features(root, us)
    return (
        f0,
        _end_delta(root, us, f0, best),
        _end_delta(root, us, f0, played),
        _end_delta(root, us, f0, alt),
        phase(root),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--set", choices=["population", "player"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.config)

    df = load_analyzed(cfg, args.set)
    rows = list(zip(df.fen, df.best_line, df.played_line, df.alt_line))
    print(f"{len(rows):,} positions")
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        out = pool.map(_work, rows, chunksize=200)
    F0, dB, dP, dA, ph = (np.stack(x) if i < 4 else np.array(x) for i, x in enumerate(zip(*out)))

    path = data_dir(cfg, args.set) / "deltas.npz"
    np.savez_compressed(
        path, pos_id=df.pos_id.to_numpy().astype(str), F0=F0, dB=dB, dP=dP, dA=dA,
        phase=ph, feature_names=np.array(FEATURE_NAMES),
    )
    valid = np.isfinite(dB).all(1) & np.isfinite(dP).all(1)
    print(f"done in {time.time() - t0:.0f} s; {valid.sum():,}/{len(df):,} with usable best+played lines -> {path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
