"""Final game results per position (the mover's score: 1 / 0.5 / 0)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import data_dir


def mover_results(cfg: dict, df: pd.DataFrame, which: str) -> np.ndarray:
    """Mover's final score per row of df; NaN where unknown.
    you: the `result` column written by 00 (re-run 00 after this was added);
    peers: data/population/results.parquet from 01b (White's score per game)."""
    if "result" in df.columns and df["result"].notna().any():
        return df["result"].to_numpy(float)
    if which == "population":
        p = data_dir(cfg, "population") / "results.parquet"
        if p.exists():
            r = pd.read_parquet(p).drop_duplicates("game_id").set_index("game_id").white_score
            w = df.game_id.map(r).to_numpy(float)
            return np.where(df.mover_white.to_numpy(), w, 1.0 - w)
    return np.full(len(df), np.nan)
