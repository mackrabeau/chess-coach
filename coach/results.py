"""Load positions joined with their engine results."""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from .config import data_dir

FILES = {"population": "positions_engine.parquet", "player": "positions.parquet"}


def load_analyzed(cfg: dict, which: str, drop_errors: bool = True) -> pd.DataFrame:
    pos = pd.read_parquet(data_dir(cfg, which) / FILES[which])
    parts = sorted(data_dir(cfg, which, "engine").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no engine results for '{which}' yet; run 02_engine_lines.py")
    eng = pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)
    eng = eng.drop_duplicates("pos_id", keep="last")
    df = pos.merge(eng, on="pos_id", how="inner")
    if drop_errors:
        df = df[df.error.isna()].reset_index(drop=True)
    return df


def load_with_deltas(cfg: dict, which: str):
    """Analyzed positions aligned row-for-row with their feature deltas (05_build_deltas)."""
    import numpy as np

    df = load_analyzed(cfg, which)
    path = data_dir(cfg, which) / "deltas.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run 05_build_deltas.py --set {which}")
    z = np.load(path, allow_pickle=False)
    idx = pd.Series(np.arange(len(z["pos_id"])), index=z["pos_id"])
    df = df[df.pos_id.isin(idx.index)].reset_index(drop=True)
    rows = idx.loc[df.pos_id].to_numpy()
    deltas = {k: z[k][rows] for k in ("F0", "dB", "dP", "dA")}
    df["phase"] = z["phase"][rows]
    return df, deltas, list(z["feature_names"])
