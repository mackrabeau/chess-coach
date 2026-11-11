"""Load positions joined with their engine results."""
from __future__ import annotations

import pandas as pd

from .config import data_dir
from .engine import LINES
from .parts import read_parts

FILES = {"population": "positions_engine.parquet", "player": "positions.parquet"}


def load_analyzed(cfg: dict, which: str, drop_errors: bool = True,
                  raw_lines: bool = False) -> pd.DataFrame:
    """Positions joined with engine results.

    If quiesce.use is true and 03_settle_lines.py has run, best_line / played_line /
    alt_line are replaced by the settled (extended) lines; the 02 originals stay in
    *_line_raw. raw_lines=True skips this.
    """
    pos = pd.read_parquet(data_dir(cfg, which) / FILES[which])
    eng = read_parts(data_dir(cfg, which, "engine"))
    if eng is None:
        raise FileNotFoundError(f"no engine results for '{which}' yet; run 02_engine_lines.py")
    df = pos.merge(eng, on="pos_id", how="inner")
    if drop_errors:
        df = df[df.error.isna()].reset_index(drop=True)
    if not raw_lines and cfg.get("quiesce", {}).get("use", False):
        df = _use_settled(cfg, which, df)
    return df


def _use_settled(cfg: dict, which: str, df: pd.DataFrame) -> pd.DataFrame:
    q = read_parts(data_dir(cfg, which, "engine_q"))
    if q is None:
        print(f"note: no settled lines for '{which}' (03_settle_lines.py not run); using raw 02 lines")
        return df
    missing = ~df.pos_id.isin(q.pos_id)
    if missing.any():
        raise RuntimeError(
            f"settled lines cover {(~missing).sum():,}/{len(df):,} '{which}' positions. Finish "
            f"03_settle_lines.py --set {which}, or set quiesce.use: false to use the raw lines.")
    df = df.merge(q.drop(columns="error").assign(q_error=q.error.to_numpy()), on="pos_id", how="left")
    bad = df.q_error.notna()
    if bad.any():
        print(f"note: {bad.sum()} '{which}' positions failed to settle; using their raw lines")
    for L in LINES:
        df[f"{L}_line_raw"] = df[f"{L}_line"]
        df[f"{L}_line"] = df[f"{L}_line_q"].where(~bad, df[f"{L}_line_raw"])
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


def load_deep(cfg: dict, which: str):
    """Deep tactic lines from 06c (None if it hasn't run)."""
    return read_parts(data_dir(cfg, which, "engine_deep"))
