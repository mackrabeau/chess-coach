"""Load positions joined with their engine results."""
from __future__ import annotations

import pandas as pd

from .config import data_dir
from .engine import LINES
from .parts import read_parts
from .runner import position_file


def load_analyzed(cfg: dict, which: str, drop_errors: bool = True,
                  raw_lines: bool = False, refined: bool = True) -> pd.DataFrame:
    """Positions joined with engine results.

    refined: where refine.py has re-searched a position deeper (data/<set>/engine_L<k>/),
    the deepest result replaces the 02 one (all engine columns; `refine_level` says which).
    Refine always treats player and peers with the same rule, so comparisons stay fair; pass
    refined=False when the other side of a comparison was never refined (e.g. 17's bands).
    If quiesce.use is true and 03_settle_lines.py has run, best_line / played_line /
    alt_line are replaced by the settled (extended) lines for unrefined positions (refined
    ones keep their own, deeper lines); the 02 originals stay in *_line_raw.
    raw_lines=True skips the settled lines.
    """
    pos = pd.read_parquet(data_dir(cfg, which) / position_file(which))
    eng = read_parts(data_dir(cfg, which, "engine"))
    if eng is None:
        raise FileNotFoundError(f"no engine results for '{which}' yet; run 02_engine_lines.py")
    eng["refine_level"] = 0
    if refined:
        eng = _use_refined(cfg, which, eng)
    df = pos.merge(eng, on="pos_id", how="inner")
    if drop_errors:
        df = df[df.error.isna()].reset_index(drop=True)
    if not raw_lines and cfg.get("quiesce", {}).get("use", False):
        df = _use_settled(cfg, which, df)
    return df


def _use_refined(cfg: dict, which: str, eng: pd.DataFrame) -> pd.DataFrame:
    """Replace engine rows by the deepest successful refine result per position."""
    base = data_dir(cfg, which)
    levels = sorted(int(p.name[8:]) for p in base.glob("engine_L*") if p.name[8:].isdigit())
    for k in levels:
        r = read_parts(base / f"engine_L{k}")
        if r is None:
            continue
        r = r[r.error.isna()].copy()
        r["refine_level"] = k
        eng = pd.concat([eng[~eng.pos_id.isin(r.pos_id)], r[eng.columns.intersection(r.columns)]],
                        ignore_index=True)
    return eng


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
    bad = bad | (df.refine_level > 0)       # refined positions keep their own (deeper) lines
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
