"""Resumable multiprocess Stockfish pass over a position file (used by 02 and 17)."""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import pandas as pd

from .config import data_dir, resolve
from .engine import RESULT_SCHEMA, Analyzer
from .parts import done_ids, next_part_index, write_part

_analyzer: Analyzer | None = None


def position_file(which: str) -> str:
    """Positions file of a set: the player's moves, or a mined population / band sample."""
    return "positions.parquet" if which == "player" else "positions_engine.parquet"


def _init_worker(ecfg: dict, sf_path: str) -> None:
    global _analyzer
    _analyzer = Analyzer(ecfg, sf_path)


def _work(chunk: list[tuple[str, str, str]]) -> list[dict]:
    out = []
    for pos_id, fen, move in chunk:
        try:
            res = _analyzer.analyse_position(fen, move)
            res["pos_id"] = pos_id
            res["error"] = None
        except Exception as e:  # keep going; record the failure
            res = {"pos_id": pos_id, "error": f"{type(e).__name__}: {e}"}
        out.append(res)
    return out


def engine_pass(cfg: dict, which: str, limit: int | None = None, workers: int | None = None,
                ecfg_override: dict | None = None, out_sub: str = "engine") -> None:
    """Stockfish every position of set `which` not yet in data/<which>/<out_sub>/ (resumable)."""
    ecfg = {**cfg["engine"], **(ecfg_override or {})}
    sf_path = str(resolve(cfg, cfg["paths"]["stockfish"]))
    workers = workers or ecfg["workers"]
    src = data_dir(cfg, which) / position_file(which)
    engine_dir = data_dir(cfg, which, out_sub)
    df = pd.read_parquet(src, columns=["pos_id", "fen", "move"])
    if limit:
        df = df.head(limit)
    done = done_ids(engine_dir)
    todo = df[~df.pos_id.isin(done)]
    print(f"[{which}] {len(df):,} positions, {len(done):,} already done, {len(todo):,} to go; "
          f"{workers} workers @ {ecfg['nodes']:,} nodes")
    if todo.empty:
        return
    if not Path(sf_path).is_file():
        sys.exit(f"Stockfish not found at {sf_path}\n"
                 "Set paths.stockfish in the config to the full path of the executable.")
    Analyzer(ecfg, sf_path).close()
    items = list(todo.itertuples(index=False, name=None))
    cs = ecfg["chunk_size"]
    chunks = [items[i: i + cs] for i in range(0, len(items), cs)]
    idx = next_part_index(engine_dir)
    t0, n = time.time(), 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(ecfg, sf_path)) as pool:
        for rows in pool.imap_unordered(_work, chunks):
            write_part(engine_dir, idx, rows, RESULT_SCHEMA)
            idx += 1
            n += len(rows)
            el = time.time() - t0
            errs = sum(r.get("error") is not None for r in rows)
            print(f"  {n:,}/{len(items):,}  {n / el:.2f} pos/s  ETA {el / n * (len(items) - n) / 3600:.1f} h"
                  + (f"  ({errs} errors in last chunk)" if errs else ""), flush=True)
    print(f"done in {(time.time() - t0) / 3600:.2f} h -> {engine_dir}")
