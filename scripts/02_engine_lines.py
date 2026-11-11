"""Run Stockfish over decision positions. Resumable: safe to kill and restart.

Writes one parquet part per chunk to data/<set>/engine/part-XXXXXX.parquet; on restart,
positions already present in any part are skipped.

    python scripts/02_engine_lines.py --config configs/v0.yaml --set population
    python scripts/02_engine_lines.py --config configs/v0.yaml --set player
    python scripts/02_engine_lines.py --config configs/v0.yaml --set population --limit 200   # quick test
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from coach.config import data_dir, load_config, resolve
from coach.engine import RESULT_SCHEMA, Analyzer
from coach.parts import done_ids, next_part_index, write_part

INPUTS = {
    "population": ("population", "positions_engine.parquet"),
    "player": ("player", "positions.parquet"),
}

_analyzer: Analyzer | None = None


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--set", choices=sorted(INPUTS), required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N positions (testing)")
    ap.add_argument("--workers", type=int, default=None, help="override engine.workers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ecfg = dict(cfg["engine"])
    sf_path = str(resolve(cfg, cfg["paths"]["stockfish"]))
    workers = args.workers or ecfg["workers"]

    sub, fname = INPUTS[args.set]
    src = data_dir(cfg, sub) / fname
    engine_dir = data_dir(cfg, sub, "engine")
    df = pd.read_parquet(src, columns=["pos_id", "fen", "move"])
    if args.limit:
        df = df.head(args.limit)

    done = done_ids(engine_dir)
    todo = df[~df.pos_id.isin(done)]
    print(f"{len(df):,} positions, {len(done):,} already done, {len(todo):,} to go; "
          f"{workers} workers @ {ecfg['nodes']:,} nodes")
    if todo.empty:
        return

    # Fail fast if Stockfish isn't where the config says.
    if not Path(sf_path).is_file():
        sys.exit(f"Stockfish not found at {sf_path}\n"
                 "Set paths.stockfish in the config to the full path of the .exe.")
    Analyzer(ecfg, sf_path).close()

    items = list(todo.itertuples(index=False, name=None))
    cs = ecfg["chunk_size"]
    chunks = [items[i : i + cs] for i in range(0, len(items), cs)]
    idx = next_part_index(engine_dir)

    t0, n = time.time(), 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(ecfg, sf_path)) as pool:
        for rows in pool.imap_unordered(_work, chunks):
            write_part(engine_dir, idx, rows, RESULT_SCHEMA)
            idx += 1
            n += len(rows)
            el = time.time() - t0
            eta = el / n * (len(items) - n)
            errs = sum(r.get("error") is not None for r in rows)
            print(f"  {n:,}/{len(items):,}  {n / el:.2f} pos/s  ETA {eta / 3600:.1f} h"
                  + (f"  ({errs} errors in last chunk)" if errs else ""), flush=True)
    print(f"done in {(time.time() - t0) / 3600:.2f} h -> {engine_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
