"""Extend every engine line (best / played / 2nd best) until its end position is quiet.

The 6-ply PVs from 02 often stop mid-exchange, so features at the line end can show a
piece "lost" that is about to be recaptured. This pass continues each line with cheap
engine moves, two plies at a time (the end keeps the root's side to move), while the
position at the end is not quiet:
  * side to move is in check, or has <= quiesce.forced_max_legal legal moves
  * the engine's move there is a capture, promotion or check
  * it's an "only move": best beats 2nd best by >= quiesce.forced_gap win probability
up to quiesce.max_extra_plies extra plies. Resumable like 02; does not touch 02's output.

    python scripts/03_settle_lines.py --config configs/v0.yaml --set population --limit 200
    python scripts/03_settle_lines.py --config configs/v0.yaml --set population
    python scripts/03_settle_lines.py --config configs/v0.yaml --set player
    python scripts/03_settle_lines.py --config configs/v0.yaml --set bands    # every band_* sample (17)
    python scripts/03_settle_lines.py --config configs/v0.yaml --set band_1600

Output: data/<set>/engine_q/part-*.parquet. 05 onwards use these lines when
quiesce.use is true.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from coach.bands import resolve_sets, settings
from coach.config import data_dir, load_config, resolve
from coach.engine import LINES, SETTLE_SCHEMA, Analyzer
from coach.parts import done_ids, next_part_index, read_parts, write_part
from coach.results import load_analyzed

_analyzer: Analyzer | None = None


def _init_worker(ecfg: dict, sf_path: str, qcfg: dict) -> None:
    global _analyzer
    _analyzer = Analyzer(ecfg, sf_path, qcfg)


def _work(chunk: list[tuple]) -> list[dict]:
    out = []
    for pos_id, fen, best, played, alt in chunk:
        try:
            res = _analyzer.settle_position(fen, {"best": best, "played": played, "alt": alt})
            res["pos_id"] = pos_id
            res["error"] = None
        except Exception as e:
            res = {"pos_id": pos_id, "error": f"{type(e).__name__}: {e}"}
        out.append(res)
    return out


def summarize(q: pd.DataFrame) -> None:
    q = q[q.error.isna()]
    print(f"\n{len(q):,} positions settled")
    for L in LINES:
        ext = q[f"{L}_q_plies"]
        trig = q[f"{L}_q_trigger"].replace("", "already quiet").value_counts(normalize=True)
        end = q[f"{L}_q_end"].value_counts(normalize=True)
        print(f"  {L:6s} extended {(ext > 0).mean():5.1%}, mean +{ext.mean():.1f} plies | "
              + "trigger: " + ", ".join(f"{k} {v:.0%}" for k, v in trig.items())
              + " | end: " + ", ".join(f"{k} {v:.0%}" for k, v in end.items()))


def settle(cfg, which: str, limit: int | None, workers: int) -> None:
    ecfg, qcfg = dict(cfg["engine"]), dict(cfg["quiesce"])
    sf_path = str(resolve(cfg, cfg["paths"]["stockfish"]))
    df = load_analyzed(cfg, which, raw_lines=True)
    if limit:
        df = df.head(limit)
    out_dir = data_dir(cfg, which, "engine_q")
    done = done_ids(out_dir)
    todo = df[~df.pos_id.isin(done)]
    print(f"{which}: {len(df):,} positions, {len(done):,} already settled, {len(todo):,} to go; "
          f"{workers} workers @ {qcfg['nodes']:,} nodes")

    if not todo.empty:
        if not Path(sf_path).is_file():
            sys.exit(f"Stockfish not found at {sf_path}\n"
                     "Set paths.stockfish in the config to the full path of the .exe.")
        items = [(p, f, list(b), list(pl), list(a)) for p, f, b, pl, a in
                 zip(todo.pos_id, todo.fen, todo.best_line, todo.played_line, todo.alt_line)]
        cs = ecfg["chunk_size"]
        chunks = [items[i : i + cs] for i in range(0, len(items), cs)]
        idx = next_part_index(out_dir)
        t0, n = time.time(), 0
        with mp.Pool(workers, initializer=_init_worker, initargs=(ecfg, sf_path, qcfg)) as pool:
            for rows in pool.imap_unordered(_work, chunks):
                write_part(out_dir, idx, rows, SETTLE_SCHEMA)
                idx += 1
                n += len(rows)
                el = time.time() - t0
                errs = sum(r.get("error") is not None for r in rows)
                print(f"  {n:,}/{len(items):,}  {n / el:.2f} pos/s  "
                      f"ETA {el / n * (len(items) - n) / 3600:.1f} h"
                      + (f"  ({errs} errors in last chunk)" if errs else ""), flush=True)
        print(f"done in {(time.time() - t0) / 3600:.2f} h -> {out_dir}")

    summarize(read_parts(out_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--set", required=True,
                    help="player, population, a band sample (band_<lo>) or 'bands' (every band_* set of 17)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N positions (testing)")
    ap.add_argument("--workers", type=int, default=None, help="override engine.workers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["quiesce"]["max_extra_plies"] % 2:
        sys.exit("quiesce.max_extra_plies must be even")
    workers = args.workers or cfg["engine"]["workers"]
    if args.set == "bands":
        names = [n for n, _ in resolve_sets(cfg, settings(cfg))
                 if n.startswith("band_") and any(data_dir(cfg, n, "engine").glob("part-*.parquet"))]
        if not names:
            print("no band_* sets with engine results (17_multiband mines them); nothing to settle")
    else:
        names = [args.set]
    for n in names:
        settle(cfg, n, args.limit, workers)


if __name__ == "__main__":
    mp.freeze_support()
    main()
