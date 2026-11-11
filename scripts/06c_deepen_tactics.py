"""Deep pass for blunders / tactical mistakes with no material swing in the settled lines.

About half of all blunders show no >= 2-pawn material swing within the settled lines
(~6-14 plies). This continues their best and played lines by up to deep.plies more plies
(engine best moves, deep.nodes each) so that material won or lost beyond the horizon
(pawn races, promotions, mating nets) gets classified. Resumable like 02/03.
The deep lines are used only to classify tactics (07b, 08), never for features.

    python scripts/06c_deepen_tactics.py --config configs/v0.yaml --set population
    python scripts/06c_deepen_tactics.py --config configs/v0.yaml --set player

Needs 06 (for the track routing). Output: data/<set>/engine_deep/part-*.parquet
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach.config import data_dir, load_config, resolve
from coach.engine import DEEP_SCHEMA, Analyzer
from coach.ideas import IdeaModel, move_tracks
from coach.motifs import tactic_labels
from coach.parts import done_ids, next_part_index, write_part
from coach.results import load_with_deltas

_analyzer: Analyzer | None = None
_dcfg: dict = {}


def _init_worker(ecfg: dict, sf_path: str, dcfg: dict) -> None:
    global _analyzer, _dcfg
    _analyzer = Analyzer(ecfg, sf_path)
    _dcfg = dcfg


def _work(chunk: list[tuple]) -> list[dict]:
    out = []
    for pos_id, fen, best, played in chunk:
        try:
            game = object()
            out.append({
                "pos_id": pos_id, "error": None,
                "best_line_deep": _analyzer.deepen_line(fen, best, game, _dcfg["plies"], _dcfg["nodes"]),
                "played_line_deep": _analyzer.deepen_line(fen, played, game, _dcfg["plies"], _dcfg["nodes"]),
            })
        except Exception as e:
            out.append({"pos_id": pos_id, "error": f"{type(e).__name__}: {e}"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--set", choices=["population", "player"], required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg, dcfg = dict(cfg["engine"]), dict(cfg["deep"])
    sf_path = str(resolve(cfg, cfg["paths"]["stockfish"]))
    workers = args.workers or ecfg["workers"]

    model = IdeaModel.load(data_dir(cfg, "ideas") / "model.pkl")
    df, D, names = load_with_deltas(cfg, args.set)
    tl = tactic_labels(df, move_tracks(model, cfg, df, D, names))
    sel = df.loc[tl.index[tl.direction == "no_material"]]
    out_dir = data_dir(cfg, args.set, "engine_deep")
    done = done_ids(out_dir)
    todo = sel[~sel.pos_id.isin(done)]
    print(f"{len(tl):,} blunders + tactical mistakes, {len(sel):,} with no material swing; "
          f"{len(done):,} already deepened, {len(todo):,} to go; "
          f"+{dcfg['plies']} plies @ {dcfg['nodes']:,} nodes, {workers} workers")
    if todo.empty:
        return
    if not Path(sf_path).is_file():
        sys.exit(f"Stockfish not found at {sf_path}")

    items = [(p, f, list(b), list(pl)) for p, f, b, pl in
             zip(todo.pos_id, todo.fen, todo.best_line, todo.played_line)]
    cs = max(1, min(ecfg["chunk_size"], len(items) // (workers * 2) or 1))
    chunks = [items[i : i + cs] for i in range(0, len(items), cs)]
    idx = next_part_index(out_dir)
    t0, n = time.time(), 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(ecfg, sf_path, dcfg)) as pool:
        for rows in pool.imap_unordered(_work, chunks):
            write_part(out_dir, idx, rows, DEEP_SCHEMA)
            idx += 1
            n += len(rows)
            el = time.time() - t0
            print(f"  {n:,}/{len(items):,}  {n / el:.2f} pos/s  ETA {el / n * (len(items) - n) / 60:.0f} min",
                  flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
