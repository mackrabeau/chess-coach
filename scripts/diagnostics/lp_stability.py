"""Line-pattern label stability: re-search a sample of peer opportunities at 4x nodes (future_work step 5).

For up to --per-pattern opportunity positions per pattern (from <population>/linepatterns.parquet, written by
24), rerun the 02 analysis at engine.nodes x --mult, settle the lines like 03, re-tag with
coach/linepatterns, and count how often the opportunity label flips off. Target: flip rate < 25%.
coach/linepatterns.pattern_table multiplies label confidence by (1 - flip rate) once this file exists.

    python scripts/diagnostics/lp_stability.py --config configs/v0.yaml --per-pattern 25

Output: <population>/linepatterns_stability.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from coach.config import data_dir, load_config, resolve
from coach.engine import Analyzer
from coach.linepatterns import CONTROL, PATTERNS, VERSION, _row
from coach.results import load_analyzed

_an = None


def _init(ecfg, sf, qcfg):
    global _an
    _an = Analyzer(ecfg, sf, qcfg)


def _work(item):
    pid, fen, played = item
    try:
        r = _an.analyse_position(fen, played)
        q = _an.settle_position(fen, {"best": r["best_line"], "played": r["played_line"], "alt": r["alt_line"]})
        wp_alt = 1.0 / (1.0 + np.exp(-0.00368208 * r["cp_alt"])) if r["cp_alt"] is not None else float("nan")
        rows = _row((pid, fen, q["best_line_q"], q["alt_line_q"], q["played_line_q"], r["played_is_best"],
                     r["wp_best"], r["wp_played"], wp_alt))
        return pid, {x[1] for x in rows if x[3]}
    except Exception as e:
        print(f"  {pid}: {type(e).__name__}: {e}", flush=True)
        return pid, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--per-pattern", type=int, default=25)
    ap.add_argument("--mult", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    lp = pd.read_parquet(data_dir(cfg, "population") / "linepatterns.parquet")
    opp = lp[lp.opportunity & (lp.pattern != CONTROL)]
    sample = pd.concat([d.sample(min(len(d), args.per_pattern), random_state=0)
                        for _, d in opp.groupby("pattern")], ignore_index=True)
    df = load_analyzed(cfg, "population", refined=False).set_index("pos_id")
    ids = sample.pos_id.unique()
    items = [(p, df.at[p, "fen"], df.at[p, "move"]) for p in ids]
    ecfg = dict(cfg["engine"])
    ecfg["nodes"] = ecfg["nodes"] * args.mult
    sf = str(resolve(cfg, cfg["paths"]["stockfish"]))
    print(f"{len(items)} positions ({len(sample)} opportunity labels) at {ecfg['nodes']:,} nodes")
    t0, got = time.time(), {}
    with mp.Pool(args.workers or cfg["engine"]["workers"], initializer=_init,
                 initargs=(ecfg, sf, dict(cfg["quiesce"]))) as pool:
        for i, (pid, pats) in enumerate(pool.imap_unordered(_work, items, chunksize=1), 1):
            got[pid] = pats
            if i % 25 == 0:
                print(f"  {i}/{len(items)}  {i / (time.time() - t0):.2f} pos/s", flush=True)
    raw = data_dir(cfg, "population") / "linepatterns_stability_raw.json"     # engine work, kept on a crash
    raw.write_text(json.dumps({k: sorted(v) if v is not None else None for k, v in got.items()}), encoding="utf-8")
    out = {}
    for pat, d in sample.groupby("pattern"):
        keep = [p for p in d.pos_id if got.get(p) is not None]
        flips = [pat not in got[p] for p in keep]
        out[pat] = {"n": len(keep), "flip_rate": float(np.mean(flips)) if keep else None}
        print(f"  {pat:20s} n {len(keep):3d}  flip rate {out[pat]['flip_rate'] if keep else float('nan'):.0%}")
    allf = [pat not in got[p] for pat, p in zip(sample.pattern, sample.pos_id) if got.get(p) is not None]
    res = {"version": VERSION, "nodes": ecfg["nodes"], "mult": args.mult, "overall_flip_rate": float(np.mean(allf)),
           "patterns": out, "families": {f: None for f in dict.fromkeys(v[0] for v in PATTERNS.values())}}
    for f in res["families"]:
        fl = [pat not in got[p] for pat, p in zip(sample.pattern, sample.pos_id)
              if PATTERNS[pat][0] == f and got.get(p) is not None]
        res["families"][f] = float(np.mean(fl)) if fl else None
    path = data_dir(cfg, "population") / "linepatterns_stability.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"overall flip rate {res['overall_flip_rate']:.0%} -> {path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
