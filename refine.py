"""Pour spare compute into the database: deeper searches and more data, for as long as you let it.

Leave it running whenever the PC is free. Everything is saved as it goes (atomic part files +
data/refine_state.json), so a crash or power cut loses at most one chunk; rerun the same command
and it continues. Afterwards, `python main.py` rebuilds whatever the new results made stale.

Work happens in levels; level k searches with engine.nodes x 4^k nodes (1M -> 4M -> 16M -> 64M):
  1. priority positions, you AND peers with the same rule: mistakes/blunders, positions near the
     ok/mistake/blunder thresholds or the critical gap (labels that a deeper search might flip),
     and critical positions (puzzle candidates)
  2. opening tree at openings.nodes x 4^k (14 keeps the deepest result per position)
  3. more multi-band data: multiband.target x (k + 1) positions per band (17 mines incrementally)
  4. every remaining position of yours and the peers'
Deeper results land in data/<set>/engine_L<k>/ and replace the shallower ones when loading
(coach/results.load_analyzed; bands stay at the base depth so 17 compares like with like).

    python refine.py                     # run until --max-level is exhausted (or forever with a big one)
    python refine.py --hours 8           # stop starting new work after 8 hours
    python refine.py --status            # what's done, what's next
    python refine.py --update-reports    # run main.py after each finished level
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from coach.config import data_dir, load_config  # noqa: E402
from coach.engine import win_prob  # noqa: E402
from coach.results import load_analyzed  # noqa: E402
from coach.runner import engine_pass, level_dir  # noqa: E402


def priority_ids(cfg, which) -> list[str]:
    """Positions whose conclusions a deeper search is most likely to change."""
    e = cfg["engine"]
    df = load_analyzed(cfg, which, raw_lines=True)
    wl = df.wp_loss.to_numpy(float)
    gap = df.wp_best.to_numpy(float) - np.array([win_prob(c) if c == c else np.nan for c in df.cp_alt], float)
    near = (np.abs(wl - e["ok_loss"]) < 0.015) | (np.abs(wl - e["blunder_loss"]) < 0.03) \
        | (np.abs(gap - e["critical_gap"]) < 0.03)
    keep = df.label.isin(["mistake", "blunder"]).to_numpy() | near | df.critical.astype(bool).to_numpy()
    return df.pos_id[keep].tolist()


def all_ids(cfg, which) -> list[str]:
    return load_analyzed(cfg, which, raw_lines=True, refined=False).pos_id.tolist()


def level_cfg(cfg, k):
    e = cfg["engine"]
    return {"nodes": e["nodes"] * 4 ** k, "extend_nodes": e["extend_nodes"] * 2 ** k}


def tasks(cfg, k, workers):
    """(name, function(deadline) -> finished?) for level k, in order."""
    ov = level_cfg(cfg, k)
    out = []
    for which in ("player", "population"):
        out.append((f"L{k} priority {which} @ {ov['nodes']:,} nodes",
                    lambda d, w=which: engine_pass(cfg, w, workers=workers, ecfg_override=ov,
                                                   out_sub=level_dir(k), ids=priority_ids(cfg, w), deadline=d)))
    on = cfg.get("openings", {}).get("nodes", 300000) * 4 ** k
    out.append((f"L{k} opening tree @ {on:,} nodes", lambda d: run(["14_openings.py", "--nodes", str(on)])))
    target = cfg.get("multiband", {}).get("target", 4000) * (k + 1)
    out.append((f"L{k} multi-band data to {target:,} positions per band",
                lambda d: run(["17_multiband.py", "--stage", "mine", "--target", str(target)])
                and run(["17_multiband.py", "--stage", "engine"])))
    for which in ("player", "population"):
        out.append((f"L{k} all {which} @ {ov['nodes']:,} nodes",
                    lambda d, w=which: engine_pass(cfg, w, workers=workers, ecfg_override=ov,
                                                   out_sub=level_dir(k), ids=all_ids(cfg, w), deadline=d)))
    return out


CONFIG = "configs/v0.yaml"


def run(script_args) -> bool:
    cmd = [sys.executable, str(ROOT / "scripts" / script_args[0]), "--config", CONFIG, *script_args[1:]]
    print("   $ " + " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode == 0


def main() -> None:
    global CONFIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--hours", type=float, default=None, help="stop starting new work after this long")
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--update-reports", action="store_true")
    args = ap.parse_args()
    CONFIG = args.config
    cfg = load_config(CONFIG)
    state_path = data_dir(cfg) / "refine_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": []}
    deadline = time.time() + 3600 * args.hours if args.hours else None
    plan = [(k, name, fn) for k in range(1, args.max_level + 1) for name, fn in tasks(cfg, k, args.workers)]

    if args.status:
        for k, name, _ in plan:
            print(f"  [{'x' if name in state['done'] else ' '}] {name}")
        return
    for k, name, fn in plan:
        if name in state["done"]:
            continue
        if deadline is not None and time.time() > deadline:
            print("time budget used; rerun to continue")
            return
        print(f"\n== {name}", flush=True)
        finished = fn(deadline)
        if not finished:
            print(f"   not finished: {name} (resumes next run)")
            return
        state["done"].append(name)
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(state_path)
        last_of_level = name == [n for kk, n, _ in plan if kk == k][-1]
        if args.update_reports and last_of_level:
            subprocess.run([sys.executable, str(ROOT / "main.py"), "--config", CONFIG, "--offline"], cwd=ROOT)
    print("all planned levels done; raise --max-level to keep going")


if __name__ == "__main__":
    mp.freeze_support()
    main()
