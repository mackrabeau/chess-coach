"""Run the whole pipeline, from nothing to the reports. Safe to stop and rerun at any time.

Each stage is skipped when its outputs are newer than its inputs (make-style), and the long
engine/Maia passes are resumable on their own, so a rerun picks up where it stopped.

    python main.py                       # everything that is missing or stale
    python main.py --list                # stages, and what would run
    python main.py --only openings,traps # just these
    python main.py --from deltas         # this stage and everything after it
    python main.py --force skill         # rerun even if up to date
    python main.py --update              # also refetch your newest chess.com games first
    python main.py --skip-maia           # no GPU / torch: skip the Maia stages (and what needs them)

Prerequisites (see README): Stockfish at paths.stockfish, a Lichess .pgn.zst dump at
paths.lichess_dump, player.username set. Maia stages need torch + maia3.
For more accuracy later, leave refine.py running (deeper searches, more data); then rerun this.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from coach.config import load_config, resolve  # noqa: E402


@dataclass
class Stage:
    name: str
    script: str
    args: list = field(default_factory=list)
    inputs: list = field(default_factory=list)    # globs relative to the data dir
    outputs: list = field(default_factory=list)   # globs relative to the data dir
    always: bool = False                          # resumable pass: always invoke (no-op when done)
    maia: bool = False                            # needs torch + maia3
    network: bool = False                         # skipped with --offline
    offline_args: list | None = None              # with --offline, run with these args instead of skipping


ENGINE = ["{s}/engine/*.parquet", "{s}/engine_L*/*.parquet"]


def eng(s):
    return [g.format(s=s) for g in ENGINE]


STAGES = [
    # data
    Stage("fetch", "00_fetch_player.py", outputs=["player/positions.parquet"], offline_args=["--offline"]),
    Stage("mine", "01_mine_population.py", outputs=["population/positions_engine.parquet"]),
    Stage("results", "01b_fetch_results.py", always=True, network=True),
    Stage("syzygy", "01c_download_syzygy.py", always=True, network=True),
    # engine (long, resumable)
    Stage("engine-player", "02_engine_lines.py", ["--set", "player"], always=True),
    Stage("engine-population", "02_engine_lines.py", ["--set", "population"], always=True),
    Stage("settle-player", "03_settle_lines.py", ["--set", "player"], always=True),
    Stage("settle-population", "03_settle_lines.py", ["--set", "population"], always=True),
    Stage("deep-player", "06c_deepen_tactics.py", ["--set", "player"], always=True),
    Stage("deep-population", "06c_deepen_tactics.py", ["--set", "population"], always=True),
    # features, ideas
    Stage("deltas-player", "05_build_deltas.py", ["--set", "player"],
          eng("player") + ["player/engine_q/*.parquet"], ["player/deltas.npz"]),
    Stage("deltas-population", "05_build_deltas.py", ["--set", "population"],
          eng("population") + ["population/engine_q/*.parquet"], ["population/deltas.npz"]),
    Stage("ideas", "06_learn_ideas.py", [], ["*/deltas.npz"], ["ideas/model.pkl"]),
    # Maia (GPU)
    Stage("maia", "09b_maia_difficulty.py", ["--set", "both"], always=True, maia=True),
    Stage("anatomy", "09c_error_anatomy.py", [], eng("player") + eng("population") + ["*/maia_*/*.parquet"],
          ["player/anatomy.parquet", "population/anatomy.parquet"], maia=True),
    Stage("opponent-model", "09d_opponent_model.py", [], eng("player") + eng("population"),
          ["player/opp_model.parquet"], maia=True),
    # analysis: layer 1
    Stage("tactics-report", "07b_inspect_tactics.py", [], ["*/engine_deep/*.parquet", "ideas/model.pkl"],
          ["ideas/tactics_report.html"]),
    Stage("ideas-report", "07_inspect_ideas.py", [], ["ideas/model.pkl"], ["ideas/ideas_report.html"]),
    Stage("player-report", "08_player_report.py", [],
          ["ideas/model.pkl", "*/anatomy.parquet", "*/engine_deep/*.parquet"], ["player/player_report.html"]),
    Stage("endgames", "11_endgames.py", [], eng("player") + eng("population") + ["population/results.parquet"],
          ["player/endgame_report.html"]),
    Stage("balanced-endgames", "11b_balanced_endgames.py", [], eng("player") + eng("population"),
          ["player/balanced_endgames.html"]),
    Stage("openings", "14_openings.py", [], ["player/archives/*.json", "openings/engine*/*.parquet"],
          ["player/openings_report.html"]),
    Stage("repertoire", "19_repertoire.py", [], ["openings/book.parquet", "openings/engine*/*.parquet",
                                                  "player/archives/*.json"],
          ["player/repertoire.html"], maia=True),
    Stage("score", "15_score_accounting.py", [], eng("player") + ["population/results.parquet"],
          ["player/score_accounting.parquet"]),
    Stage("skill", "16_skill_profile.py", [], eng("player") + eng("population") + ["*/maia_*/*.parquet"],
          ["player/skill_profile.parquet"], maia=True),
    Stage("multiband", "17_multiband.py", [], always=True),
    # layer 2: exploitation (exploratory)
    Stage("practical-tree", "13_practical_tree.py", [], eng("player") + eng("population"),
          ["player/practical_tree.parquet"], maia=True),
    Stage("traps", "18_opening_traps.py", [], ["openings/book.parquet", "openings/engine*/*.parquet"],
          ["player/opening_traps.html"], maia=True),
]


def newest(data: Path, globs) -> float:
    ts = [p.stat().st_mtime for g in globs for p in data.glob(g)]
    return max(ts, default=0.0)


def oldest(data: Path, globs) -> float | None:
    ts = []
    for g in globs:
        hits = list(data.glob(g))
        if not hits:
            return None                      # an output is missing
        ts += [p.stat().st_mtime for p in hits]
    return min(ts)


def status(st: Stage, data: Path) -> str:
    if st.always:
        return "resumable (always invoked; no-op when done)"
    out = oldest(data, st.outputs)
    if out is None:
        return "missing"
    if st.inputs and newest(data, st.inputs) > out:
        return "stale"
    return "up to date"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated stage names")
    ap.add_argument("--from", dest="start", default=None, help="start at this stage")
    ap.add_argument("--force", default="", help="comma-separated stages to rerun regardless")
    ap.add_argument("--update", action="store_true", help="refetch your chess.com games")
    ap.add_argument("--skip-maia", action="store_true")
    ap.add_argument("--offline", action="store_true", help="skip stages that need the internet")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = resolve(cfg, cfg["paths"]["data_dir"])
    names = [s.name for s in STAGES]
    stages = STAGES
    if args.only:
        want = set(args.only.split(","))
        bad = want - set(names)
        if bad:
            sys.exit(f"unknown stages: {', '.join(sorted(bad))}; see --list")
        stages = [s for s in STAGES if s.name in want]
    elif args.start:
        if args.start not in names:
            sys.exit(f"unknown stage {args.start}; see --list")
        stages = STAGES[names.index(args.start):]
    force = set(filter(None, args.force.split(",")))
    if args.update:
        force.add("fetch")

    if args.list:
        for s in STAGES:
            tag = " [maia]" if s.maia else ""
            print(f"  {s.name:20s} {s.script:28s} {status(s, data)}{tag}")
        return

    log_dir = data / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"main_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log = open(log_path, "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"log: {log_path}")
    failed = []
    t_all = time.time()
    for s in stages:
        if args.skip_maia and s.maia:
            say(f"-- {s.name}: skipped (--skip-maia)")
            continue
        if args.offline and s.network:
            say(f"-- {s.name}: skipped (--offline)")
            continue
        st = status(s, data)
        if s.name not in force and st == "up to date":
            say(f"-- {s.name}: up to date")
            continue
        extra = s.offline_args if (args.offline and s.offline_args) else []
        cmd = [sys.executable, str(ROOT / "scripts" / s.script), "--config", args.config, *s.args, *extra]
        say(f"\n== {s.name} ({st}) :: {' '.join(cmd[1:])}")
        t0 = time.time()
        # stream the stage's output to the screen and the log (python -u: no buffering)
        proc = subprocess.Popen([cmd[0], "-u", *cmd[1:]], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        rc = proc.wait()
        if rc != 0:
            say(f"!! {s.name} failed (exit {rc})")
            failed.append(s.name)
            if not args.keep_going:
                say(f"stopped at {s.name}; fix it and rerun (finished work is kept), or use --keep-going")
                say(f"log: {log_path}")
                sys.exit(1)
        else:
            say(f"   {s.name} done in {(time.time() - t0) / 60:.1f} min")
    say(f"\nall done in {(time.time() - t_all) / 3600:.2f} h" + (f"; failed: {', '.join(failed)}" if failed else ""))
    say(f"reports: {data / 'player'} (*.html), {data / 'ideas'} (*.html)")
    say(f"log: {log_path}")


if __name__ == "__main__":
    main()
