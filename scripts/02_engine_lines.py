"""Run Stockfish over decision positions. Resumable: safe to kill and restart.

Writes one parquet part per chunk to data/<set>/engine/part-*.parquet; on restart,
positions already present in any part are skipped. --set is player, population, or any
mined band (band_1600, ... from 17_multiband.py).

    python scripts/02_engine_lines.py --config configs/v0.yaml --set population
    python scripts/02_engine_lines.py --config configs/v0.yaml --set player
    python scripts/02_engine_lines.py --config configs/v0.yaml --set population --limit 200   # quick test
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach.config import load_config
from coach.runner import engine_pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--set", required=True, help="player, population, or a band set (band_1600, ...)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N positions (testing)")
    ap.add_argument("--workers", type=int, default=None, help="override engine.workers")
    args = ap.parse_args()
    engine_pass(load_config(args.config), args.set, args.limit, args.workers)


if __name__ == "__main__":
    mp.freeze_support()
    main()
