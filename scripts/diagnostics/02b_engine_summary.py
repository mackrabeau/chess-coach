"""Summarize engine results and compare the player to the population.

This is also the rating-band calibration check: if the player's loss profile is far
from the population's, shift population.rating_center and re-mine.

    python scripts/diagnostics/02b_engine_summary.py --config configs/v0.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from coach.config import load_config
from coach.results import load_analyzed


def profile(df: pd.DataFrame) -> dict:
    n = len(df)
    crit = df[df.critical]
    return {
        "positions": n,
        "mean_wp_loss": df.wp_loss.mean(),
        "median_wp_loss": df.wp_loss.median(),
        "best_move_rate": df.played_is_best.mean(),
        "mistake_rate": (df.label == "mistake").mean(),
        "blunder_rate": (df.label == "blunder").mean(),
        "critical_rate": df.critical.mean(),
        "critical_found_rate": crit.played_is_best.mean() if len(crit) else float("nan"),
        "complete_best_lines": df.best_line_complete.mean(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cols = {}
    for which in ("population", "player"):
        try:
            cols[which] = profile(load_analyzed(cfg, which))
        except FileNotFoundError as e:
            print(f"[{which}] {e}")
    if not cols:
        return
    table = pd.DataFrame(cols)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table)
    if len(cols) == 2:
        d = cols["player"]["mean_wp_loss"] - cols["population"]["mean_wp_loss"]
        print(f"\nplayer - population mean win-prob loss: {d:+.4f}")
        print("Negative: you lose less per move than this band -> consider raising rating_center. "
              "Positive: consider lowering it. Check blunder_rate moves the same way before shifting.")


if __name__ == "__main__":
    main()
