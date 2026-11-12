"""Inspection report for the tactical / blunder track: peer examples per category.

Each blunder or tactical mistake gets a direction (missed = the best line wins >= 2 pawns
you didn't take; allowed = your move lets the opponent win >= 2) and the first motif the
heuristics detect (coach/motifs.py). This page shows the worst peer examples per category
so the detectors can be checked by eye before trusting the player report's tactics table.

    python scripts/07b_inspect_tactics.py
Output: data/ideas/tactics_report.html
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from coach.config import data_dir, load_config
from coach.ideas import IdeaModel, move_tracks
from coach.motifs import tactic_labels
from coach.report import page, position_card
from coach.results import load_deep, load_with_deltas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--per-category", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = IdeaModel.load(data_dir(cfg, "ideas") / "model.pkl")

    df, D, names = load_with_deltas(cfg, "population")
    tr = move_tracks(model, cfg, df, D, names)
    tl = tactic_labels(df, tr, load_deep(cfg, "population"))
    counts = tl.category.value_counts()
    loss = df.wp_loss.to_numpy()

    parts = [
        "<h1>Tactics: peer examples per category</h1>",
        f"<p class='muted'>{len(tl):,} peer blunders and tactical mistakes. Direction: “missed” = the "
        "best line wins ≥ 2 pawns of material you didn't take; “allowed” = your move lets the "
        "opponent win ≥ 2. Motif = the first one the heuristics detect in that side's moves. Check "
        "that each category's boards actually show its motif. Green arrow = best move, red = played.</p>",
        "<table><tr><th>category</th><th>n</th><th>share</th><th>win prob lost per 100 moves</th></tr>"
        + "".join(f"<tr><td><a href='#c{i}'>{html.escape(c)}</a></td><td>{n}</td>"
                  f"<td>{n / len(tl):.0%}</td>"
                  f"<td>{100 * loss[tl.index[tl.category == c]].sum() / len(df):.2f}</td></tr>"
                  for i, (c, n) in enumerate(counts.items()))
        + "</table>",
    ]
    for i, (c, n) in enumerate(counts.items()):
        ids = tl.index[tl.category == c]
        ids = ids[np.argsort(-loss[ids])][: args.per_category]
        sub = tl.loc[ids]
        cards = "".join(
            "<div>" + position_card(df.iloc[j])
            + f"<div class='muted' style='font-size:12px;margin-top:4px'>missed: "
              f"{html.escape(', '.join(sub.motifs_missed[j]) or '—')} · allowed: "
              f"{html.escape(', '.join(sub.motifs_allowed[j]) or '—')} · "
              f"best line {sub.gain[j]:+d}, played line {sub.loss[j]:+d} material"
              f"{' · found by the deep pass (06c)' if sub.deep[j] else ''}</div></div>"
            for j in ids)
        parts.append(f"<h2 id='c{i}'>{html.escape(c)} <span class='muted'>({n})</span></h2>"
                     f"<div class='grid'>{cards}</div>")
    path = data_dir(cfg, "ideas") / "tactics_report.html"
    path.write_text(page("Tactics report", "".join(parts), back=None), encoding="utf-8")
    print(f"{len(tl):,} peer blunders + tactical mistakes ({tl.deep.sum():,} classified via deep lines); categories:")
    for c, n in counts.items():
        print(f"  {c:34s} {n:5d}  ({n / len(tl):.0%})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
