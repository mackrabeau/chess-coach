"""HTML report of every learned idea with its strongest population examples.

This is the go/no-go check: for each stable idea, do the top positions visibly share
a chess idea? Green arrow = engine's best move, red = move played.

    python scripts/07_inspect_ideas.py
Output: data/ideas/ideas_report.html (open in a browser)
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.ideas import IdeaModel, shares
from coach.report import feats_html, page, position_card
from coach.results import load_analyzed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = data_dir(cfg, "ideas")
    model = IdeaModel.load(out / "model.pkl")
    z = np.load(out / "pop_codes.npz")
    W = z["W"]

    df = load_analyzed(cfg, "population").set_index("pos_id")
    d = np.load(data_dir(cfg, "population") / "deltas.npz")
    df["phase"] = pd.Series(d["phase"], index=d["pos_id"])
    sub = df.loc[z["pos_id"]].reset_index()

    S = shares(W)
    dominant = S.argmax(1)
    mass = W.sum(0)
    order = sorted(range(model.k), key=lambda k: (not model.is_stable(k), -mass[k]))
    n_stable = sum(model.is_stable(k) for k in range(model.k))

    top_n = cfg["report"]["top_positions"]
    parts = [
        "<h1>Ideas learned from peer mistakes</h1>",
        f"<p class='muted'>{len(sub):,} positional population mistakes (Lichess rapid around "
        f"{model.meta.get('rating_center')}; best-move win prob in {model.meta.get('contested')}, "
        f"settled line ends within {model.meta.get('tactical_material')} pawns of material), "
        f"K = {model.k} ideas, {n_stable} stable across "
        f"split halves (cos ≥ {model.stable_cos}). Each idea is a pattern in how the position at the "
        f"end of the engine's line differs from the end of the played line. ↑ = the best line "
        f"ends with more of this than the played line, ↓ = less. Green arrow = best move, red = played.</p>",
        "<p class='muted'>Judge each idea by its boards, not its label: if most of the top positions "
        "share a recognizable idea, it's real.</p>",
    ]
    for k in order:
        members = dominant == k
        score = W[:, k] * S[:, k]
        top = np.argsort(-score)[:top_n]
        stable = model.is_stable(k)
        ph = sub.phase[members].value_counts(normalize=True)
        ph_txt = ", ".join(f"{p} {v:.0%}" for p, v in ph.items()) or "—"
        parts.append(
            f"<h2 id='idea-{k}'>Idea {k}</h2>"
            f"<div><span class='tag{' stable' if stable else ''}'>"
            f"{'stable' if stable else 'unstable'} · cos {model.stability[k]:.2f}</span>"
            f"<span class='muted'>dominant in {members.sum():,} positions "
            f"({members.mean():.1%}) · mean loss {sub.wp_loss[members].mean():.1%} · {html.escape(ph_txt)}"
            f"</span></div>"
            + feats_html(model.describe(k))
            + "<div class='grid'>"
            + "".join(position_card(sub.iloc[i]) for i in top if S[i, k] > 0.2)
            + "</div>"
        )
    path = out / "ideas_report.html"
    path.write_text(page("Ideas report", "".join(parts)), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
