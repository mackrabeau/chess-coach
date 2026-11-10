"""Project your games onto the learned ideas and compare with the peer population.

Two measures per idea k:
  leak_k     = 100 / N * sum over your mistakes of share_k(x) * winprob_loss(x)
               (win probability lost to idea k per 100 moves; N = all your analyzed moves)
  miss_k     = in *critical* positions where idea k is what separates the best move from the
               2nd best, how often you didn't play the best move (shrunk toward the peer rate)
Ideas are ranked by leak_you - leak_peers, with a bootstrap interval over your games.

    python scripts/05_build_deltas.py --set player     (first)
    python scripts/08_player_report.py
Output: data/player/player_report.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.ideas import IdeaModel, shares
from coach.report import feats_html, page, position_card
from coach.results import load_with_deltas


def mistake_codes(model: IdeaModel, df: pd.DataFrame, D: dict):
    """Codes for mistakes with usable lines. Returns (rows_index, W, S)."""
    m = ((df.label == "mistake") & ~df.played_is_best).to_numpy()
    W, keep, _ = model.encode((D["dB"] - D["dP"])[m])
    rows = np.flatnonzero(m)[keep]
    return rows, W, shares(W)


def leak(model, df, rows, S) -> np.ndarray:
    contrib = np.zeros(model.k)
    if len(rows):
        contrib = (S * df.wp_loss.to_numpy()[rows, None]).sum(0)
    return 100 * contrib / max(len(df), 1)


def leak_bootstrap(model, df, rows, S, n_boot: int, seed: int = 0):
    """Percentile interval for the player's leak, resampling whole games."""
    games, g_idx = np.unique(df.game_id.to_numpy(), return_inverse=True)
    n_moves = np.bincount(g_idx, minlength=len(games))
    contrib = np.zeros((len(games), model.k))
    if len(rows):
        np.add.at(contrib, g_idx[rows], S * df.wp_loss.to_numpy()[rows, None])
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, model.k))
    for b in range(n_boot):
        pick = rng.integers(0, len(games), len(games))
        boots[b] = 100 * contrib[pick].sum(0) / max(n_moves[pick].sum(), 1)
    return np.percentile(boots, 5, axis=0), np.percentile(boots, 95, axis=0)


def critical_stats(model, df, D):
    """Per idea: (n critical positions where it's dominant, misses)."""
    m = (df.critical & np.isfinite(D["dA"]).all(1)).to_numpy()
    W, keep, _ = model.encode((D["dB"] - D["dA"])[m])
    rows = np.flatnonzero(m)[keep]
    n = np.zeros(model.k, int)
    miss = np.zeros(model.k, int)
    if len(rows):
        dom = W.argmax(1)
        missed = ~df.played_is_best.to_numpy()[rows]
        np.add.at(n, dom, 1)
        np.add.at(miss, dom, missed.astype(int))
    return n, miss


def chesscom_links(cfg) -> dict[str, str]:
    links = {}
    for p in data_dir(cfg, "player", "archives").glob("*.json"):
        for g in json.loads(p.read_text(encoding="utf-8")).get("games", []):
            gid = g.get("uuid") or g.get("url", "").rstrip("/").split("/")[-1]
            if g.get("url"):
                links[gid] = g["url"]
    return links


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rcfg = cfg["report"]
    model = IdeaModel.load(data_dir(cfg, "ideas") / "model.pkl")

    pop, Dp, _ = load_with_deltas(cfg, "population")
    me, Dm, _ = load_with_deltas(cfg, "player")
    pr, _, pS = mistake_codes(model, pop, Dp)
    mr, mW, mS = mistake_codes(model, me, Dm)

    leak_pop = leak(model, pop, pr, pS)
    leak_me = leak(model, me, mr, mS)
    lo, hi = leak_bootstrap(model, me, mr, mS, rcfg["bootstrap"])
    diff = leak_me - leak_pop

    n_pop, miss_pop = critical_stats(model, pop, Dp)
    n_me, miss_me = critical_stats(model, me, Dm)
    rate_pop = (miss_pop + 1) / (n_pop + 2)
    s = rcfg["prior_strength"]
    rate_me = (miss_me + s * rate_pop) / (n_me + s)

    links = chesscom_links(cfg)
    # Stable ideas first; within each group, biggest gap to peers first.
    order = sorted(range(model.k), key=lambda k: (not model.is_stable(k), -diff[k]))
    weak = [k for k in order if diff[k] > 0][: rcfg["player_top_ideas"]]
    strong = sorted([k for k in range(model.k) if diff[k] < 0],
                    key=lambda k: (not model.is_stable(k), diff[k]))[:3]

    def verdict(k):
        if lo[k] > leak_pop[k]:
            return "<span class='down'>clear</span>"
        if hi[k] < leak_pop[k]:
            return "<span class='up'>clear</span>"
        return "<span class='muted'>within noise</span>"

    def stat_table(ks):
        rows = "".join(
            f"<tr><td><a href='#idea-{k}'>Idea {k}</a>{'' if model.is_stable(k) else ' <span class=muted>(unstable)</span>'}</td>"
            f"<td>{leak_me[k]:.2f} <span class='muted'>[{lo[k]:.2f}, {hi[k]:.2f}]</span></td>"
            f"<td>{leak_pop[k]:.2f}</td><td>{leak_me[k] / max(leak_pop[k], 1e-9):.1f}×</td>"
            f"<td>{verdict(k)}</td>"
            f"<td>{rate_me[k]:.0%} <span class='muted'>(n={n_me[k]})</span></td><td>{rate_pop[k]:.0%}</td></tr>"
            for k in ks)
        return ("<table><tr><th>idea</th><th>your leak (90% CI)</th><th>peers</th><th>ratio</th>"
                "<th>vs peers</th><th>your miss rate</th><th>peers</th></tr>" + rows + "</table>"
                "<p class='muted'>“clear” = the peer value is outside your 90% interval. "
                "“within noise” rows are leads, not conclusions.</p>")

    def idea_section(k):
        own = np.argsort(-(mW[:, k] * mS[:, k]))[: rcfg["player_examples"]]
        own = [i for i in own if mS[i, k] > 0.2]
        cards = []
        for i in own:
            r = me.iloc[mr[i]]
            cards.append(position_card(r, link=links.get(r.game_id)))
        drill_idx = np.argsort(-(pS[:, k]))[: rcfg["drills"]]
        drills = [position_card(pop.iloc[pr[i]], puzzle=True) for i in drill_idx if pS[i, k] > 0.3]
        return (
            f"<h2 id='idea-{k}'>Idea {k}</h2>"
            f"<p class='muted'>You lose {leak_me[k]:.2f}% win probability per 100 moves to this idea "
            f"vs {leak_pop[k]:.2f}% for peers. In critical positions where it decides the best move, "
            f"you miss it {rate_me[k]:.0%} of the time (peers {rate_pop[k]:.0%}, your n = {n_me[k]}).</p>"
            + feats_html(model.describe(k))
            + "<h3>Your games</h3><div class='grid'>" + "".join(cards) + "</div>"
            + "<h3>Practice positions (peers' games)</h3><div class='grid'>" + "".join(drills) + "</div>"
        )

    calib = (me.wp_loss.mean(), pop.wp_loss.mean())
    body = [
        "<h1>Your ideas report</h1>",
        f"<p class='muted'>{len(me):,} of your moves ({me.game_id.nunique():,} games) compared with "
        f"{len(pop):,} peer moves (Lichess rapid ≈ {model.meta.get('rating_center')}). "
        f"Mean win-prob loss per move: you {calib[0]:.2%}, peers {calib[1]:.2%}. "
        "Leak = win probability lost to an idea per 100 moves. These are patterns in where your "
        "losses concentrate relative to peers, not proven causes.</p>",
        "<h2>Where you lose more than peers</h2>", stat_table(weak) if weak else "<p>None.</p>",
        "<h2>Where you lose less than peers</h2>", stat_table(strong) if strong else "<p>None.</p>",
        *[idea_section(k) for k in weak],
    ]
    path = data_dir(cfg, "player") / "player_report.html"
    path.write_text(page("Your ideas report", "".join(body)), encoding="utf-8")
    print(f"your mistakes encoded: {len(mr):,}; wrote {path}")
    for k in weak:
        d = ", ".join(f"{a}{n}" for a, n, _ in model.describe(k, 3))
        print(f"  idea {k:2d}: leak {leak_me[k]:.2f} vs {leak_pop[k]:.2f} "
              f"[{lo[k]:.2f}, {hi[k]:.2f}]  {d}")


if __name__ == "__main__":
    main()
