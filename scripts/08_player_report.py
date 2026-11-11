"""Project your games onto the learned ideas and compare with the peer population.

Loss breakdown (per 100 moves): blunders, and mistakes split into the same tracks 06 uses
(decided position / tactical / positional).

Per positional idea k:
  leak_k     = 100 / N * sum over your positional mistakes of share_k(x) * winprob_loss(x)
               (win probability lost to idea k per 100 moves; N = all your analyzed moves)
  rel_k      = leak_k / sum_j leak_j: idea k's share of your positional losses. Comparing
               rel_k with the peers' removes overall strength, so it shows your *profile*:
               rel_you > rel_peers = weakness, < = strength.
  miss_k     = in *critical* positions where idea k is what separates the best move from the
               2nd best, how often you didn't play the best move (shrunk toward the peer rate)
Bootstrap intervals resample your games.

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
from coach.ideas import IdeaModel, shares, track
from coach.report import feats_html, page, position_card
from coach.results import load_with_deltas


def track_cfg(model: IdeaModel, cfg: dict) -> dict:
    """The routing the model was trained with (falls back to the config for old models)."""
    m, i = model.meta, cfg["ideas"]
    return {"contested": m.get("contested", i["contested"]),
            "tactical_material": m.get("tactical_material", i["tactical_material"])}


def tracks(model, cfg, df, D, names) -> np.ndarray:
    """Per move: ok / blunder / decided / tactical / positional / unusable."""
    tr = track(df.wp_best.to_numpy(), D["dB"] - D["dP"], names, track_cfg(model, cfg))
    out = np.where(tr == "", "unusable", tr).astype(object)
    out[(df.label == "ok").to_numpy() | df.played_is_best.to_numpy()] = "ok"
    out[(df.label == "blunder").to_numpy() & ~df.played_is_best.to_numpy()] = "blunder"
    return out


def mistake_codes(model: IdeaModel, df: pd.DataFrame, D: dict, tr: np.ndarray):
    """Codes for positional mistakes. Returns (rows_index, W, S)."""
    m = tr == "positional"
    W, keep, _ = model.encode((D["dB"] - D["dP"])[m])
    rows = np.flatnonzero(m)[keep]
    return rows, W, shares(W)


def leak(model, df, rows, S) -> np.ndarray:
    contrib = np.zeros(model.k)
    if len(rows):
        contrib = (S * df.wp_loss.to_numpy()[rows, None]).sum(0)
    return 100 * contrib / max(len(df), 1)


def game_bootstrap(df, rows, values: np.ndarray, n_boot: int, seed: int = 0) -> np.ndarray:
    """Resample whole games. values: (len(rows), m) per-move contributions.
    Returns (n_boot, m) of 100 * sum(values) / moves."""
    games, g_idx = np.unique(df.game_id.to_numpy(), return_inverse=True)
    n_moves = np.bincount(g_idx, minlength=len(games))
    contrib = np.zeros((len(games), values.shape[1]))
    if len(rows):
        np.add.at(contrib, g_idx[rows], values)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, values.shape[1]))
    for b in range(n_boot):
        pick = rng.integers(0, len(games), len(games))
        boots[b] = 100 * contrib[pick].sum(0) / max(n_moves[pick].sum(), 1)
    return boots


def ci(boots: np.ndarray):
    return np.percentile(boots, 5, axis=0), np.percentile(boots, 95, axis=0)


TRACKS = ["blunder", "tactical", "positional", "decided", "ok", "unusable"]
TRACK_TEXT = {
    "blunder": "blunders (> 20% win prob)",
    "tactical": "tactical mistakes (material ends ≥ 2 apart)",
    "positional": "positional mistakes (→ ideas below)",
    "decided": "mistakes in already-decided positions",
    "ok": "small inaccuracies (< 3% each)",
    "unusable": "mistakes with unusable lines",
}


def breakdown(df, tr) -> np.ndarray:
    """Win prob lost per 100 moves, per track."""
    loss = df.wp_loss.to_numpy()
    return np.array([100 * loss[tr == t].sum() / max(len(df), 1) for t in TRACKS])


def critical_stats(model, cfg, df, D, names):
    """Per idea: (n critical positions where it's dominant, misses). Only positions routed
    'positional' when comparing the best line with the 2nd-best line."""
    tr = track(df.wp_best.to_numpy(), D["dB"] - D["dA"], names, track_cfg(model, cfg))
    m = df.critical.to_numpy() & (tr == "positional")
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

    pop, Dp, names = load_with_deltas(cfg, "population")
    me, Dm, _ = load_with_deltas(cfg, "player")
    tr_pop, tr_me = tracks(model, cfg, pop, Dp, names), tracks(model, cfg, me, Dm, names)
    pr, _, pS = mistake_codes(model, pop, Dp, tr_pop)
    mr, mW, mS = mistake_codes(model, me, Dm, tr_me)
    B = rcfg["bootstrap"]

    # Loss breakdown by track
    bd_pop, bd_me = breakdown(pop, tr_pop), breakdown(me, tr_me)
    onehot = np.stack([(tr_me == t) for t in TRACKS], 1) * me.wp_loss.to_numpy()[:, None]
    bd_lo, bd_hi = ci(game_bootstrap(me, np.arange(len(me)), onehot, B))

    # Per idea: absolute leak and relative share of positional losses
    leak_pop = leak(model, pop, pr, pS)
    leak_me = leak(model, me, mr, mS)
    boots = game_bootstrap(me, mr, mS * me.wp_loss.to_numpy()[mr, None], B)
    lo, hi = ci(boots)
    rel_pop = leak_pop / max(leak_pop.sum(), 1e-12)
    rel_me = leak_me / max(leak_me.sum(), 1e-12)
    rel_lo, rel_hi = ci(boots / np.maximum(boots.sum(1, keepdims=True), 1e-12))
    diff = rel_me - rel_pop

    n_pop, miss_pop = critical_stats(model, cfg, pop, Dp, names)
    n_me, miss_me = critical_stats(model, cfg, me, Dm, names)
    rate_pop = (miss_pop + 1) / (n_pop + 2)
    s = rcfg["prior_strength"]
    rate_me = (miss_me + s * rate_pop) / (n_me + s)

    links = chesscom_links(cfg)
    # Stable ideas only; biggest relative gap first.
    stable = [k for k in range(model.k) if model.is_stable(k)]
    n_show = rcfg["player_top_ideas"]
    weak = sorted([k for k in stable if diff[k] > 0], key=lambda k: -diff[k])[:n_show]
    strong = sorted([k for k in stable if diff[k] < 0], key=lambda k: diff[k])[:n_show]

    def verdict(k):
        if rel_lo[k] > rel_pop[k]:
            return "<span class='down'>clear</span>"
        if rel_hi[k] < rel_pop[k]:
            return "<span class='up'>clear</span>"
        return "<span class='muted'>within noise</span>"

    def name(k):
        return html.escape(", ".join(f"{a}{n}" for a, n, _ in model.describe(k, 2)))

    def stat_table(ks):
        rows = "".join(
            f"<tr><td><a href='#idea-{k}'>Idea {k}</a> <span class='muted'>{name(k)}</span></td>"
            f"<td>{rel_me[k]:.0%} <span class='muted'>[{rel_lo[k]:.0%}, {rel_hi[k]:.0%}]</span></td>"
            f"<td>{rel_pop[k]:.0%}</td><td>{verdict(k)}</td>"
            f"<td>{leak_me[k]:.2f} <span class='muted'>[{lo[k]:.2f}, {hi[k]:.2f}]</span></td>"
            f"<td>{leak_pop[k]:.2f}</td>"
            f"<td>{rate_me[k]:.0%} <span class='muted'>(n={n_me[k]})</span></td><td>{rate_pop[k]:.0%}</td></tr>"
            for k in ks)
        return ("<table><tr><th>idea</th><th>share of your positional loss (90% CI)</th><th>peers</th>"
                "<th>vs peers</th><th>leak / 100 moves</th><th>peers</th>"
                "<th>critical miss rate</th><th>peers</th></tr>" + rows + "</table>")

    def breakdown_table():
        rows = "".join(
            f"<tr><td>{TRACK_TEXT[t]}</td><td>{bd_me[i]:.2f} <span class='muted'>"
            f"[{bd_lo[i]:.2f}, {bd_hi[i]:.2f}]</span></td><td>{bd_pop[i]:.2f}</td>"
            f"<td>{bd_me[i] / max(bd_pop[i], 1e-9):.2f}×</td></tr>"
            for i, t in enumerate(TRACKS))
        return ("<table><tr><th>where the win probability goes (per 100 moves)</th>"
                "<th>you (90% CI)</th><th>peers</th><th>ratio</th></tr>" + rows + "</table>")

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
        "These are patterns in where your losses concentrate relative to peers, not proven causes.</p>",
        "<h2>Where your losses come from</h2>", breakdown_table(),
        "<p class='muted'>Decided = the best move's win probability was outside "
        f"{track_cfg(model, cfg)['contested']}. Only positional mistakes feed the ideas below.</p>",
        "<h2>Weaknesses: ideas that take a bigger share of your losses than peers'</h2>",
        stat_table(weak) if weak else "<p>None among stable ideas.</p>",
        "<h2>Strengths: ideas where you can trust your instincts</h2>",
        stat_table(strong) if strong else "<p>None among stable ideas.</p>",
        "<p class='muted'>Share = this idea's part of your positional losses, so it compares your "
        "profile with peers' independent of overall strength. “clear” = the peer share is outside "
        "your 90% interval; “within noise” rows are leads, not conclusions.</p>",
        *[idea_section(k) for k in weak],
    ]
    path = data_dir(cfg, "player") / "player_report.html"
    path.write_text(page("Your ideas report", "".join(body)), encoding="utf-8")
    print(f"your positional mistakes encoded: {len(mr):,}; wrote {path}")
    print("loss per 100 moves, you vs peers:")
    for i, t in enumerate(TRACKS):
        print(f"  {t:10s} {bd_me[i]:5.2f} [{bd_lo[i]:.2f}, {bd_hi[i]:.2f}]  vs {bd_pop[i]:5.2f}")
    for title, ks in (("weaknesses", weak), ("strengths", strong)):
        print(f"{title} (share of positional loss, you vs peers):")
        for k in ks:
            v = "CLEAR" if (rel_lo[k] > rel_pop[k] or rel_hi[k] < rel_pop[k]) else "noise"
            print(f"  idea {k:2d}: {rel_me[k]:.0%} [{rel_lo[k]:.0%}, {rel_hi[k]:.0%}] vs {rel_pop[k]:.0%}  "
                  f"{v:5s}  {', '.join(f'{a}{n}' for a, n, _ in model.describe(k, 3))}")


if __name__ == "__main__":
    main()
