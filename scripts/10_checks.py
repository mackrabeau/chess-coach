"""Three quick checks on existing data (no engine or Maia passes).

1. Does each loss metric track skill?  Slope of loss per 100 moves against rating, per
   game, with a game-bootstrap 90% interval, for: total, blunders, tactical, positional,
   decided. A metric that measures skill should fall as rating rises. Done on your whole
   history (chess.com ~1000-1900) and on the peers (Lichess 2000-2200).
2. Think time within difficulty: your error rate by think time *inside* each Maia-difficulty
   band (hard positions get long thinks, so the raw curve is confounded).
3. By phase: loss per 100 moves per phase x track, you (games >= min_player_rating) vs peers.
   Note: positions start at ply 20, so "opening" here only means moves 10-15.

    python scripts/10_checks.py --config configs/v0.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.difficulty import p_best_own
from coach.ideas import IdeaModel, move_tracks
from coach.results import load_with_deltas
from coach.stats import ci, game_bootstrap

TRACKS = ["blunder", "tactical", "positional", "decided"]


def per_game(df, tr) -> pd.DataFrame:
    """Per game: rating and loss per 100 moves, total and per track."""
    loss = df.wp_loss.to_numpy()
    d = pd.DataFrame({"game": df.game_id.to_numpy(), "rating": df.mover_rating.to_numpy(float),
                      "total": loss})
    for t in TRACKS:
        d[t] = loss * (tr == t)
    g = d.groupby("game").agg(rating=("rating", "median"), n=("total", "size"),
                               **{c: (c, "sum") for c in ["total"] + TRACKS})
    for c in ["total"] + TRACKS:
        g[c] = 100 * g[c] / g.n
    return g


def slopes(g: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Move-weighted least-squares slope of each metric on rating, per 100 rating points."""
    rng = np.random.default_rng(seed)
    x, w = g.rating.to_numpy(), g.n.to_numpy(float)

    def fit(idx, y):
        xx, yy, ww = x[idx], y[idx], w[idx]
        xm = np.average(xx, weights=ww)
        return 100 * np.sum(ww * (xx - xm) * (yy - np.average(yy, weights=ww))) / np.sum(ww * (xx - xm) ** 2)

    out = {}
    all_idx = np.arange(len(g))
    for c in ["total"] + TRACKS:
        y = g[c].to_numpy()
        b = [fit(rng.integers(0, len(g), len(g)), y) for _ in range(n_boot)]
        mean = np.average(y, weights=w)
        out[c] = (fit(all_idx, y), *np.percentile(b, [5, 95]), mean)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    B = cfg["report"]["bootstrap"]
    model = IdeaModel.load(data_dir(cfg, "ideas") / "model.pkl")
    pop, Dp, names = load_with_deltas(cfg, "population")
    me, Dm, _ = load_with_deltas(cfg, "player")
    tr_pop, tr_me = move_tracks(model, cfg, pop, Dp, names), move_tracks(model, cfg, me, Dm, names)

    # 1. does each metric track skill?
    print("1. Slope of loss per 100 moves per +100 rating (90% CI); a skill metric should be clearly < 0")
    for label, df, tr in (("you, whole history", me, tr_me), ("peers, Lichess 2000-2200", pop, tr_pop)):
        g = per_game(df, tr)
        s = slopes(g)
        print(f"  {label}: {len(g):,} games, rating {g.rating.min():.0f}-{g.rating.max():.0f}")
        for c, (b, lo, hi, mean) in s.items():
            rel = b / max(mean, 1e-9)
            tag = "falls with rating" if hi < 0 else ("rises with rating" if lo > 0 else "no clear trend")
            print(f"    {c:10s} mean {mean:5.2f}  slope {b:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"({rel:+.1%} of the mean per 100 points)  {tag}")

    # 2. think time within difficulty (you, whole history)
    an = data_dir(cfg, "player") / "anatomy.parquet"
    d = p_best_own(cfg, me, "player")
    if an.exists() and d is not None:
        T = me[["pos_id"]].merge(pd.read_parquet(an)[["pos_id", "think_s"]].drop_duplicates("pos_id"),
                                 on="pos_id", how="left").think_s.to_numpy(float)
        err = (me.label.isin(["mistake", "blunder"]) & ~me.played_is_best).to_numpy()
        print("\n2. Your error rate by think time, within Maia-difficulty bands (whole history)")
        tb = [(0, 10, "< 10 s"), (10, 30, "10-30 s"), (30, 90, "30-90 s"), (90, 1e9, "90 s +")]
        db = [(0.0, 0.2, "hard (Maia finds best < 20%)"), (0.2, 0.5, "medium (20-50%)"),
              (0.5, 1.01, "easy (>= 50%)")]
        for a, b, name in db:
            band = np.isfinite(d) & (d >= a) & (d < b) & np.isfinite(T)
            cells = []
            for t0, t1, tname in tb:
                m = band & (T >= t0) & (T < t1)
                if m.sum() < 50:
                    cells.append(f"{tname}: n<50")
                    continue
                bt = game_bootstrap(me, np.arange(len(me)), np.stack([m, m & err], 1).astype(float), B)
                lo, hi = ci(bt[:, 1] / np.maximum(bt[:, 0], 1e-9))
                cells.append(f"{tname}: {(m & err).sum() / m.sum():.1%} [{lo:.1%}, {hi:.1%}] (n {m.sum():,})")
            print(f"  {name}:\n    " + "\n    ".join(cells))
    else:
        print("\n2. skipped (needs 09b and 09c)")

    # 3. by phase, you (>= min rating) vs peers
    keep = (me.mover_rating >= cfg["player"]["min_player_rating"]).to_numpy()
    mc, trc = me[keep].reset_index(drop=True), tr_me[keep]
    print(f"\n3. Loss per 100 moves by phase (you: games >= {cfg['player']['min_player_rating']}; "
          "'opening' = moves 10-15 only)")
    for ph in ("opening", "middlegame", "endgame"):
        pm, pp = (mc.phase == ph).to_numpy(), (pop.phase == ph).to_numpy()
        line = [f"  {ph:10s} share of moves {pm.mean():4.0%} / {pp.mean():4.0%}"]
        for t in ["all"] + TRACKS:
            sel_m = pm & ((trc == t) if t != "all" else True)
            sel_p = pp & ((tr_pop == t) if t != "all" else True)
            # loss per 100 moves *within the phase*
            v_m = 100 * mc.wp_loss.to_numpy()[sel_m].sum() / max(pm.sum(), 1)
            v_p = 100 * pop.wp_loss.to_numpy()[sel_p].sum() / max(pp.sum(), 1)
            bt = game_bootstrap(mc, np.arange(len(mc)),
                                np.stack([pm, mc.wp_loss.to_numpy() * sel_m], 1).astype(float), B)
            lo, hi = ci(100 * bt[:, 1] / np.maximum(bt[:, 0], 1e-9))
            flag = " *" if (lo > v_p or hi < v_p) else ""
            line.append(f"{t} {v_m:.2f} [{lo:.2f}, {hi:.2f}] vs {v_p:.2f}{flag}")
        print("  |  ".join(line))
    print("  (* = peer value outside your 90% interval; per 100 moves *within that phase*)")


if __name__ == "__main__":
    main()
