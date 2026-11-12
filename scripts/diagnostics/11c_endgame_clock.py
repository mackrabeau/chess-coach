"""Is the balanced-endgame gap a clock effect?

Your games are mostly chess.com 10|0 (no increment); peers are Lichess rapid, often with
an increment. Endgames come late, so you may simply be playing them with less time.

For contested positions (best-move win prob in endgames.balanced, default [0.2, 0.8]),
endgame phase, plus the middlegame as a control:
  0. time controls in both sets (share of positions with increment 0).
  1. clock left: share of moves per clock bin, you vs peers vs peers without increment.
  2. error rate and loss per 100 moves per clock bin (you with 90% CI over games).
  3. clock-matched comparison: peers reweighted to YOUR mix of clock bins
     (direct standardization). If the gap closes, it's time, not endgame skill.
  4. your own fast moves (< 10 s) per clock bin (from 09c think times, if present).
Positions with < fetch.min_clock_seconds (30 s) were never stored, so time scrambles
below that aren't in either set.

    python scripts/diagnostics/11c_endgame_clock.py --config configs/v0.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess
import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.handcrafted import phase
from coach.results import load_analyzed
from coach.stats import ci, game_bootstrap

BINS = [(0, 60, "< 1 min"), (60, 120, "1-2 min"), (120, 180, "2-3 min"), (180, 300, "3-5 min"),
        (300, 1e9, "5 min +")]


def contested(cfg, which):
    df = load_analyzed(cfg, which)
    lo, hi = cfg.get("endgames", {}).get("balanced", [0.2, 0.8])
    df = df[(df.wp_best >= lo) & (df.wp_best <= hi) & df.clock_before.notna()].reset_index(drop=True)
    df["phase"] = [phase(chess.Board(f)) for f in df.fen]
    df["err"] = (df.label.isin(["mistake", "blunder"]) & ~df.played_is_best).to_numpy()
    df["cbin"] = -1
    for i, (a, b, _) in enumerate(BINS):
        df.loc[(df.clock_before >= a) & (df.clock_before < b), "cbin"] = i
    return df


def rates(df):
    """Per clock bin: (moves, error rate, loss per 100 moves)."""
    g = df.groupby("cbin")
    return pd.DataFrame({"n": g.size(), "err": g.err.mean(), "loss": 100 * g.wp_loss.mean()}).reindex(range(len(BINS)))


def you_ci(df, B):
    """Your error rate and loss/100 with game-bootstrap CIs, overall and per bin."""
    vals = np.stack([df.err.to_numpy(float), df.wp_loss.to_numpy(float)], 1)
    bt = game_bootstrap(df, np.arange(len(df)), vals, B)       # 100 * sum / moves
    lo, hi = ci(bt)
    return (df.err.mean(), lo[0] / 100, hi[0] / 100), (100 * df.wp_loss.mean(), lo[1], hi[1])


def standardized(me, pop):
    """Peers' error rate / loss reweighted to your clock-bin mix (bins where peers have >= 20 moves)."""
    w = me.cbin.value_counts(normalize=True)
    r = rates(pop)
    ok = [b for b in w.index if b >= 0 and r.n.get(b, 0) >= 20]
    ww = w[ok] / w[ok].sum()
    return float((ww * r.err[ok]).sum()), float((ww * r.loss[ok]).sum()), float(w[ok].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    B = cfg["report"]["bootstrap"]
    cmp_min = cfg["player"]["min_player_rating"]
    me_all, pop_all = contested(cfg, "player"), contested(cfg, "population")
    me_all = me_all[me_all.mover_rating >= cmp_min].reset_index(drop=True)

    print(f"0. time controls (contested positions): you {(me_all.increment == 0).mean():.0%} without increment "
          f"(base: {me_all.base_seconds.value_counts(normalize=True).head(3).round(2).to_dict()}); "
          f"peers {(pop_all.increment == 0).mean():.0%} without increment")
    print(f"   peer increments: {pop_all.increment.value_counts(normalize=True).head(5).round(2).to_dict()}")

    for ph in ("endgame", "middlegame"):
        me = me_all[me_all.phase == ph].reset_index(drop=True)
        pop = pop_all[pop_all.phase == ph].reset_index(drop=True)
        pop0 = pop[pop.increment == 0].reset_index(drop=True)
        print(f"\n===== contested {ph}: you {len(me):,} moves / {me.game_id.nunique()} games (>= {cmp_min}); "
              f"peers {len(pop):,} (no increment: {len(pop0):,}) =====")
        rm, rp, r0 = rates(me), rates(pop), rates(pop0)
        print("1-2. per clock bin: share of moves | error rate | loss per 100 moves")
        print(f"   {'clock left':10s} {'you':>28s}   {'peers':>24s}   {'peers, no increment':>24s}")
        for i, (_, _, name) in enumerate(BINS):
            def cell(r, d):
                n = r.n.get(i)
                if not n or np.isnan(n) or n < 20:
                    return f"{'(n < 20)':>24s}"
                return f"{n / len(d):4.0%} | {r.err[i]:5.1%} | {r.loss[i]:5.2f}"
            sub = me[me.cbin == i]
            extra = ""
            if len(sub) >= 20 and sub.game_id.nunique() >= 5:
                (e, elo, ehi), _ = you_ci(sub, B)
                extra = f" [{elo:.0%}, {ehi:.0%}]"
            print(f"   {name:10s} {cell(rm, me) + extra:>28s}   {cell(rp, pop)}   {cell(r0, pop0)}")
        (e, elo, ehi), (l, llo, lhi) = you_ci(me, B)
        print(f"3. overall: you error {e:.1%} [{elo:.1%}, {ehi:.1%}], loss {l:.2f} [{llo:.2f}, {lhi:.2f}] per 100 moves")
        print(f"   peers raw:                         error {pop.err.mean():.1%}, loss {100 * pop.wp_loss.mean():.2f}")
        for name, p in (("peers matched to your clock", pop), ("peers no-increment, clock-matched", pop0)):
            if len(p) < 100:
                print(f"   {name}: too few moves ({len(p)})")
                continue
            se, sl, cov = standardized(me, p)
            verdict = []
            for lab, val, lo_, hi_ in (("error", se, elo, ehi), ("loss", sl, llo, lhi)):
                verdict.append(f"{lab} {'clear gap' if val < lo_ else ('you better' if val > hi_ else 'within noise')}")
            print(f"   {name + ':':35s} error {se:.1%}, loss {sl:.2f}  ({', '.join(verdict)}; "
                  f"covers {cov:.0%} of your moves)")

        an = data_dir(cfg, "player") / "anatomy.parquet"
        if an.exists():
            t = me[["pos_id", "cbin", "err"]].merge(
                pd.read_parquet(an)[["pos_id", "think_s"]].drop_duplicates("pos_id"), on="pos_id", how="left")
            t = t[t.think_s.notna()]
            print("4. your think time per clock bin: share of moves < 10 s | error rate on those | on the rest")
            for i, (_, _, name) in enumerate(BINS):
                s = t[t.cbin == i]
                if len(s) < 20:
                    continue
                fast = s.think_s < 10
                rest = f"{s.err[~fast].mean():5.1%}" if (~fast).sum() >= 10 else "  n/a"
                print(f"   {name:10s} {fast.mean():4.0%} | {s.err[fast].mean():5.1%} | {rest}  (n {len(s)})")


if __name__ == "__main__":
    main()
