"""Skill profile: at what level do you play in each kind of position? (Maia best-fit Elo per category)

09b stored, for every position, log P(played move | Maia at Elo e) on maia.grid. For a set of
positions, the Elo that maximizes the summed log-likelihood is the level whose moves look most
like yours. Per category X we fit an OFFSET c_X relative to each game's own level (you: your
chess.com rating then, mapped to Maia's scale; peers: their Lichess rating), so your whole history
counts even though your rating changed.

Maia may fit some categories higher for everyone (e.g. easy endgames), so each category is
calibrated on peers:

    relative level_X = (c_X,you - c_all,you) - (c_X,peers - c_all,peers)

= how much stronger (+) or weaker (-) you play in X than in your games overall, beyond what's
normal in X. 90% CI from a game bootstrap of your side (peers treated as fixed, they're larger).
"clear" = CI excludes 0. Reminder: Maia Elo measures how much your move CHOICES look like a
player of that level, not engine accuracy (validated: tracks peers' real rating, slope ~1.27).

    python scripts/16_skill_profile.py --config configs/v0.yaml

Output: stdout + data/player/skill_profile.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.difficulty import at_elo, grid_of, maia_dir, own_elo, p_best_own
from coach.endgame_features import material_class
from coach.handcrafted import phase
from coach.parts import read_parts
from coach.results import load_analyzed

OFFSETS = np.arange(-500, 501, 10)


def load(cfg, which):
    df = load_analyzed(cfg, which)
    q = read_parts(maia_dir(cfg, which))
    if q is None:
        sys.exit(f"no 09b curves for {which}; run 09b_maia_difficulty.py first")
    df = df.merge(q[["pos_id", "p_played"]], on="pos_id", how="inner").reset_index(drop=True)
    boards = [chess.Board(f) for f in df.fen]
    df["ph"] = [phase(b) for b in boards]
    df["etype"] = [material_class(b) if p == "endgame" else "" for b, p in zip(boards, df.ph)]
    bm = [chess.Move.from_uci(m) for m in df.best_move]
    df["best_kind"] = ["capture" if b.is_capture(m) else ("check" if b.gives_check(m) else "quiet")
                       for b, m in zip(boards, bm)]
    df["pbest"] = p_best_own(cfg, df, which)
    df["err"] = (df.label.isin(["mistake", "blunder"]) & ~df.played_is_best).to_numpy()
    return df


def categories(df):
    """name -> boolean mask."""
    wp, c = df.wp_best.to_numpy(), df.clock_before.to_numpy(float)
    cats = {
        "phase: opening (moves 10-15)": df.ph == "opening",
        "phase: middlegame": df.ph == "middlegame",
        "phase: endgame": df.ph == "endgame",
        "eval: losing (< 0.2)": wp < 0.2,
        "eval: contested (0.2-0.8)": (wp >= 0.2) & (wp <= 0.8),
        "eval: winning (> 0.8)": wp > 0.8,
        "critical (one good move)": df.critical.astype(bool),
        "not critical": ~df.critical.astype(bool),
        "best move is a capture": df.best_kind == "capture",
        "best move is a check": df.best_kind == "check",
        "best move is quiet": df.best_kind == "quiet",
        "difficulty: hard (Maia finds best < 20%)": df.pbest < 0.2,
        "difficulty: medium (20-50%)": (df.pbest >= 0.2) & (df.pbest < 0.5),
        "difficulty: easy (>= 50%)": df.pbest >= 0.5,
        "clock: < 1 min": c < 60,
        "clock: 1-3 min": (c >= 60) & (c < 180),
        "clock: 3 min +": c >= 180,
    }
    for t in ("rook endings", "minor-piece endings", "pawn endings", "queen endings", "mixed"):
        cats[f"endgame: {t}"] = df.etype == t
    cats["contested middlegame"] = (df.ph == "middlegame") & (wp >= 0.2) & (wp <= 0.8)
    cats["contested endgame"] = (df.ph == "endgame") & (wp >= 0.2) & (wp <= 0.8)
    # clock confound: endgames come late, when your clock is low
    slow = c >= 180
    cats["endgame, clock 3 min +"] = (df.ph == "endgame") & slow
    cats["rook endings, clock 3 min +"] = (df.etype == "rook endings") & slow
    cats["minor-piece endings, clock 3 min +"] = (df.etype == "minor-piece endings") & slow
    cats["contested endgame, clock 3 min +"] = (df.ph == "endgame") & (wp >= 0.2) & (wp <= 0.8) & slow
    cats["winning (> 0.8), clock 3 min +"] = (wp > 0.8) & slow
    cats["middlegame, clock < 3 min"] = (df.ph == "middlegame") & ~slow
    cats["middlegame, clock 3 min +"] = (df.ph == "middlegame") & slow
    # critical positions (one good move): where is the only move missed?
    crit = df.critical.astype(bool).to_numpy()
    cats["critical: best move quiet"] = crit & (df.best_kind == "quiet")
    cats["critical: best move capture/check"] = crit & (df.best_kind != "quiet")
    cats["critical: defending (wp_best < 0.5)"] = crit & (wp < 0.5)
    cats["critical: pressing (wp_best >= 0.5)"] = crit & (wp >= 0.5)
    cats["critical: contested (0.2-0.8)"] = crit & (wp >= 0.2) & (wp <= 0.8)
    cats["critical: middlegame"] = crit & (df.ph == "middlegame")
    cats["critical: endgame"] = crit & (df.ph == "endgame")
    cats["critical: clock 3 min +"] = crit & slow
    return {k: np.asarray(v, bool) for k, v in cats.items()}


def per_game_ll(df, cfg, which):
    """(games x offsets) summed log-lik per game for every offset, per row (rows x offsets)."""
    grid = grid_of(cfg["maia"])
    L = np.log(np.clip(np.stack([np.asarray(v, float) for v in df.p_played]), 1e-9, 1))
    base = own_elo(df, which, cfg["maia"])
    return np.stack([at_elo(L, base + o, grid) for o in OFFSETS], 1)      # rows x offsets


def fit(rows_ll, g, n_games, idx=None):
    """Best offset for the rows in idx (all if None), refined with a parabola."""
    tot = np.zeros((n_games, len(OFFSETS)))
    m = slice(None) if idx is None else idx
    np.add.at(tot, g[m], rows_ll[m])
    return tot


def argbest(curve):
    k = int(np.argmax(curve))
    if 0 < k < len(curve) - 1:
        y0, y1, y2 = curve[k - 1], curve[k], curve[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(OFFSETS[k] + 0.5 * (y0 - y2) / den * (OFFSETS[1] - OFFSETS[0]))
    return float(OFFSETS[k])


def engine_ratio(me, pop, mm, mp, picks, g):
    """(text, your rate, peer rate, ratio, lo, hi) of the engine error rate in a category."""
    if mm.sum() < 100 or mp.sum() < 100:
        return ("(too few recent moves)", np.nan, np.nan, np.nan, np.nan, np.nan)
    n_g = g.max() + 1
    mv, er = np.zeros(n_g), np.zeros(n_g)
    np.add.at(mv, g[mm], 1)
    np.add.at(er, g[mm], me.err.to_numpy()[mm])
    r_me, r_pop = er.sum() / mv.sum(), pop.err.to_numpy()[mp].mean()
    b = np.array([er[p].sum() / max(mv[p].sum(), 1) for p in picks]) / r_pop
    lo, hi = np.percentile(b, [5, 95])
    tag = " clear" if lo > 1 or hi < 1 else ""
    return (f"{r_me:5.1%} / {r_pop:5.1%} = {r_me / r_pop:.2f} [{lo:.2f}, {hi:.2f}]{tag}", r_me, r_pop,
            r_me / r_pop, lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--min-moves", type=int, default=300)
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config)
    me, pop = load(cfg, "player"), load(cfg, "population")
    rows_me, rows_pop = per_game_ll(me, cfg, "player"), per_game_ll(pop, cfg, "population")
    gm_ids, gm = np.unique(me.game_id.to_numpy(), return_inverse=True)
    gp_ids, gp = np.unique(pop.game_id.to_numpy(), return_inverse=True)
    cats_me, cats_pop = categories(me), categories(pop)

    all_me = fit(rows_me, gm, len(gm_ids))
    all_pop = fit(rows_pop, gp, len(gp_ids))
    c_me, c_pop = argbest(all_me.sum(0)), argbest(all_pop.sum(0))
    rng = np.random.default_rng(0)
    picks = [rng.integers(0, len(gm_ids), len(gm_ids)) for _ in range(args.boot)]
    picks_r = [rng.integers(0, len(gm_ids), len(gm_ids)) for _ in range(args.boot)]
    gr = gm
    print(f"you: {len(me):,} moves in {len(gm_ids):,} games (whole history); peers {len(pop):,} moves")
    print(f"overall offset vs expected level: you {c_me:+.0f} (0 = config offset is right), peers {c_pop:+.0f}")
    recent = (me.mover_rating >= cfg["player"]["min_player_rating"]).to_numpy()
    print(f"engine column: error rate (mistake/blunder), your games >= {cfg['player']['min_player_rating']} vs peers, "
          f"ratio with 90% CI over games")
    print(f"\n{'category':44s} {'your moves':>10s} {'you':>6s} {'peers':>6s} {'Maia relative level (90% CI)':>30s}"
          f"   {'engine error rate you / peers':>30s}")
    out = []
    for name, mm in cats_me.items():
        mp = cats_pop[name]
        if mm.sum() < args.min_moves or mp.sum() < args.min_moves:
            continue
        cat_me = fit(rows_me, gm, len(gm_ids), np.where(mm)[0])
        cx_me = argbest(cat_me.sum(0))
        cx_pop = argbest(fit(rows_pop, gp, len(gp_ids), np.where(mp)[0]).sum(0))
        bias = cx_pop - c_pop
        rel = cx_me - c_me - bias
        boots = np.array([argbest(cat_me[p].sum(0)) - argbest(all_me[p].sum(0)) - bias for p in picks])
        lo, hi = np.percentile(boots, [5, 95])
        tag = "clear" if lo > 0 or hi < 0 else ""
        edge = " (grid edge)" if max(abs(cx_me), abs(cx_pop)) >= OFFSETS[-1] - 10 else ""
        er = engine_ratio(me, pop, mm & recent, mp, picks_r, gr)
        print(f"{name:44s} {mm.sum():10,d} {cx_me:+6.0f} {cx_pop:+6.0f} {rel:+8.0f} [{lo:+5.0f}, {hi:+5.0f}] "
              f"{tag:5s}{edge}   {er[0]}")
        out.append({"category": name, "moves": int(mm.sum()), "c_you": cx_me, "c_peers": cx_pop,
                    "relative": rel, "lo": lo, "hi": hi, "err_you": er[1], "err_peers": er[2],
                    "ratio": er[3], "ratio_lo": er[4], "ratio_hi": er[5]})
    pd.DataFrame(out).to_parquet(data_dir(cfg, "player") / "skill_profile.parquet", index=False)
    print("\nyou / peers = best-fit offset in that category (Elo above the expected level for each game);"
          "\nrelative = your offset minus your overall offset, minus the peers' category bias.")


if __name__ == "__main__":
    main()
