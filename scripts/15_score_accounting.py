"""Score accounting: where your points come from and go, per phase, vs your actual opponents.

Your positions are every one of your moves from ply 20 on (clock >= 30 s), with Stockfish's
win prob before your move (wp_best) and after it (wp_played), both from your side. Per game
this telescopes exactly:

    S - 0.5 =  (wp_best at your first analysed move - 0.5)                    opening (moves 1-10)
             + sum_i -(wp_best_i - wp_played_i)                               your losses
             + sum_i  (wp_best_{i+1} - wp_played_i)  for consecutive moves    your opponents' gifts
             + jumps across unanalysed stretches                              gaps
             + (S - wp_played at your last analysed move)                     end of game

Your losses are charged to the phase of the position you moved in, opponents' gifts to the
phase of the position they handed you. The end term is split by how the game ended (chess.com
result codes: timeout, resigned, checkmated, agreed, ...): it holds time scrambles (< 30 s is not
analysed), losses on time and any miscalibration of win prob vs score (checked at the bottom).

Peers can't be accounted the same way (only ~12 sampled positions per game), and it isn't
needed: on average a phase nets zero over a population, so the comparison is you vs the
opponents you actually played. Peers only give the share of games ending on time (01b status).

    python scripts/15_score_accounting.py --config configs/v0.yaml
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
from coach.fetch_chesscom import iter_player_games
from coach.handcrafted import phase
from coach.results import load_analyzed

PHASES = ("opening", "middlegame", "endgame")


def terminations(cfg) -> dict:
    """game_id -> how it ended, from your side: 'lost on time', 'won on time', 'draw on time', 'checkmate',
    'resigned', 'draw', 'abandoned', 'unknown'."""
    pcfg = cfg["player"]
    user = pcfg["username"].lower()
    out = {}
    for g in iter_player_games(sorted(data_dir(cfg, "player", "archives").glob("*.json")), pcfg):
        white = g["white"]["username"].lower() == user
        me, opp = g["white" if white else "black"], g["black" if white else "white"]
        gid = g.get("uuid") or g["url"].rstrip("/").split("/")[-1]
        codes = {me.get("result"), opp.get("result")}
        if "timeout" in codes:
            out[gid] = "lost on time" if me.get("result") == "timeout" else "won on time"
        elif "timevsinsufficient" in codes:
            out[gid] = "draw on time"
        elif "checkmated" in codes:
            out[gid] = "checkmate"
        elif "resigned" in codes:
            out[gid] = "resigned"
        elif "abandoned" in codes:
            out[gid] = "abandoned"
        elif codes & {"agreed", "repetition", "stalemate", "insufficient", "50move"}:
            out[gid] = "draw"
        else:
            out[gid] = "unknown"
    return out


def account(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game: the telescoping terms (columns), summing to result - 0.5."""
    rows = []
    for gid, g in df.sort_values("ply").groupby("game_id", sort=False):
        S = g.result.iat[0]
        if not np.isfinite(S):
            continue
        r = {"game_id": gid, "S": S, "rating": g.mover_rating.iat[0], "date": g.date.iat[0],
             "first_ply": g.ply.iat[0], "last_clock": g.clock_before.iat[-1]}
        for ph in PHASES:
            r[f"loss {ph}"] = 0.0
            r[f"gift {ph}"] = 0.0
        r["gaps"] = 0.0
        r["opening (moves 1-10)"] = g.wp_best.iat[0] - 0.5
        ply, wb, wpl, ph = g.ply.to_numpy(), g.wp_best.to_numpy(float), g.wp_played.to_numpy(float), g.ph.to_numpy()
        for i in range(len(g)):
            r[f"loss {ph[i]}"] -= wb[i] - wpl[i]
            if i + 1 < len(g):
                if ply[i + 1] == ply[i] + 2:
                    r[f"gift {ph[i + 1]}"] += wb[i + 1] - wpl[i]
                else:
                    r["gaps"] += wb[i + 1] - wpl[i]
        r["end of game"] = S - wpl[-1]
        r["wp_last"] = wpl[-1]
        rows.append(r)
    return pd.DataFrame(rows)


def ci_mean(x, B, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 5), np.percentile(m, 95)


def table(A, B, title):
    lines = [title, f"   {'term (points per game)':42s} {'mean':>7s} {'90% CI':>17s}"]
    def row(name, x):
        m, lo, hi = ci_mean(x, B)
        lines.append(f"   {name:42s} {m:+7.3f}  [{lo:+.3f}, {hi:+.3f}]")
    row("total: score - 0.5", A.S - 0.5)
    row("opening (moves 1-10, to your first analysed move)", A["opening (moves 1-10)"])
    for ph in PHASES:
        row(f"{ph}: your losses", A[f"loss {ph}"])
        row(f"{ph}: your opponents' gifts", A[f"gift {ph}"])
        row(f"{ph}: net", A[f"loss {ph}"] + A[f"gift {ph}"])
    row("gaps (unanalysed stretches mid-game)", A.gaps)
    row("end of game (after your last analysed move)", A["end of game"])
    for t, g in A.groupby("termination"):
        if len(g) >= 5:
            lines.append(f"      ending '{t}': {len(g) / len(A):4.0%} of games, end term "
                         f"{g['end of game'].sum() / len(A):+.3f} per game (all games)")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    Bn = cfg["report"]["bootstrap"]
    cmp_min = cfg["player"]["min_player_rating"]
    df = load_analyzed(cfg, "player")
    df = df[df.result.notna()].copy()
    df["ph"] = [phase(chess.Board(f)) for f in df.fen]
    A = account(df)
    term = terminations(cfg)
    A["termination"] = A.game_id.map(term).fillna("unknown")
    A["year"] = A.date.astype(str).str[:4]
    check = (A.drop(columns=["game_id", "S", "rating", "date", "first_ply", "last_clock", "wp_last",
                             "termination", "year"]).sum(1) - (A.S - 0.5)).abs().max()
    recent = A[A.rating >= cmp_min]
    L = [f"{len(A):,} games with results ({len(recent):,} at >= {cmp_min}); telescoping check max error {check:.1e}",
         f"first analysed move at ply {recent.first_ply.median():.0f} (median); "
         f"{(recent.first_ply > 21).mean():.0%} of games start later (gap counted in the opening term)"]
    L += table(recent, Bn, f"\nA. games >= {cmp_min}")

    # time: you vs peers
    L.append("\nB. games ending on time")
    for t in ("lost on time", "won on time", "draw on time"):
        L.append(f"   {t:14s} {(recent.termination == t).mean():5.1%} of your games")
    lost_t = recent[recent.termination == "lost on time"]
    if len(lost_t):
        L.append(f"   lost on time with win prob >= 0.6 at your last analysed move: {(lost_t.wp_last >= 0.6).sum()} games "
                 f"({(lost_t.wp_last >= 0.6).sum() / len(recent):.1%}), points thrown {(lost_t.wp_last[lost_t.wp_last >= 0.6]).sum() / len(recent):.3f} per game")
    won_t = recent[recent.termination == "won on time"]
    if len(won_t):
        L.append(f"   won on time with win prob <= 0.4: {(won_t.wp_last <= 0.4).sum()} games, points gained "
                 f"{(1 - won_t.wp_last[won_t.wp_last <= 0.4]).sum() / len(recent):.3f} per game")
    rp = data_dir(cfg, "population") / "results.parquet"
    if rp.exists():
        r = pd.read_parquet(rp)
        if "status" in r.columns:
            fin = r[r.white_score.notna()]
            L.append(f"   peers (Lichess {cfg['population']['rating_center']} band): {(fin.status == 'outoftime').mean():.1%} "
                     f"of games end on time (either side), you {recent.termination.str.contains('on time').mean():.1%}")
        else:
            L.append("   peers: rerun 01b_fetch_results.py to get game status (time forfeits)")

    # calibration: does win prob match score in your games?
    L.append("\nC. calibration: your final score by win prob at your positions (games >= {}):".format(cmp_min))
    d = df[df.mover_rating >= cmp_min]
    for a, b in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        m = (d.wp_best >= a) & (d.wp_best < b)
        if m.sum() >= 50:
            L.append(f"   win prob {a:.1f}-{min(b, 1):.1f}: mean {d.wp_best[m].mean():.3f} -> score {d.result[m].mean():.3f} "
                     f"(n {m.sum():,} moves)")

    # by year (you vs yourself)
    L.append("\nD. by year (whole history), points per game: opening | middlegame net | endgame net | end of game | total")
    for y, g in A.groupby("year"):
        if len(g) < 20:
            continue
        L.append(f"   {y} ({len(g):4d} games): {g['opening (moves 1-10)'].mean():+.3f} | "
                 f"{(g['loss middlegame'] + g['gift middlegame']).mean():+.3f} | "
                 f"{(g['loss endgame'] + g['gift endgame']).mean():+.3f} | {g['end of game'].mean():+.3f} | "
                 f"{(g.S - 0.5).mean():+.3f}")
    print("\n".join(L))
    A.to_parquet(data_dir(cfg, "player") / "score_accounting.parquet", index=False)


if __name__ == "__main__":
    main()
