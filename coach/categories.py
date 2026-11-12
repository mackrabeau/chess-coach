"""Position categories shared by the skill profile (16) and the multi-band comparison (17).

annotate(df) adds: ph (opening / middlegame / endgame), etype (material class in endgames),
best_kind (capture / check / quiet), err (mistake or blunder, not the best move), blunder.
categories(df) -> {name: boolean mask}; difficulty rows only when a `pbest` column exists.
"""
from __future__ import annotations

import chess
import numpy as np

from .endgame_features import material_class
from .handcrafted import phase


def annotate(df):
    boards = [chess.Board(f) for f in df.fen]
    df["ph"] = [phase(b) for b in boards]
    df["etype"] = [material_class(b) if p == "endgame" else "" for b, p in zip(boards, df.ph)]
    bm = [chess.Move.from_uci(m) for m in df.best_move]
    df["best_kind"] = ["capture" if b.is_capture(m) else ("check" if b.gives_check(m) else "quiet")
                       for b, m in zip(boards, bm)]
    df["err"] = (df.label.isin(["mistake", "blunder"]) & ~df.played_is_best).to_numpy()
    df["blunder"] = (df.label == "blunder").to_numpy()
    return df


def categories(df):
    """name -> boolean mask (all rows first, under "all moves")."""
    wp, c = df.wp_best.to_numpy(), df.clock_before.to_numpy(float)
    cats = {
        "all moves": np.ones(len(df), bool),
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
        "clock: < 1 min": c < 60,
        "clock: 1-3 min": (c >= 60) & (c < 180),
        "clock: 3 min +": c >= 180,
    }
    if "pbest" in df and df.pbest.notna().any():
        cats["difficulty: hard (Maia finds best < 20%)"] = df.pbest < 0.2
        cats["difficulty: medium (20-50%)"] = (df.pbest >= 0.2) & (df.pbest < 0.5)
        cats["difficulty: easy (>= 50%)"] = df.pbest >= 0.5
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
