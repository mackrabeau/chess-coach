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


# Display names for the report (the keys above stay the stable ids in data and findings).
PLAIN = {
    "all moves": "All your moves",
    "phase: opening (moves 10-15)": "Leaving the opening (moves 10–15)",
    "phase: middlegame": "Middlegame",
    "phase: endgame": "Endgame",
    "eval: losing (< 0.2)": "Worse positions",
    "eval: contested (0.2-0.8)": "Balanced positions",
    "eval: winning (> 0.8)": "Winning positions",
    "critical (one good move)": "Critical moments (one good move)",
    "not critical": "Non-critical positions",
    "best move is a capture": "When the best move is a capture",
    "best move is a check": "When the best move is a check",
    "best move is quiet": "When the best move is quiet",
    "clock: < 1 min": "Under 1 minute on the clock",
    "clock: 1-3 min": "1–3 minutes on the clock",
    "clock: 3 min +": "3+ minutes on the clock",
    "difficulty: hard (Maia finds best < 20%)": "Hard positions (few humans find the move)",
    "difficulty: medium (20-50%)": "Medium-difficulty positions",
    "difficulty: easy (>= 50%)": "Easy positions",
    "endgame: rook endings": "Rook endings",
    "endgame: minor-piece endings": "Minor-piece endings",
    "endgame: pawn endings": "Pawn endings",
    "endgame: queen endings": "Queen endings",
    "endgame: mixed": "Mixed endgames",
    "contested middlegame": "Balanced middlegames",
    "contested endgame": "Balanced endgames",
    "endgame, clock 3 min +": "Endgames with 3+ minutes left",
    "rook endings, clock 3 min +": "Rook endings with 3+ minutes left",
    "minor-piece endings, clock 3 min +": "Minor-piece endings with 3+ minutes left",
    "contested endgame, clock 3 min +": "Balanced endgames with 3+ minutes left",
    "winning (> 0.8), clock 3 min +": "Winning positions with 3+ minutes left",
    "middlegame, clock < 3 min": "Middlegames under 3 minutes",
    "middlegame, clock 3 min +": "Middlegames with 3+ minutes left",
    "critical: best move quiet": "Critical moments: the move is quiet",
    "critical: best move capture/check": "Critical moments: the shot is a capture or check",
    "critical: defending (wp_best < 0.5)": "Critical moments when defending",
    "critical: pressing (wp_best >= 0.5)": "Critical moments when you're pressing",
    "critical: contested (0.2-0.8)": "Critical moments in balanced positions",
    "critical: middlegame": "Critical moments in the middlegame",
    "critical: endgame": "Critical moments in the endgame",
    "critical: clock 3 min +": "Critical moments with 3+ minutes left",
}


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
