"""Error anatomy: how deep was the punishment, and was it on a human's radar?

For every mistake / blunder (played move not best), along the settled lines:
  allowed side (played line):
    refute_ply   first ply (1 = your move, 2 = their reply, ...) after which material is
                 >= `swing` below the root and stays there to the end of the line (NaN if never)
    reply        the opponent's immediate reply (ply 2) and its kind: capture / check / quiet
    p_reply      Maia's probability of that reply from the position after your move, at your
                 level (09c fills it): low = a move humans at your level don't consider
  missed side (best line):
    gain_ply     first ply after which material is >= `swing` above the root and stays
    best_kind    kind of the best move: capture / check / quiet
  think_s        seconds you spent on the move (your games only: clock before this move +
                 increment - clock before your next move; NaN if the next move isn't stored)
"""
from __future__ import annotations

import chess
import numpy as np
import pandas as pd

from .motifs import material


def move_kind(board: chess.Board, uci: str) -> str:
    m = chess.Move.from_uci(uci)
    if board.gives_check(m):
        return "check"
    if board.is_capture(m):
        return "capture"
    return "quiet"


def settle_ply(fen: str, line, us: chess.Color, swing: float, sign: int) -> float:
    """First ply after which material (us POV) is >= swing beyond the root in direction `sign`
    (+1 gain, -1 loss) and stays there to the end of the line; NaN if it never does."""
    b = chess.Board(fen)
    m0 = material(b, us)
    traj = []
    for u in line:
        b.push_uci(u)
        traj.append(sign * (material(b, us) - m0) >= swing)
    if not traj or not traj[-1]:
        return float("nan")
    k = len(traj)
    while k > 0 and traj[k - 1]:
        k -= 1
    return float(k + 1)


def anatomy_rows(df: pd.DataFrame, swing: float = 2) -> pd.DataFrame:
    """One row per error (mistake / blunder, played move not best), index = df row."""
    err = (df.label.isin(["mistake", "blunder"]) & ~df.played_is_best).to_numpy()
    rows = []
    for i in np.flatnonzero(err):
        fen = df.fen.iat[i]
        root = chess.Board(fen)
        us = root.turn
        played, best = list(df.played_line.iat[i]), list(df.best_line.iat[i])
        r = {"refute_ply": settle_ply(fen, played, us, swing, -1),
             "gain_ply": settle_ply(fen, best, us, swing, +1),
             "best_kind": move_kind(root, df.best_move.iat[i]),
             "reply": None, "reply_kind": None}
        if len(played) >= 2:
            after = root.copy(stack=False)
            after.push_uci(played[0])
            r["reply"] = played[1]
            r["reply_kind"] = move_kind(after, played[1])
        rows.append(r)
    return pd.DataFrame(rows, index=np.flatnonzero(err))


def think_times(df: pd.DataFrame) -> np.ndarray:
    """Seconds spent per move, from consecutive own moves of the same game (NaN if unknown)."""
    out = np.full(len(df), np.nan)
    key = pd.Series(range(len(df)), index=pd.MultiIndex.from_arrays([df.game_id, df.ply]))
    nxt = key.reindex(pd.MultiIndex.from_arrays([df.game_id, df.ply + 2])).to_numpy()
    ok = ~pd.isna(nxt)
    j = nxt[ok].astype(int)
    i = np.flatnonzero(ok)
    t = df.clock_before.to_numpy(float)[i] + df.increment.to_numpy(float)[i] - df.clock_before.to_numpy(float)[j]
    out[i] = np.where(t >= 0, t, np.nan)
    return out


def depth_bucket(p: float) -> str:
    if np.isnan(p):
        return "no material swing"
    if p <= 2:
        return "punished next move (ply 2)"
    if p <= 4:
        return "2-move combination (ply 3-4)"
    return "deeper (ply 5+)"
