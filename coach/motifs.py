"""Tactical motif detection along engine lines (blunder / tactical track).

For a position where a mistake swings material, we ask two questions:
  missed win : did the best line win material for you?   -> motifs in *your* moves of the best line
  allowed    : did the played line lose material for you? -> motifs in the *opponent's* moves of the played line
and label the first motif found (scanning the first `max_moves` moves of that side):

  mate         the line delivers checkmate (back_rank_mate if the mated king is on its back rank)
  promotion    a pawn promotes
  fork         after the move, the moved piece attacks >= 2 worthwhile non-pawn targets
  pin          after the move, a slider attacks a piece that shields a more valuable one (or the king)
  skewer       after the move, a slider attacks a valuable piece (or king) with a lesser one behind it
  discovered   the move opens a line: another slider of the same side now attacks a worthwhile target
  remove_defender  a capture of a piece that defended something captured on the next move
  winning_capture  capturing an undefended piece, or a piece worth more than the capturer
  other        material changed hands but none of the above was detected

"Worthwhile target" = the king (check), a piece worth more than the attacker, or an undefended piece.
These are heuristics on python-chess boards, not a tactics engine; 07b shows examples per
motif so the detectors can be checked by eye.
"""
from __future__ import annotations

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
MOTIFS = ["mate", "back_rank_mate", "promotion", "fork", "pin", "skewer", "discovered",
          "remove_defender", "winning_capture", "other"]
DIRECTIONS = ["missed_win", "allowed", "both", "no_material"]
DIRS = {chess.BISHOP: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
        chess.ROOK: [(1, 0), (-1, 0), (0, 1), (0, -1)]}
DIRS[chess.QUEEN] = DIRS[chess.BISHOP] + DIRS[chess.ROOK]


def material(b: chess.Board, pov: chess.Color) -> int:
    return sum((VAL[p.piece_type] if p.piece_type != chess.KING else 0) * (1 if p.color == pov else -1)
               for p in b.piece_map().values())


def _defended(b: chess.Board, sq: int) -> bool:
    p = b.piece_at(sq)
    return p is not None and bool(b.attackers(p.color, sq))


def _worthy(b: chess.Board, attacker_sq: int, target_sq: int) -> bool:
    a, t = b.piece_at(attacker_sq), b.piece_at(target_sq)
    if a is None or t is None or a.color == t.color:
        return False
    return t.piece_type == chess.KING or VAL[t.piece_type] > VAL[a.piece_type] or not _defended(b, target_sq)


def _targets(b: chess.Board, sq: int) -> list[int]:
    p = b.piece_at(sq)
    if p is None:
        return []
    return [t for t in b.attacks(sq) & b.occupied_co[not p.color] if _worthy(b, sq, t)]


def _ray_pairs(b: chess.Board, sq: int):
    """For a slider on sq: (first enemy piece, piece right behind it) along each ray."""
    p = b.piece_at(sq)
    if p is None or p.piece_type not in DIRS:
        return
    f0, r0 = chess.square_file(sq), chess.square_rank(sq)
    for df, dr in DIRS[p.piece_type]:
        hits = []
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8 and len(hits) < 2:
            s = chess.square(f, r)
            if b.piece_at(s) is not None:
                hits.append(s)
            f, r = f + df, r + dr
        if len(hits) == 2:
            a, c = b.piece_at(hits[0]), b.piece_at(hits[1])
            if a.color != p.color and c.color != p.color:
                yield hits[0], hits[1]


def move_motifs(before: chess.Board, move: chess.Move, nxt: chess.Move | None) -> list[str]:
    """Motifs created by `move` (played from `before`). `nxt` = the same side's next move."""
    out = []
    mover = before.turn
    after = before.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        k = after.king(not mover)
        back = 0 if (not mover) == chess.WHITE else 7
        out.append("back_rank_mate" if k is not None and chess.square_rank(k) == back else "mate")
    if move.promotion:
        out.append("promotion")
    if len([t for t in _targets(after, move.to_square) if after.piece_at(t).piece_type != chess.PAWN]) >= 2:
        out.append("fork")
    for sq in after.pieces(chess.BISHOP, mover) | after.pieces(chess.ROOK, mover) | after.pieces(chess.QUEEN, mover):
        if sq == move.to_square:
            continue
        new = set(_targets(after, sq)) - set(before.attacks(sq) & before.occupied_co[not mover])
        if any(move.from_square in chess.SquareSet(chess.between(sq, t)) for t in new):
            out.append("discovered")
            break
    if before.is_capture(move):
        victim_sq = move.to_square
        victim = before.piece_at(victim_sq)
        if victim is not None and nxt is not None and after.is_capture(nxt):
            # did the captured piece defend the square captured next?
            if nxt.to_square in before.attacks(victim_sq):
                out.append("remove_defender")
    for front, back_sq in _ray_pairs(after, move.to_square):
        fv, bv = VAL[after.piece_at(front).piece_type], VAL[after.piece_at(back_sq).piece_type]
        if bv > fv:
            out.append("pin")
        elif fv > bv and fv >= VAL[chess.ROOK]:
            out.append("skewer")
    if before.is_capture(move):
        victim_sq = move.to_square
        victim = before.piece_at(victim_sq)
        attacker = before.piece_at(move.from_square)
        if victim is not None and (not _defended(before, victim_sq)
                                   or VAL[victim.piece_type] > VAL[attacker.piece_type]):
            out.append("winning_capture")
    return out


def line_motifs(fen: str, line: list[str], side_offset: int, max_moves: int = 4) -> list[str]:
    """Motifs in the moves of one side along `line`: side_offset 0 = root mover's moves
    (plies 0, 2, ...), 1 = opponent's (plies 1, 3, ...). Ordered, first occurrence first."""
    b = chess.Board(fen)
    moves = [chess.Move.from_uci(u) for u in line]
    found: list[str] = []
    seen = 0
    for i, m in enumerate(moves):
        if i % 2 == side_offset:
            nxt = moves[i + 2] if i + 2 < len(moves) else None
            for t in move_motifs(b, m, nxt):
                if t not in found:
                    found.append(t)
            seen += 1
            if seen >= max_moves:
                break
        b.push(m)
    return found


def classify(fen: str, best_line: list[str], played_line: list[str], swing: float = 2) -> dict:
    """Direction + motifs for one position (lines are the settled ones, even length)."""
    root = chess.Board(fen)
    us = root.turn
    m0 = material(root, us)

    def end_mat(line):
        b = root.copy(stack=False)
        for u in line:
            b.push_uci(u)
        return material(b, us)

    gain = end_mat(best_line) - m0      # what the best line wins for you
    loss = end_mat(played_line) - m0    # what the played line costs you (negative = lost)
    missed, allowed = gain >= swing, loss <= -swing
    direction = ("both" if missed and allowed else "missed_win" if missed
                 else "allowed" if allowed else "no_material")
    mot_missed = line_motifs(fen, best_line, 0)
    mot_allowed = line_motifs(fen, played_line, 1)
    if direction == "allowed":
        primary = mot_allowed
    elif direction in ("missed_win", "both"):
        primary = mot_missed or mot_allowed
    else:  # no material swing: only mates are meaningful here
        primary = [m for m in mot_allowed + mot_missed if m in ("mate", "back_rank_mate")]
    return {"direction": direction, "gain": gain, "loss": loss,
            "motif": primary[0] if primary else "other",
            "motifs_missed": mot_missed, "motifs_allowed": mot_allowed}


TACTIC_TRACKS = ["blunder", "tactical"]
DIR_TEXT = {"missed_win": "missed", "allowed": "allowed", "both": "missed + allowed",
            "no_material": "no material swing"}


def tactic_labels(df, tracks):
    """Direction + primary motif for every blunder / tactical mistake (index = df row)."""
    import numpy as np
    import pandas as pd

    idx = np.flatnonzero(np.isin(tracks, TACTIC_TRACKS))
    rows = [classify(df.fen.iat[i], list(df.best_line.iat[i]), list(df.played_line.iat[i]))
            for i in idx]
    out = pd.DataFrame(rows, index=idx)
    if len(out):
        out["category"] = [f"{DIR_TEXT[d]}: {m.replace('_', ' ')}" for d, m in zip(out.direction, out.motif)]
    else:
        out["category"] = pd.Series(dtype=str)
    return out
