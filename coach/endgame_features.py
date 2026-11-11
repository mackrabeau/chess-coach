"""Endgame-specific features (for balanced-endgame diagnosis, 11b) and material classes.

Each feature is computed for one side ("us" = the side to move at the root). Distances
are king (Chebyshev) distances; ranks are relative to the side (0 = own back rank).
"""
from __future__ import annotations

import chess

CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]


def material_class(b: chess.Board) -> str:
    pieces = {pt: len(b.pieces(pt, chess.WHITE)) + len(b.pieces(pt, chess.BLACK))
              for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)}
    q, r, minors = pieces[chess.QUEEN], pieces[chess.ROOK], pieces[chess.KNIGHT] + pieces[chess.BISHOP]
    if q == r == minors == 0:
        return "pawn endings"
    if q == minors == 0:
        return "rook endings"
    if r == minors == 0:
        return "queen endings"
    if q == r == 0:
        return "minor-piece endings"
    return "mixed"


def _rel_rank(sq: int, c: chess.Color) -> int:
    r = chess.square_rank(sq)
    return r if c == chess.WHITE else 7 - r


def passed_pawns(b: chess.Board, c: chess.Color) -> list[int]:
    out = []
    enemy = b.pieces(chess.PAWN, not c)
    for sq in b.pieces(chess.PAWN, c):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        for e in enemy:
            ef, er = chess.square_file(e), chess.square_rank(e)
            ahead = er > r if c == chess.WHITE else er < r
            if abs(ef - f) <= 1 and ahead:
                blocked = True
                break
        if not blocked:
            out.append(sq)
    return out


def _unstoppable(b: chess.Board, c: chess.Color, sq: int) -> bool:
    """Rule of the square: the enemy king can't catch the pawn (only if the enemy has no pieces)."""
    enemy = not c
    if any(b.pieces(pt, enemy) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)):
        return False
    k = b.king(enemy)
    if k is None:
        return False
    promo = chess.square(chess.square_file(sq), 7 if c == chess.WHITE else 0)
    steps = 7 - _rel_rank(sq, c)
    if _rel_rank(sq, c) == 1:
        steps -= 1                                    # double step from the start square
    tempo = 0 if b.turn == c else 1                   # the defender moves first if it's their turn
    return chess.square_distance(k, promo) - tempo > steps


def side_features(b: chess.Board, c: chess.Color) -> dict[str, float]:
    k = b.king(c)
    own_passed = passed_pawns(b, c)
    enemy_passed = passed_pawns(b, not c)
    all_passed = own_passed + enemy_passed
    own_pawns = b.pieces(chess.PAWN, c)
    pawn_files = [chess.square_file(p) for p in b.pieces(chess.PAWN, chess.WHITE) | b.pieces(chess.PAWN, chess.BLACK)]
    rooks = b.pieces(chess.ROOK, c)
    f = {
        "king_centralization": 7 - min(chess.square_distance(k, s) for s in CENTER) if k is not None else 0,
        "king_advance": _rel_rank(k, c) if k is not None else 0,
        "king_near_passers": -min((chess.square_distance(k, p) for p in all_passed), default=7) if k is not None else 0,
        "king_stops_enemy_passer": float(any(
            k is not None and chess.square_file(k) == chess.square_file(p)
            and (_rel_rank(k, c) < _rel_rank(p, c)) for p in enemy_passed)),
        "passed_pawns": len(own_passed),
        "passed_advance": max((_rel_rank(p, c) for p in own_passed), default=0),
        "protected_passers": sum(1 for p in own_passed if b.attackers(c, p) & own_pawns),
        "outside_passers": sum(1 for p in own_passed
                               if min((abs(chess.square_file(p) - pf) for pf in pawn_files
                                       if pf != chess.square_file(p)), default=0) >= 3),   # needs other pawns
        "unstoppable_passers": sum(1 for p in own_passed if _unstoppable(b, c, p)),
        "rook_behind_passer": sum(1 for r in rooks for p in all_passed
                                  if chess.square_file(r) == chess.square_file(p)
                                  and ((_rel_rank(r, b.color_at(p)) < _rel_rank(p, b.color_at(p))))),
        "rook_7th": sum(1 for r in rooks if _rel_rank(r, c) == 6),
        "rook_mobility": sum(len(b.attacks(r)) for r in rooks),
        "pieces_in_enemy_half": sum(1 for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                                    for s in b.pieces(pt, c) if _rel_rank(s, c) >= 4),
        "pawns": len(own_pawns),
        "pieces": sum(len(b.pieces(pt, c)) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)),
    }
    return f


FEATURES = list(side_features(chess.Board(), chess.WHITE).keys())

READABLE = {
    "king_centralization": "king centralization", "king_advance": "king advanced up the board",
    "king_near_passers": "king close to the passed pawns", "king_stops_enemy_passer": "king in front of their passer",
    "passed_pawns": "passed pawns", "passed_advance": "passed pawn advanced",
    "protected_passers": "protected passed pawns", "outside_passers": "outside passed pawns",
    "unstoppable_passers": "unstoppable passed pawns (rule of the square)",
    "rook_behind_passer": "rook behind a passed pawn", "rook_7th": "rook on the 7th rank",
    "rook_mobility": "rook activity (squares attacked)", "pieces_in_enemy_half": "pieces in the enemy half",
    "pawns": "pawns", "pieces": "pieces (fewer = traded down)",
}


def features(b: chess.Board, us: chess.Color) -> dict[str, float]:
    """{'your <f>': ..., 'their <f>': ...} for the position b, seen from side `us`."""
    out = {}
    for who, c in (("your", us), ("their", not us)):
        for k, v in side_features(b, c).items():
            out[f"{who} {k}"] = v
    return out


def end_of(fen: str, line) -> chess.Board:
    b = chess.Board(fen)
    for u in line:
        b.push_uci(u)
    return b
