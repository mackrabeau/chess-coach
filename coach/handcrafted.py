"""Named, interpretable position features (baseline feature extractor).

Every feature is computed for "us" (the side that moved at the root) and "them".
Because engine lines have an even number of plies, the end position has the same
side to move as the root, so us/them mean the same thing at both ends.
"""
from __future__ import annotations

import chess
import numpy as np

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
WHITE_HALF = chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4
BLACK_HALF = chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7 | chess.BB_RANK_8

SIDE_FEATURES = [
    "pawns", "knights", "bishops", "rooks", "queens", "bishop_pair",
    "doubled_pawns", "isolated_pawns", "backward_pawns", "passed_pawns",
    "passed_pawn_advance", "pawn_islands",
    "rooks_open_file", "rooks_halfopen_file", "rooks_7th", "outposts",
    "knight_mobility", "bishop_mobility", "rook_mobility", "queen_mobility",
    "king_shield", "king_zone_attacks", "king_open_files", "king_center_dist",
    "hanging_pieces", "attacked_by_lower", "center_control", "space",
    "undeveloped_minors",
]
FEATURE_NAMES = [f"us_{n}" for n in SIDE_FEATURES] + [f"them_{n}" for n in SIDE_FEATURES]
N_FEATURES = len(FEATURE_NAMES)


def readable(name: str) -> str:
    side, _, rest = name.partition("_")
    who = "your" if side == "us" else "opponent's"
    return f"{who} {rest.replace('_', ' ')}"


def _rel_rank(sq: int, c: chess.Color) -> int:
    r = chess.square_rank(sq)
    return r if c == chess.WHITE else 7 - r


def _sq(file: int, rel_rank: int, c: chess.Color) -> int:
    return chess.square(file, rel_rank if c == chess.WHITE else 7 - rel_rank)


def side_features(board: chess.Board, c: chess.Color) -> list[float]:
    enemy = not c
    own_p = board.pieces(chess.PAWN, c)
    en_p = board.pieces(chess.PAWN, enemy)
    own_files = [0] * 8
    en_files = [0] * 8
    for s in own_p:
        own_files[chess.square_file(s)] += 1
    for s in en_p:
        en_files[chess.square_file(s)] += 1

    counts = [len(board.pieces(pt, c)) for pt in PIECES]
    bishop_pair = float(counts[2] >= 2)

    # --- pawn structure
    doubled = sum(max(0, n - 1) for n in own_files)
    isolated = passed = passed_adv = backward = 0
    for s in own_p:
        f, rr = chess.square_file(s), _rel_rank(s, c)
        adj = [g for g in (f - 1, f + 1) if 0 <= g < 8]
        if all(own_files[g] == 0 for g in adj):
            isolated += 1
        blocked = any(
            chess.square_file(e) in (f, *adj) and _rel_rank(e, c) > rr for e in en_p
        )
        if not blocked:
            passed += 1
            passed_adv += rr
            continue
        supported = any(chess.square_file(o) in adj and _rel_rank(o, c) <= rr for o in own_p)
        if not supported and rr < 7:
            stop = _sq(f, rr + 1, c)
            if board.attackers(enemy, stop) & en_p:
                backward += 1
    islands, prev = 0, False
    for n in own_files:
        if n and not prev:
            islands += 1
        prev = bool(n)

    # --- rooks
    r_open = r_half = r_7th = 0
    for s in board.pieces(chess.ROOK, c):
        f = chess.square_file(s)
        if own_files[f] == 0 and en_files[f] == 0:
            r_open += 1
        elif own_files[f] == 0:
            r_half += 1
        if _rel_rank(s, c) == 6:
            r_7th += 1

    # --- outposts: minor piece on ranks 4-6, defended by own pawn, no enemy pawn can ever attack it
    outposts = 0
    for s in board.pieces(chess.KNIGHT, c) | board.pieces(chess.BISHOP, c):
        rr, f = _rel_rank(s, c), chess.square_file(s)
        if not 3 <= rr <= 5 or not (board.attackers(c, s) & own_p):
            continue
        adj = (f - 1, f + 1)
        if not any(chess.square_file(e) in adj and _rel_rank(e, c) > rr for e in en_p):
            outposts += 1

    # --- mobility (squares attacked that aren't occupied by own pieces)
    not_own = ~board.occupied_co[c] & chess.BB_ALL
    mob = []
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        mob.append(sum((board.attacks_mask(s) & not_own).bit_count() for s in board.pieces(pt, c)))

    # --- king safety
    ks = board.king(c)
    if ks is None:
        shield = zone_att = open_k = 0
        center_dist = 0.0
    else:
        kf, kr = chess.square_file(ks), _rel_rank(ks, c)
        files = [g for g in (kf - 1, kf, kf + 1) if 0 <= g < 8]
        shield = 0
        for g in files:
            for dr in (1, 2):
                if kr + dr <= 7 and _sq(g, kr + dr, c) in own_p:
                    shield += 1
        zone = chess.SquareSet(chess.BB_KING_ATTACKS[ks] | chess.BB_SQUARES[ks])
        zone_att = sum(len(board.attackers(enemy, z)) for z in zone)
        open_k = sum(1 for g in files if own_files[g] == 0)
        center_dist = max(abs(kf - 3.5), abs(chess.square_rank(ks) - 3.5))

    # --- loose / threatened pieces (non-pawn, non-king)
    hanging = by_lower = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for s in board.pieces(pt, c):
            atk = board.attackers(enemy, s)
            if not atk:
                continue
            if not board.is_attacked_by(c, s):
                hanging += 1
            if any(VALUE[board.piece_type_at(a)] < VALUE[pt] for a in atk):
                by_lower += 1

    # --- center and space
    center = sum(len(board.attackers(c, sq)) for sq in CENTER)
    att = 0
    for s in chess.SquareSet(board.occupied_co[c]):
        att |= board.attacks_mask(s)
    space = (att & (BLACK_HALF if c == chess.WHITE else WHITE_HALF)).bit_count()

    back = chess.BB_RANK_1 if c == chess.WHITE else chess.BB_RANK_8
    undeveloped = len((board.pieces(chess.KNIGHT, c) | board.pieces(chess.BISHOP, c)) & back)

    return [
        *counts, bishop_pair,
        doubled, isolated, backward, passed, passed_adv, islands,
        r_open, r_half, r_7th, outposts,
        *mob,
        shield, zone_att, open_k, center_dist,
        hanging, by_lower, center, space,
        undeveloped,
    ]


def features(board: chess.Board, us: chess.Color) -> np.ndarray:
    return np.asarray(side_features(board, us) + side_features(board, not us), dtype=np.float32)


def phase(board: chess.Board) -> str:
    npm = sum(VALUE[pt] * len(board.pieces(pt, c)) for pt in PIECES[1:] for c in chess.COLORS)
    if npm <= 26:
        return "endgame"
    return "opening" if board.fullmove_number <= 15 else "middlegame"
