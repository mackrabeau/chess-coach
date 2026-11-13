"""Line patterns: named patterns of the engine line (docs/future_work.md "Concept groups" -> "Rethink").

A concept is something the BEST LINE does that the 2nd-best line doesn't ("give a pawn in the endgame",
"open the position with a pawn break"), so it can be trained as a decision:
  opportunity  the best line has pattern X, the 2nd-best line doesn't, and best beats 2nd by >= GAP win prob
  choice       took = the played line has X (the player started it; any move that does X counts)
  cost         win prob lost by the played move (wp_best - wp_played); Elo uses it on declined rows
Rows are also kept where only the played line has X (no opportunity): the reverse error (unsound sac,
premature trade) for later use.

Lines are the settled ones (03_settle_lines), so a line doesn't stop mid-exchange and fake a sacrifice.
Material along a line is read in move pairs: after each opponent reply, but undone if our next move
recaptures at once ("settled material" s_j = max(mat after their reply, mat after our next move)).
The tactic patterns reuse coach/motifs.line_motifs (cashed-in fork / pin / skewer / discovered); passed
pawns come from coach/endgame_features.passed_pawns; phase from coach/handcrafted.phase.

Each pattern: family (claims are made at family level; one player has ~10-60 opportunities per narrow
pattern), an advice sentence, and a rule on the replayed line. `ctrl_light_square` is a NEGATIVE CONTROL
(best move lands on a light square): it has no chess meaning, so the player's uptake ratio must come out
~1 and it must never pass the gate; it is never reported as a finding.

    pattern_table(cfg, which, df) -> rows (pos_id, pattern, family, opportunity, took, alternative, cost,
                                     confidence), cached in <set>/linepatterns.parquet
"""
from __future__ import annotations

import json
import multiprocessing as mp

import chess
import numpy as np
import pandas as pd

from .config import data_dir
from .endgame_features import passed_pawns
from .handcrafted import phase
from .motifs import VAL, _cashed, _targets, material, move_motifs

VERSION = 2             # bump when a rule changes: the cache is rebuilt
GAP = 0.05              # opportunity: best beats 2nd best by at least this much win probability
SAC_GAP = 0.08          # sacrifices: a bigger gap (small gaps were mostly incidental material)
CONF_GAP = 0.15         # label confidence = min(1, gap / CONF_GAP) x (1 - flip rate at 4x nodes, if measured)
HORIZON = 4             # our moves of a line that a rule looks at (8 plies)
CONTROL = "ctrl_light_square"

# name -> (family, advice). Order = display order.
PATTERNS = {
    "sac_combination": ("sacrifice", "Check captures and checks that hang material: this sacrifice wins it back "
                                     "(or more) a few moves later."),
    "sac_compensation": ("sacrifice", "Give material for a lasting attack or initiative when the line keeps the "
                                      "pressure; don't only count pawns."),
    "pawn_sac_endgame": ("sacrifice", "In endgames, give a pawn for activity: an active king or rook, or a passed "
                                      "pawn, is often worth more than the pawn."),
    "gambit_pawn": ("sacrifice", "Give a pawn for development, open lines or the initiative when it's on offer."),
    "queen_trade": ("exchanges", "Trade queens here: it kills their counterplay or makes your edge easier to "
                                 "convert."),
    "simplify_ahead": ("exchanges", "When winning, trade pieces down to a simple won ending."),
    "pawn_break": ("pawns", "Open the position with a pawn break (a pawn move that attacks their pawns)."),
    "passer_push": ("pawns", "Push your passed pawn: it ties their pieces down."),
    "king_activation": ("king", "In the endgame, bring the king forward: it's a fighting piece now."),
    "forcing_checks": ("king", "Look for a run of checks: forcing moves against the king first."),
    "fork": ("tactics", "Look for forks: one piece hitting two targets, cashed in next move."),
    "line_tactic": ("tactics", "Look for pins, skewers and discovered attacks along open lines."),
    "retreat": ("piece play", "Consider backward moves: the best move here is a retreat."),
    "quiet_defence": ("piece play", "First check their threat: the best move is a quiet move that meets it."),
    CONTROL: ("control", "Negative control (no chess meaning): the best move lands on a light square."),
}
FAMILIES = list(dict.fromkeys(f for f, _ in PATTERNS.values()))
# Precision by eye (2026-11-12/13): Sonnet reviewers judged ~20 sampled peer opportunities per pattern
# (FEN + best / 2nd lines, replayed with python-chess) after 3 rounds of rule fixes; last round per pattern.
# n ~ 20 -> +-10-20 points. The gate (coach/weakness.uptake_gate) uses these: >= CLAIM_PRECISION may carry a
# claim, >= LEAD_PRECISION only a lead, below that the pattern is shown as exploratory and never ranked.
PRECISION = {"retreat": 1.00, "quiet_defence": 1.00, "passer_push": 0.95, "forcing_checks": 0.95,
             "queen_trade": 0.80, "pawn_break": 0.80, "king_activation": 0.78, "fork": 0.75,
             "sac_combination": 0.55, "line_tactic": 0.50, "pawn_sac_endgame": 0.50, "gambit_pawn": 0.40,
             "sac_compensation": 0.35, "simplify_ahead": 0.35}
CLAIM_PRECISION, LEAD_PRECISION = 0.85, 0.70


def _rel_rank(sq: int, c: chess.Color) -> int:
    r = chess.square_rank(sq)
    return r if c == chess.WHITE else 7 - r


def _npm(b: chess.Board) -> int:
    return sum(VAL[p.piece_type] for p in b.piece_map().values() if p.piece_type not in (chess.KING, chess.PAWN))


def _queens(b: chess.Board) -> tuple[int, int]:
    return len(b.pieces(chess.QUEEN, chess.WHITE)), len(b.pieces(chess.QUEEN, chess.BLACK))


def _wins_now(b: chess.Board) -> set[int]:
    """Squares of pieces (worth >= 3) the side to move can win at once (undefended, or worth more than the
    capturer), plus the enemy king's square if it has a mate in one."""
    out = set()
    for m in b.legal_moves:
        if b.is_capture(m) and not b.is_en_passant(m):
            v, a = b.piece_at(m.to_square), b.piece_at(m.from_square)
            if v is not None and VAL[v.piece_type] >= 3 and (
                    VAL[v.piece_type] > VAL[a.piece_type] or not b.attackers(v.color, m.to_square)):
                out.add(m.to_square)
        b.push(m)
        mate = b.is_checkmate()
        b.pop()
        if mate:
            k = b.king(not b.turn)
            if k is not None:
                out.add(k)
    return out


def threats(b: chess.Board) -> set[int]:
    """What the side NOT to move threatens (see _wins_now) if the side to move passed. Empty in check
    (every move is a defence there; not a "spot the threat" decision)."""
    if b.is_check():
        return set()
    n = b.copy(stack=False)
    n.push(chess.Move.null())
    return _wins_now(n)


def replay(fen: str, line: list[str]) -> dict:
    """Per-ply facts of a line from the root mover's view."""
    b = chess.Board(fen)
    us = b.turn
    mats, boards, moves = [material(b, us)], [b.copy(stack=False)], []
    for u in line:
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            break
        moves.append(m)
        b.push(m)
        mats.append(material(b, us))
        boards.append(b.copy(stack=False))
    return {"us": us, "mats": mats, "boards": boards, "moves": moves, "line": [m.uci() for m in moves]}


def settled_material(mats: list[int], horizon: int = HORIZON) -> list[int]:
    """s_j for j = 1..: material after their j-th reply, undone if our next move recaptures."""
    out = []
    for j in range(1, horizon + 1):
        k = 2 * j
        if k >= len(mats):
            break
        out.append(max(mats[k], mats[k + 1]) if k + 1 < len(mats) else mats[k])
    return out


def end_index(boards: list[chess.Board], moves: list[chess.Move]) -> int:
    """Last ply index with us to move whose preceding opponent move was not a capture (a line that stops
    right after their capture has our recapture pending)."""
    k = len(boards) - 1 if (len(boards) - 1) % 2 == 0 else len(boards) - 2
    while k >= 2 and boards[k - 1].is_capture(moves[k - 1]):
        k -= 2
    return k


def strict_endgame(b: chess.Board) -> bool:
    """No queens and at most 8 points of pieces per side (e.g. rook + minor)."""
    if b.pieces(chess.QUEEN, chess.WHITE) or b.pieces(chess.QUEEN, chess.BLACK):
        return False
    return all(sum(VAL[pt] * len(b.pieces(pt, c)) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK)) <= 8
               for c in chess.COLORS)


def _new_big_attack(after: chess.Board, sq: int) -> bool:
    """The piece on sq attacks an enemy piece worth >= 3 that is undefended or worth more than it."""
    p = after.piece_at(sq)
    for t in after.attacks(sq) & after.occupied_co[not p.color]:
        v = after.piece_at(t)
        if v.piece_type != chess.KING and VAL[v.piece_type] >= 3 and (
                VAL[v.piece_type] > VAL[p.piece_type] or not after.attackers(v.color, t)):
            return True
    return False


def _cashed_line_tactic(boards, moves, us) -> bool:
    """A pin / skewer / discovered attack created by one of our first 3 moves (coach/motifs geometry) and
    cashed in by a later capture of ours: for pins and skewers the captured piece must be the pinned one
    or the one behind (the cheaper of the two), not just any piece on the line."""
    for i in range(0, min(len(moves), 6), 2):
        before, m = boards[i], moves[i]
        if i == 0 and before.is_capture(m):
            continue                                   # the first move captures: a capture, not a pin
        nxt = moves[i + 2] if i + 2 < len(moves) else None
        for kind, targets in move_motifs(before, m, nxt):
            if kind not in ("pin", "skewer", "discovered") or not targets:
                continue
            after = boards[i + 1]
            if kind != "discovered":
                targets = {min(targets, key=lambda t: VAL[after.piece_at(t).piece_type])}
            targets = {t for t in targets if after.piece_at(t).piece_type != chess.PAWN}
            if targets and _cashed(after, moves[:i + 5], i + 1, targets, us):
                return True
    return False


def _cashed_fork(boards, moves, us) -> bool:
    """One of our first 3 moves attacks >= 2 worthwhile non-pawn targets (coach/motifs._targets), the forker
    can't be taken by a cheaper piece, and we capture one of the targets within our next 2 moves."""
    for i in range(0, min(len(moves), 6), 2):
        after, m = boards[i + 1], moves[i]
        tg = [t for t in _targets(after, m.to_square) if after.piece_at(t).piece_type != chess.PAWN]
        if len(tg) < 2:
            continue
        fv = VAL[after.piece_at(m.to_square).piece_type]
        if any(VAL[after.piece_at(a).piece_type] < fv for a in after.attackers(not us, m.to_square)):
            continue
        if _cashed(after, moves[:i + 5], i + 1, set(tg), us):
            return True
    return False


def detect(fen: str, line: list[str], ph: str | None = None, root_threats: set | None = None,
           wp: float = 0.5) -> tuple[set[str], dict]:
    """Patterns present in one line (starting with the root mover's move), and facts used to compare
    lines (dip = material given up, net, within HORIZON of our moves)."""
    if not line:
        return set(), {"dip": 0}
    r = replay(fen, line)
    if not r["moves"]:
        return set(), {"dip": 0}
    us, mats, boards, moves = r["us"], r["mats"], r["boards"], r["moves"]
    root, first = boards[0], moves[0]
    ph = ph or phase(root)
    out = set()
    m0 = mats[0]
    ke = end_index(boards, moves)
    end, endb = mats[ke], boards[ke]
    s = settled_material(mats)
    dip = m0 - min(s) if s else 0
    race = any(m.promotion for m in moves[:2 * HORIZON])
    if not race and wp >= 0.3:
        if dip >= 2:
            if end >= m0:
                out.add("sac_combination")
            elif end <= m0 - 2:
                out.add("sac_compensation")
        elif dip == 1 and end <= m0 - 1 and m0 - 1 in s[:2]:
            if ph == "endgame" and not (root.pieces(chess.QUEEN, chess.WHITE) or root.pieces(chess.QUEEN, chess.BLACK)):
                out.add("pawn_sac_endgame")
            elif ph != "endgame" and wp < 0.8:
                out.add("gambit_pawn")
    q0, q1 = _queens(root), _queens(endb)
    ours = [(boards[i], moves[i]) for i in range(0, min(len(moves), 2 * HORIZON), 2)]
    qtrade = False
    if q0[0] >= 1 and q0[1] >= 1 and q1 == (0, 0) and abs(end - m0) <= 1:
        for bd, m in ours[:2]:                         # we start it: take their queen or offer ours
            if bd.piece_at(m.from_square).piece_type != chess.QUEEN:
                continue
            tgt = bd.piece_at(m.to_square)
            aft = bd.copy(stack=False)
            aft.push(m)
            if (tgt is not None and tgt.piece_type == chess.QUEEN) or \
                    aft.attackers(not us, m.to_square) & aft.pieces(chess.QUEEN, not us):
                qtrade = True
                break
    if qtrade:
        out.add("queen_trade")
    after = boards[1]
    p = root.piece_at(first.from_square)
    capture, check = root.is_capture(first), root.gives_check(first)
    if (wp >= 0.7 and q1 == q0 and _npm(endb) <= _npm(root) - 6 and -1 <= end - m0 <= 0
            and sum(bd.gives_check(m) for bd, m in ours) < 2):
        offer = any(VAL[after.piece_at(a).piece_type] == VAL[p.piece_type]
                    for a in after.attackers(not us, first.to_square))
        victim = root.piece_at(first.to_square) if capture else None
        if (victim is not None and victim.piece_type != chess.PAWN) or (offer and p.piece_type != chess.PAWN):
            out.add("simplify_ahead")
    if p.piece_type == chess.PAWN and not capture and not first.promotion:
        hits = after.attacks(first.to_square) & after.pieces(chess.PAWN, not us)
        levers = after.attackers(not us, first.to_square) & after.pieces(chess.PAWN, not us)
        big = any(after.piece_at(t).piece_type not in (chess.PAWN, chess.KING)
                  for t in after.attacks(first.to_square) & after.occupied_co[not us])
        if (hits or levers) and not big:
            out.add("pawn_break")
        if (first.from_square in passed_pawns(root, us) and _rel_rank(first.to_square, us) >= 4 and not big
                and not check):
            out.add("passer_push")
    if (strict_endgame(root) and p.piece_type == chess.KING and not capture
            and not (len(boards) > 2 and boards[2].is_check())):
        kmoves = [m for bd, m in ours[:3]
                  if bd.piece_at(m.from_square).piece_type == chess.KING and not bd.is_capture(m)]
        if len(kmoves) >= 2:
            squares = [kmoves[0].from_square] + [m.to_square for m in kmoves]
            dist = [min(chess.square_distance(q, g) for g in (chess.D4, chess.E4, chess.D5, chess.E5))
                    for q in squares]
            if len(set(squares)) == len(squares) and dist[-1] < dist[0]:
                out.add("king_activation")
    if wp >= 0.4 and check:
        chk = [(bd.gives_check(m), bd.is_capture(m)) for bd, m in ours[:3]]
        run = len(chk) == 3 and all(c for c, _ in chk) and sum(not x for _, x in chk) >= 2
        mate = any(bd.is_checkmate() for bd in boards[1::2])
        seen = [bd.board_fen() + str(bd.turn) for bd in boards]
        if (run or mate) and len(set(seen)) == len(seen):
            out.add("forcing_checks")
    if end >= m0 + 2:                                  # tactics must win material, net
        if _cashed_fork(boards, moves, us):
            out.add("fork")
        if not capture and _cashed_line_tactic(boards, moves, us):
            out.add("line_tactic")
    if (p.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) and not capture and not check
            and _rel_rank(first.to_square, us) < _rel_rank(first.from_square, us)
            and (root.attackers(not us, first.from_square) or _rel_rank(first.from_square, us) >= 4)
            and not _new_big_attack(after, first.to_square)
            and not (p.piece_type == chess.QUEEN
                     and after.attackers(not us, first.to_square) & after.pieces(chess.QUEEN, not us))):
        out.add("retreat")
    th = threats(root) if root_threats is None else root_threats
    if (th and not capture and not check and first.from_square not in th and not _wins_now(after)
            and not _new_big_attack(after, first.to_square) and min([end] + s) >= m0):
        out.add("quiet_defence")
    if (chess.square_file(first.to_square) + chess.square_rank(first.to_square)) % 2 == 1:
        out.add(CONTROL)
    return out, {"dip": dip, "queens_off": q0 != (0, 0) and _queens(boards[-1]) == (0, 0)}


def first_kind(fen: str, move: str) -> str:
    b = chess.Board(fen)
    m = chess.Move.from_uci(move)
    if b.is_castling(m):
        return "castles"
    if b.is_capture(m):
        return "capture"
    if b.gives_check(m):
        return "check"
    return "pawn move" if b.piece_at(m.from_square).piece_type == chess.PAWN else "quiet piece move"


def _row(item):
    pid, fen, best, alt, played, is_best, wp_best, wp_played, wp_alt = item
    try:
        b = chess.Board(fen)
        ph = phase(b)
        kw = {"ph": ph, "root_threats": threats(b), "wp": wp_best}
        pb = detect(fen, best, **kw)[0]
        pp = pb if is_best else detect(fen, played, **kw)[0]
        if not pb and not pp:
            return []
        pa, fa = detect(fen, alt, **kw) if alt else (set(), {"dip": 0, "queens_off": False})
        if len(alt) > 2:                # the 2nd-best line doing the same one move later doesn't count
            b2 = b.copy(stack=False)
            for u in alt[:2]:
                b2.push_uci(u)
            pa = pa | detect(b2.fen(), alt[2:], ph=ph, root_threats=threats(b2), wp=wp_best)[0]
        gap = wp_best - wp_alt if wp_alt == wp_alt else 1.0       # no 2nd move: forced, full gap
        alt_txt = ",".join(sorted(pp - {CONTROL})) or (first_kind(fen, played[0]) if played else "")
        out = []
        for x in sorted(pb | pp):
            opp = x in pb and x not in pa and gap >= GAP
            if PATTERNS[x][0] == "sacrifice" and (fa["dip"] > 0 or gap < SAC_GAP):
                opp = False             # the 2nd-best line loses material too: giving it isn't the choice
            if x == "queen_trade" and fa["queens_off"]:
                opp = False
            out.append((pid, x, PATTERNS[x][0], bool(opp), x in pp, "" if x in pp else alt_txt,
                        float(max(0.0, wp_best - wp_played)), float(min(1.0, gap / CONF_GAP)) if opp else np.nan))
        return out
    except Exception as e:                                       # one bad line must not kill the pass
        print(f"  linepatterns: {pid}: {type(e).__name__}: {e}", flush=True)
        return []


COLS = ["pos_id", "pattern", "family", "opportunity", "took", "alternative", "cost", "confidence"]


def tag(df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """Pattern rows for analysed positions (columns fen, best_line, alt_line, played_line, played_is_best,
    wp_best, wp_played, cp_alt)."""
    wp_alt = 1.0 / (1.0 + np.exp(-0.00368208 * df.cp_alt.astype(float).to_numpy()))
    items = list(zip(df.pos_id, df.fen, df.best_line.map(list), df.alt_line.map(lambda x: list(x) if x is not None else []),
                     df.played_line.map(list), df.played_is_best.astype(bool), df.wp_best.astype(float),
                     df.wp_played.astype(float), wp_alt))
    if workers == 1 or len(items) < 500:
        res = [_row(i) for i in items]
    else:
        with mp.Pool(workers or max(1, mp.cpu_count() - 2)) as pool:
            res = pool.map(_row, items, chunksize=200)
    return pd.DataFrame([r for rs in res for r in rs], columns=COLS)


def stability(cfg) -> dict:
    """pattern -> flip rate of the opportunity label at 4x nodes (scripts/diagnostics/lp_stability.py), if run."""
    p = data_dir(cfg, "population") / "linepatterns_stability.json"
    if not p.exists():
        return {}
    return {k: v["flip_rate"] for k, v in json.loads(p.read_text(encoding="utf-8")).get("patterns", {}).items()
            if v.get("flip_rate") is not None}


def pattern_table(cfg, which: str, df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """Rows for every position of df (settled lines), cached in <set>/linepatterns.parquet (pos_id-keyed,
    rebuilt when VERSION changes). Confidence includes the 4x-nodes flip rate when it has been measured."""
    path = data_dir(cfg, which) / "linepatterns.parquet"
    done = data_dir(cfg, which) / "linepatterns_done.parquet"     # every tagged pos_id (most have no row)
    old, seen = None, set()
    if path.exists() and done.exists():
        o, d = pd.read_parquet(path), pd.read_parquet(done)
        if len(d) and (d.v == VERSION).all():
            old, seen = o, set(d.pos_id)
    todo = df[~df.pos_id.isin(seen)]
    if len(todo):
        print(f"  line patterns: tagging {which} {len(todo):,} positions")
        new = tag(todo, workers)
        old = new if old is None else pd.concat([old[COLS], new], ignore_index=True)
        seen |= set(todo.pos_id)
        for p_, frame in ((path, old), (done, pd.DataFrame({"pos_id": sorted(seen), "v": VERSION}))):
            tmp = p_.with_suffix(".tmp")
            frame.to_parquet(tmp, index=False)
            tmp.replace(p_)
    out = old[old.pos_id.isin(set(df.pos_id))].copy() if old is not None else pd.DataFrame(columns=COLS)
    flip = stability(cfg)
    if flip:
        out["confidence"] = out.confidence * (1 - out.pattern.map(flip).fillna(0.0))
    return out.reset_index(drop=True)
