"""Follow-up lines: a move plus the engine's continuation, so a single move becomes an idea.

  follow_up(cfg, [(fen, move), ...]) -> {(fen, move): [move, reply, ...]}
      after the move, Stockfish's principal variation (extended with cheap searches when short), up to
      `lines.plies` plies in total. For a mistake that is the punishment; for the best move, the plan.
  expand_examples(cfg, findings)
      every finding example whose best/played line is one move long gets both lines expanded, and every
      drill puzzle gets its full solution (+ `wrong_line`, the refutation of the wrong move, if `wrong` is
      given). coach/findings.write calls this, so every module's openings show follow-ups automatically.

Cached per player in data/players/<you>/openings/lines/ (resumable part files keyed by fen|move|plies|nodes);
a later call with more nodes adds deeper results and the deepest is used.
"""
from __future__ import annotations

import multiprocessing as mp

import chess
import pyarrow as pa

from .config import data_dir, resolve
from .engine import Analyzer
from .parts import done_ids, next_part_index, read_parts, write_part

LINES = {"plies": 10, "nodes": 300000, "extend_nodes": 50000}
SCHEMA = pa.schema([("pos_id", pa.string()), ("fen", pa.string()), ("move", pa.string()), ("nodes", pa.int64()),
                    ("line", pa.list_(pa.string()))])

_an = None
_lc = {}


def _init(ecfg, sf, lc):
    global _an, _lc
    _an, _lc = Analyzer(ecfg, sf), lc


def _work(item):
    pid, fen, move = item
    try:
        b = chess.Board(fen)
        mv = chess.Move.from_uci(move)
        if mv not in b.legal_moves:
            return None
        b.push(mv)
        line, game = [move], object()
        nodes = _lc["nodes"]
        while len(line) < _lc["plies"] and not b.is_game_over(claim_draw=True):
            info = _an._analyse(b, nodes, game)
            pv = [m for m in info.get("pv", [])]
            if not pv:
                break
            for m in pv:
                if len(line) >= _lc["plies"] or m not in b.legal_moves:
                    break
                b.push(m)
                line.append(m.uci())
            nodes = _lc["extend_nodes"]                    # extend short PVs cheaply
        return {"pos_id": pid, "fen": fen, "move": move, "nodes": _lc["nodes"], "line": line}
    except Exception as e:
        print(f"  follow-up failed for {move}: {e}", flush=True)
        return None


def _key(fen, move, lc):
    return f"{fen}|{move}|{lc['plies']}|{lc['nodes']}"


def follow_up(cfg, pairs, workers=None) -> dict:
    lc = {**LINES, **cfg.get("lines", {})}
    d = data_dir(cfg, "openings", "lines")
    pairs = list(dict.fromkeys((f, m) for f, m in pairs if f and m))
    done = done_ids(d)
    todo = [(_key(f, m, lc), f, m) for f, m in pairs if _key(f, m, lc) not in done]
    if todo:
        print(f"  follow-up lines: {len(todo)} to compute ({lc['nodes']:,} nodes, {lc['plies']} plies)", flush=True)
        sf = str(resolve(cfg, cfg["paths"]["stockfish"]))
        idx, buf = next_part_index(d), []
        with mp.Pool(workers or cfg["engine"]["workers"], initializer=_init,
                     initargs=(dict(cfg["engine"]), sf, lc)) as pool:
            for i, r in enumerate(pool.imap_unordered(_work, todo, chunksize=2), 1):
                if r:
                    buf.append(r)
                if len(buf) >= 100 or i == len(todo):
                    write_part(d, idx, buf, SCHEMA)
                    idx, buf = idx + 1, []
    got = read_parts(d)
    out = {}
    if got is not None:
        for f, m, n, line in sorted(zip(got.fen, got.move, got.nodes, got.line), key=lambda t: t[2]):
            out[(f, m)] = list(line)                       # deepest last wins
    return {p: out[p] for p in pairs if p in out}


def expand_examples(cfg, items) -> None:
    """In place: one-move example lines and puzzle solutions -> full follow-ups."""
    pairs = []
    for f in items:
        for e in f.get("examples", []):
            for k in ("best", "played"):
                if len(e.get(k) or []) == 1:
                    pairs.append((e["fen"], e[k][0]))
        for z in (f.get("drill") or {}).get("puzzles", []):
            if len(z.get("solution") or []) == 1:
                pairs.append((z["fen"], z["solution"][0]))
            if z.get("wrong"):
                pairs.append((z["fen"], z["wrong"]))
    if not pairs:
        return
    L = follow_up(cfg, pairs)
    for f in items:
        for e in f.get("examples", []):
            for k in ("best", "played"):
                if len(e.get(k) or []) == 1 and (e["fen"], e[k][0]) in L:
                    e[k] = L[(e["fen"], e[k][0])]
        for z in (f.get("drill") or {}).get("puzzles", []):
            if len(z.get("solution") or []) == 1 and (z["fen"], z["solution"][0]) in L:
                z["solution"] = L[(z["fen"], z["solution"][0])]
            if z.get("wrong") and (z["fen"], z["wrong"]) in L:
                z["wrong_line"] = L[(z["fen"], z["wrong"])]
