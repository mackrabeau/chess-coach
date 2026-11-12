"""Stockfish analysis of decision positions.

For each position (side to move = the player who moved):
  * root search, MultiPV=2  -> best move a*, its eval, 2nd-best move + eval
  * root search restricted to the played move (searchmoves) -> eval of a_p
  * win-probability loss, classification (ok / mistake / blunder), critical flag
  * principal variations of exactly `pv_plies` (even) for a*, a_p and the 2nd-best move

Evals are always from the mover's point of view. Every position starts a fresh
`ucinewgame` (hash cleared), so results don't depend on processing order.
"""
from __future__ import annotations

import math

import chess
import chess.engine

import pyarrow as pa

MATE_CP = 10_000

# Fixed output schema so every part file matches (a chunk with no 2nd-best moves would
# otherwise infer a null column type and break reading the parts back together).
RESULT_SCHEMA = pa.schema(
    [
        ("pos_id", pa.string()),
        ("error", pa.string()),
        ("best_move", pa.string()),
        ("cp_best", pa.int32()),
        ("cp_played", pa.int32()),
        ("alt_move", pa.string()),
        ("cp_alt", pa.int32()),
        ("wp_best", pa.float32()),
        ("wp_played", pa.float32()),
        ("wp_loss", pa.float32()),
        ("label", pa.string()),
        ("critical", pa.bool_()),
        ("played_is_best", pa.bool_()),
        ("best_line", pa.list_(pa.string())),
        ("played_line", pa.list_(pa.string())),
        ("alt_line", pa.list_(pa.string())),
        ("best_line_complete", pa.bool_()),
        ("played_line_complete", pa.bool_()),
        ("alt_line_complete", pa.bool_()),
        ("n_legal", pa.int32()),
    ]
)


LINES = ("best", "played", "alt")

# Output of the settling pass (03_settle_lines.py): each line extended until quiet.
SETTLE_SCHEMA = pa.schema(
    [("pos_id", pa.string()), ("error", pa.string())]
    + [f for L in LINES for f in (
        (f"{L}_line_q", pa.list_(pa.string())),  # original line + extension (even length)
        (f"{L}_q_plies", pa.int32()),             # plies added
        (f"{L}_q_trigger", pa.string()),          # why the first extension was needed ("" = already quiet)
        (f"{L}_q_end", pa.string()),              # quiet / cap / game_over / short
    )]
)


# Deep pass (06c_deepen_tactics.py): blunders with no material swing in the settled lines,
# continued much further to see whether material changes hands beyond the horizon.
DEEP_SCHEMA = pa.schema([
    ("pos_id", pa.string()), ("error", pa.string()),
    ("best_line_deep", pa.list_(pa.string())), ("played_line_deep", pa.list_(pa.string())),
])


def win_prob(cp: float) -> float:
    """Lichess win-probability model, in [0, 1], from the side to move."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def _cp(score: chess.engine.PovScore, pov: chess.Color) -> int:
    return score.pov(pov).score(mate_score=MATE_CP)


class Analyzer:
    def __init__(self, ecfg: dict, stockfish_path: str, qcfg: dict | None = None):
        self.ecfg = ecfg
        self.qcfg = qcfg
        self.path = stockfish_path
        if ecfg["pv_plies"] % 2:
            raise ValueError("engine.pv_plies must be even (end position must have the root's side to move)")
        self.engine: chess.engine.SimpleEngine | None = None
        self._start()

    def _start(self) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self.engine.configure({"Threads": 1, "Hash": self.ecfg["hash_mb"]})

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def _analyse(self, board, nodes, game, **kw):
        try:
            return self.engine.analyse(board, chess.engine.Limit(nodes=nodes), game=game, **kw)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
            # Crashed engine: restart once and retry.
            self.close()
            self._start()
            return self.engine.analyse(board, chess.engine.Limit(nodes=nodes), game=game, **kw)

    def _line(self, board: chess.Board, pv: list[chess.Move], game) -> tuple[list[str], bool]:
        """Return (uci line of length pv_plies if possible, complete_flag).

        PVs often come back short at fixed nodes; extend them with cheap searches from
        the end of the line. If the game ends early, trim to the longest even length so
        the end position keeps the root's side to move.
        """
        L = self.ecfg["pv_plies"]
        b = board.copy(stack=False)
        line: list[chess.Move] = []
        for mv in pv[:L]:
            if mv not in b.legal_moves:
                break
            b.push(mv)
            line.append(mv)
        while len(line) < L and not b.is_game_over(claim_draw=True):
            info = self._analyse(b, self.ecfg["extend_nodes"], game)
            nxt = info.get("pv") or []
            if not nxt:
                break
            b.push(nxt[0])
            line.append(nxt[0])
        complete = len(line) == L
        if len(line) % 2:
            line = line[:-1]
        return [m.uci() for m in line], complete

    def analyse_position(self, fen: str, played_uci: str) -> dict:
        ecfg = self.ecfg
        board = chess.Board(fen)
        mover = board.turn
        played = chess.Move.from_uci(played_uci)
        game = object()  # new object => python-chess sends ucinewgame (clears hash)

        n_legal = board.legal_moves.count()
        infos = self._analyse(board, ecfg["nodes"], game, multipv=min(2, n_legal))
        best = infos[0]
        a_star = best["pv"][0]
        cp_best = _cp(best["score"], mover)
        if len(infos) > 1 and infos[1].get("pv"):
            alt, cp_alt = infos[1]["pv"][0], _cp(infos[1]["score"], mover)
            alt_pv = infos[1]["pv"]
        else:
            alt, cp_alt, alt_pv = None, None, []

        if played == a_star:
            cp_played, played_pv = cp_best, best["pv"]
        elif alt is not None and played == alt:
            cp_played, played_pv = cp_alt, alt_pv
        else:
            info_p = self._analyse(board, ecfg["nodes"], game, root_moves=[played])
            cp_played, played_pv = _cp(info_p["score"], mover), info_p.get("pv") or [played]
            if played_pv[0] != played:  # paranoia: searchmoves should guarantee this
                played_pv = [played]

        wp_best, wp_played = win_prob(cp_best), win_prob(cp_played)
        loss = max(0.0, wp_best - wp_played)
        if loss < ecfg["ok_loss"]:
            label = "ok"
        elif loss <= ecfg["blunder_loss"]:
            label = "mistake"
        else:
            label = "blunder"
        gap = (wp_best - win_prob(cp_alt)) if cp_alt is not None else None
        critical = gap is not None and gap >= ecfg["critical_gap"]

        best_line, best_complete = self._line(board, best["pv"], game)
        played_line, played_complete = self._line(board, played_pv, game)
        if alt is not None:
            alt_line, alt_complete = self._line(board, alt_pv, game)
        else:
            alt_line, alt_complete = [], False

        return {
            "best_move": a_star.uci(),
            "cp_best": cp_best,
            "cp_played": cp_played,
            "alt_move": alt.uci() if alt is not None else None,
            "cp_alt": cp_alt,
            "wp_best": wp_best,
            "wp_played": wp_played,
            "wp_loss": loss,
            "label": label,
            "critical": critical,
            "played_is_best": played == a_star,
            "best_line": best_line,
            "played_line": played_line,
            "alt_line": alt_line,
            "best_line_complete": best_complete,
            "played_line_complete": played_complete,
            "alt_line_complete": alt_complete,
            "n_legal": n_legal,
        }

    # ------------------------------------------------------------------ settling pass

    def _unsettled(self, b: chess.Board, game) -> tuple[str, chess.Move]:
        """(reason this position is not quiet, or "", engine's move here).

        Not quiet if: in check; few legal moves (forced); the engine's move is a capture,
        promotion or check; or it's an "only move" (beats the 2nd best by >= forced_gap).
        """
        q = self.qcfg
        n = b.legal_moves.count()
        infos = self._analyse(b, q["nodes"], game, multipv=min(2, n))
        mv = infos[0]["pv"][0]
        if b.is_check():
            return "in_check", mv
        if n <= q["forced_max_legal"]:
            return "few_moves", mv
        if b.is_capture(mv) or mv.promotion or b.gives_check(mv):
            return "tactical", mv
        if len(infos) > 1 and infos[1].get("pv"):
            stm = b.turn
            gap = win_prob(_cp(infos[0]["score"], stm)) - win_prob(_cp(infos[1]["score"], stm))
            if gap >= q["forced_gap"]:
                return "only_move", mv
        return "", mv

    def settle_line(self, fen: str, line: list[str], game) -> dict:
        """Extend `line` with engine moves, two plies at a time, until the end position
        (root's side to move) is quiet, or max_extra_plies is reached."""
        if len(line) < 2:
            return {"line": list(line), "plies": 0, "trigger": "", "end": "short"}
        b = chess.Board(fen)
        for u in line:
            b.push_uci(u)
        ext: list[chess.Move] = []
        trigger, end = "", "quiet"
        while True:
            if b.is_game_over(claim_draw=True):
                end = "game_over"
                break
            reason, mv = self._unsettled(b, game)
            if len(ext) % 2 == 0:  # root's side to move: allowed to stop here
                if not reason:
                    break
                if not ext:
                    trigger = reason
            if len(ext) >= self.qcfg["max_extra_plies"]:
                end = "cap"
                break
            b.push(mv)
            ext.append(mv)
        if len(ext) % 2:
            ext.pop()  # keep the root's side to move at the end
        return {"line": list(line) + [m.uci() for m in ext], "plies": len(ext),
                "trigger": trigger, "end": end}

    def settle_position(self, fen: str, lines: dict[str, list[str]]) -> dict:
        game = object()  # one ucinewgame per position
        out = {}
        for L in LINES:
            r = self.settle_line(fen, lines[L], game)
            out[f"{L}_line_q"] = r["line"]
            out[f"{L}_q_plies"] = r["plies"]
            out[f"{L}_q_trigger"] = r["trigger"]
            out[f"{L}_q_end"] = r["end"]
        return out

    # ------------------------------------------------------------------ deep pass

    def deepen_line(self, fen: str, line: list[str], game, plies: int, nodes: int) -> list[str]:
        """Continue `line` with the engine's best move for up to `plies` more plies
        (stops at game over). Trimmed to even length so the root's side is to move at the
        end, except when the last move is checkmate (only used for classification, not features)."""
        b = chess.Board(fen)
        for u in line:
            b.push_uci(u)
        ext: list[str] = []
        while len(ext) < plies and not b.is_game_over(claim_draw=True):
            info = self._analyse(b, nodes, game)
            pv = info.get("pv") or []
            if not pv:
                break
            b.push(pv[0])
            ext.append(pv[0].uci())
        if len(ext) % 2 and not b.is_checkmate():
            ext.pop()
        return list(line) + ext

    # ------------------------------------------------------------------ practical value (12)

    def candidates(self, fen: str, played: str | None, k: int, nodes: int, delta: float):
        """Stockfish's top-k moves within `delta` win probability of the best, plus the played
        move. Returns ([(uci, wp for the mover)], best_uci)."""
        b = chess.Board(fen)
        me = b.turn
        game = object()
        n = b.legal_moves.count()
        infos = self._analyse(b, nodes, game, multipv=min(k, n))
        cands = [(i["pv"][0].uci(), win_prob(_cp(i["score"], me))) for i in infos if i.get("pv")]
        best_w = cands[0][1]
        keep = [(m, w) for m, w in cands if w >= best_w - delta]
        if played and played not in {m for m, _ in keep}:
            info = self._analyse(b, nodes, game, root_moves=[chess.Move.from_uci(played)])
            keep.append((played, win_prob(_cp(info["score"], me))))
        return keep, cands[0][0]

    def move_values(self, fen: str, moves: list[str], nodes: int, multipv: int = 3):
        """Top `multipv` moves plus every move in `moves`, each with its win prob for the side
        to move. Returns (best_uci, {uci: wp}) sorted best first."""
        b = chess.Board(fen)
        me = b.turn
        game = object()
        infos = self._analyse(b, nodes, game, multipv=min(multipv, b.legal_moves.count()))
        vals = {i["pv"][0].uci(): win_prob(_cp(i["score"], me)) for i in infos if i.get("pv")}
        best = next(iter(vals), None)
        for u in moves:
            if u in vals:
                continue
            mv = chess.Move.from_uci(u)
            if mv not in b.legal_moves:
                continue
            info = self._analyse(b, nodes, game, root_moves=[mv])
            vals[u] = win_prob(_cp(info["score"], me))
        return best, dict(sorted(vals.items(), key=lambda kv: -kv[1]))

    def expand(self, fen: str, prefix: list[str], moves: list[str], nodes: int, root_eval: bool = True):
        """Push `prefix`, then value the position (if root_eval) and each move in `moves` (plus the
        engine's best when root_eval). Values = win prob for the side to move at `fen`.
        Returns (best_uci or None, root value or None, {move: (value, engine's best next move or None)})."""
        b = chess.Board(fen)
        me = b.turn
        game = object()
        for u in prefix:
            b.push_uci(u)

        def val(bd):
            if bd.is_checkmate():
                return (0.0 if bd.turn == me else 1.0), None
            if bd.is_game_over(claim_draw=True):
                return 0.5, None
            info = self._analyse(bd, nodes, game)
            return win_prob(_cp(info["score"], me)), (info["pv"][0].uci() if info.get("pv") else None)

        best, v0 = None, None
        if root_eval:
            v0, best = val(b)
        out = {}
        for u in dict.fromkeys(list(moves) + ([best] if best else [])):
            b.push_uci(u)
            out[u] = val(b)
            b.pop()
        return best, v0, out

    def reply_values(self, fen: str, move: str, replies: list[str], nodes: int):
        """After `move`: the opponent's best reply r*, its value V* (mover's win prob) and the
        value after each listed reply. Terminal positions are scored directly."""
        b = chess.Board(fen)
        me = b.turn
        game = object()
        b.push_uci(move)
        if b.is_checkmate():
            return None, 1.0, {}
        if b.is_game_over(claim_draw=True):
            return None, 0.5, {}
        info = self._analyse(b, nodes, game)
        rstar = info["pv"][0].uci() if info.get("pv") else None
        vstar = win_prob(_cp(info["score"], me))
        vals = {}
        for r in replies:
            if r == rstar:
                vals[r] = vstar
                continue
            b.push_uci(r)
            if b.is_checkmate():
                vals[r] = 0.0                     # the mover got mated
            elif b.is_game_over(claim_draw=True):
                vals[r] = 0.5
            else:
                vals[r] = win_prob(_cp(self._analyse(b, nodes, game)["score"], me))
            b.pop()
        return rstar, vstar, vals
