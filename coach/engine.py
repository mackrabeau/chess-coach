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


def win_prob(cp: float) -> float:
    """Lichess win-probability model, in [0, 1], from the side to move."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def _cp(score: chess.engine.PovScore, pov: chess.Color) -> int:
    return score.pov(pov).score(mate_score=MATE_CP)


class Analyzer:
    def __init__(self, ecfg: dict, stockfish_path: str):
        self.ecfg = ecfg
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
