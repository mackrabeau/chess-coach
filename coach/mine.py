"""Turn games into decision positions (one row per move we want to analyse).

A decision row is the position *before* a move, the move actually played, the mover's
clock before moving, and the previous positions (Maia-3 conditions on history).
"""
from __future__ import annotations

import io
import random
from collections import deque
from typing import Callable, Iterable

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq

from .pgn_stream import parse_time_control

SCHEMA = pa.schema(
    [
        ("pos_id", pa.string()),
        ("source", pa.string()),
        ("game_id", pa.string()),
        ("date", pa.string()),
        ("ply", pa.int32()),
        ("fen", pa.string()),
        ("history_fens", pa.list_(pa.string())),  # oldest first, most recent last
        ("move", pa.string()),                    # UCI of the move actually played
        ("mover_white", pa.bool_()),
        ("mover_rating", pa.int32()),
        ("opp_rating", pa.int32()),
        ("clock_before", pa.float32()),           # seconds, NaN if unknown
        ("base_seconds", pa.int32()),
        ("increment", pa.int32()),
        ("game_plies", pa.int32()),
        ("result", pa.float32()),                 # final score for the mover: 1 / 0.5 / 0, NaN if unknown
    ]
)


def mover_score(result_header: str | None, mover_white: bool) -> float:
    """PGN Result header -> the mover's final score (1 win, 0.5 draw, 0 loss; NaN if unknown)."""
    white = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get((result_header or "").strip())
    if white is None:
        return float("nan")
    return white if mover_white else 1.0 - white


def is_rapid(tc: str | None, fcfg: dict) -> bool:
    parsed = parse_time_control(tc)
    if parsed is None:
        return False
    base, inc = parsed
    est = base + 40 * inc
    return fcfg["rapid_min_est_seconds"] <= est < fcfg["rapid_max_est_seconds"]


def _int(x: str | None, default: int = -1) -> int:
    try:
        return int(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def extract_positions(
    game_text: str,
    headers: dict[str, str],
    *,
    source: str,
    game_id: str,
    fcfg: dict,
    mover_ok: Callable[[bool], bool],
    rng: random.Random | None = None,
    max_per_game: int | None = None,
) -> list[dict]:
    """Return decision rows for one game.

    mover_ok(is_white) decides which side's moves we keep (both for the population,
    only the player's side for the player).
    """
    game = chess.pgn.read_game(io.StringIO(game_text))
    if game is None or game.errors:
        return []
    if headers.get("Variant", "Standard") not in ("Standard", "Chess"):
        return []
    if headers.get("SetUp") == "1" or "FEN" in headers:
        return []  # non-standard start position

    moves = list(game.mainline())
    n_plies = len(moves)
    if n_plies < fcfg["min_game_plies"]:
        return []

    tc = parse_time_control(headers.get("TimeControl"))
    base, inc = tc if tc else (-1, 0)
    w_elo, b_elo = _int(headers.get("WhiteElo")), _int(headers.get("BlackElo"))
    date = headers.get("UTCDate") or headers.get("Date") or ""

    board = game.board()
    history: deque[str] = deque(maxlen=fcfg["history_plies"])
    # Clock remaining *before* each side's next move. Starts at the base time.
    clock = {chess.WHITE: float(base), chess.BLACK: float(base)}
    rows: list[dict] = []

    for node in moves:
        mover = board.turn
        ply = board.ply()
        fen = board.fen()
        clk_before = clock[mover] if base > 0 else float("nan")
        if (
            ply >= fcfg["min_ply"]
            and mover_ok(mover == chess.WHITE)
            # NaN fails this -> skipped; min_clock_seconds None = no clock check (imported OTB games)
            and (fcfg["min_clock_seconds"] is None or clk_before >= fcfg["min_clock_seconds"])
            and board.legal_moves.count() > 1  # forced moves teach nothing
        ):
            rows.append(
                {
                    "pos_id": f"{source}:{game_id}:{ply}",
                    "source": source,
                    "game_id": game_id,
                    "date": date,
                    "ply": ply,
                    "fen": fen,
                    "history_fens": list(history),
                    "move": node.move.uci(),
                    "mover_white": mover == chess.WHITE,
                    "mover_rating": w_elo if mover == chess.WHITE else b_elo,
                    "opp_rating": b_elo if mover == chess.WHITE else w_elo,
                    "clock_before": clk_before,
                    "base_seconds": base,
                    "increment": inc,
                    "game_plies": n_plies,
                    "result": mover_score(headers.get("Result"), mover == chess.WHITE),
                }
            )
        history.append(fen)
        board.push(node.move)
        c = node.clock()  # remaining after this move
        if c is not None:
            clock[mover] = float(c)
        elif base > 0:
            clock[mover] = float("nan")  # lost track; later moves get skipped

    if max_per_game is not None and len(rows) > max_per_game:
        rng = rng or random.Random(0)
        rows = rng.sample(rows, max_per_game)
        rows.sort(key=lambda r: r["ply"])
    return rows


class ParquetSink:
    """Buffered parquet writer (one file, row groups of `flush_rows`)."""

    def __init__(self, path, flush_rows: int = 50_000):
        self.path = path
        self.flush_rows = flush_rows
        self.buf: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.n = 0

    def add(self, rows: Iterable[dict]) -> None:
        self.buf.extend(rows)
        if len(self.buf) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        table = pa.Table.from_pylist(self.buf, schema=SCHEMA)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, SCHEMA, compression="zstd")
        self.writer.write_table(table)
        self.n += len(self.buf)
        self.buf = []

    def close(self) -> None:
        self.flush()
        if self.writer is None:  # still produce a valid empty file
            pq.write_table(pa.Table.from_pylist([], schema=SCHEMA), self.path)
        else:
            self.writer.close()
