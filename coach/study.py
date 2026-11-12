"""Lichess study export: chapters as PGN games (variations + comments), at most 64 per file.

Import at lichess.org/study → new study → "Import PGN" (one chapter per game in the file). The
chapter name comes from the Event tag; Orientation sets which side is at the bottom (Lichess reads
it in study exports; if an import ignores it, flip the board once per chapter).
"""
from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn

MAX_CHAPTERS = 64          # Lichess limit per study


def chapter(title: str, orientation: str = "white", fen: str | None = None) -> chess.pgn.Game:
    g = chess.pgn.Game()
    g.headers.clear()
    g.headers["Event"] = title
    g.headers["Orientation"] = orientation
    if fen and fen != chess.STARTING_FEN:
        g.setup(chess.Board(fen))
    return g


def puzzle_chapter(title: str, fen: str, solution: list[str], comment: str = "",
                   orientation: str | None = None) -> chess.pgn.Game:
    """A 'find the move' chapter: start position + solution line; the comment explains why."""
    b = chess.Board(fen)
    g = chapter(title, orientation or ("white" if b.turn else "black"), fen)
    node = g
    for u in solution:
        mv = chess.Move.from_uci(u)
        if mv not in node.board().legal_moves:
            break
        node = node.add_variation(mv)
    if comment and node is not g:
        g.variations[0].comment = comment
    return g


def write_studies(chapters: list[chess.pgn.Game], stem: Path) -> list[Path]:
    """stem.pgn, or stem_1.pgn, stem_2.pgn, ... when there are more than 64 chapters."""
    out = []
    parts = [chapters[i:i + MAX_CHAPTERS] for i in range(0, len(chapters), MAX_CHAPTERS)] or [[]]
    for k, part in enumerate(parts, 1):
        p = stem.with_suffix(".pgn") if len(parts) == 1 else stem.with_name(f"{stem.name}_{k}.pgn")
        exporter = lambda g: g.accept(chess.pgn.StringExporter(headers=True, variations=True, comments=True))
        p.write_text("\n\n".join(exporter(g) for g in part) + "\n", encoding="utf-8")
        out.append(p)
    return out
