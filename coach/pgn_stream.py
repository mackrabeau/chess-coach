"""Fast streaming over (possibly zstd-compressed) multi-game PGN files.

Splits the file into raw game texts and parses headers with a regex, so callers can
reject games on headers alone without paying for a full python-chess parse. That
matters for Lichess dumps, where almost every game gets rejected.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterator

_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".zst":
        import zstandard

        fh = open(path, "rb")
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_raw_games(path: str | Path) -> Iterator[tuple[dict[str, str], str]]:
    """Yield (headers, full_game_text) for every game in the file."""
    path = Path(path)
    headers: dict[str, str] = {}
    lines: list[str] = []
    in_moves = False
    with _open_text(path) as f:
        for line in f:
            if line.startswith("["):
                if in_moves:
                    # A header after movetext starts the next game.
                    yield headers, "".join(lines)
                    headers, lines, in_moves = {}, [], False
                m = _HEADER_RE.match(line)
                if m:
                    headers[m.group(1)] = m.group(2)
                lines.append(line)
            else:
                if line.strip():
                    in_moves = True
                lines.append(line)
        if headers and in_moves:
            yield headers, "".join(lines)


def parse_time_control(tc: str | None) -> tuple[int, int] | None:
    """'600+5' -> (600, 5); '600' -> (600, 0); daily/unknown -> None."""
    if not tc or tc == "-" or "/" in tc:
        return None
    try:
        if "+" in tc:
            base, inc = tc.split("+", 1)
            return int(base), int(inc)
        return int(tc), 0
    except ValueError:
        return None
