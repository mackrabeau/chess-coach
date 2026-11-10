"""Small HTML helpers for the inspection and player reports (single self-contained file)."""
from __future__ import annotations

import html

import chess
import chess.svg

GREEN, RED = "#1f9d55cc", "#d64545cc"

CSS = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --card:#ffffff; --line:#e4e1d8; --accent:#1f9d55; --bad:#c23b3b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161614; --fg:#ecebe6; --muted:#9a998f; --card:#1f1f1c; --line:#33322d; --accent:#4cc38a; --bad:#ef6b6b; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 40px 0 6px; padding-top: 16px; border-top: 1px solid var(--line); }
.muted { color: var(--muted); }
.tag { display:inline-block; font-size:12px; padding:1px 8px; border-radius:999px; border:1px solid var(--line); margin-right:6px; }
.tag.stable { border-color: var(--accent); color: var(--accent); }
.feats { margin: 8px 0 14px; padding: 0; list-style: none; display:flex; flex-wrap:wrap; gap:6px; }
.feats li { background: var(--card); border:1px solid var(--line); border-radius:6px; padding:2px 8px; font-size:13px; }
.up { color: var(--accent); font-weight: 600; } .down { color: var(--bad); font-weight: 600; }
.grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; }
.card { background: var(--card); border:1px solid var(--line); border-radius: 10px; padding: 10px; font-size: 13px; }
.card svg { width: 100%; height: auto; display:block; }
.card .line { font-family: ui-monospace, Consolas, monospace; font-size: 12px; word-break: break-word; }
.best { color: var(--accent); } .played { color: var(--bad); }
table { border-collapse: collapse; margin: 10px 0; font-size: 14px; }
td, th { padding: 4px 10px; border-bottom: 1px solid var(--line); text-align: right; }
th:first-child, td:first-child { text-align: left; }
details summary { cursor: pointer; color: var(--muted); }
"""


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>")


def feats_html(desc) -> str:
    items = []
    for arrow, name, w in desc:
        cls = "up" if arrow == "↑" else "down"
        items.append(f"<li><span class='{cls}'>{arrow}</span> {html.escape(name)} "
                     f"<span class='muted'>{w:.0%}</span></li>")
    return f"<ul class='feats'>{''.join(items)}</ul>"


def line_san(fen: str, line) -> str:
    b = chess.Board(fen)
    try:
        return b.variation_san([chess.Move.from_uci(m) for m in line])
    except Exception:
        return " ".join(line)


def board_svg(fen: str, best: str | None = None, played: str | None = None, size: int = 260) -> str:
    b = chess.Board(fen)
    arrows = []
    if played and played != best:
        m = chess.Move.from_uci(played)
        arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color=RED))
    if best:
        m = chess.Move.from_uci(best)
        arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color=GREEN))
    return chess.svg.board(b, orientation=b.turn, arrows=arrows, size=size, coordinates=True)


def _mover(fen: str) -> str:
    return "White" if chess.Board(fen).turn == chess.WHITE else "Black"


def position_card(r, *, puzzle: bool = False, link: str | None = None) -> str:
    """r: a row with fen, move, best_move, best_line, played_line, wp_loss, phase."""
    head = (f"{_mover(r.fen)} to move · {r.phase} · loss {r.wp_loss:.0%}"
            if not puzzle else f"{_mover(r.fen)} to move · {r.phase} · find the best move")
    if link:
        head += f" · <a href='{html.escape(link)}' target='_blank'>game</a>"
    if puzzle:
        svg = board_svg(r.fen)
        sol = (f"<details><summary>solution</summary><div class='line best'>"
               f"{html.escape(line_san(r.fen, r.best_line))}</div>"
               f"<div class='muted'>{board_svg(r.fen, r.best_move)}</div></details>")
        return f"<div class='card'>{svg}<div class='muted'>{head}</div>{sol}</div>"
    svg = board_svg(r.fen, r.best_move, r.move)
    return (f"<div class='card'>{svg}<div class='muted'>{head}</div>"
            f"<div class='line best'>best: {html.escape(line_san(r.fen, r.best_line))}</div>"
            f"<div class='line played'>played: {html.escape(line_san(r.fen, r.played_line))}</div></div>")
