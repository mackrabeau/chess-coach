"""Small HTML helpers for the inspection and player reports (single self-contained file).

Shares the colour tokens, typography and theme setting of the main report (coach/webreport.py)."""
from __future__ import annotations

import html

import chess
import chess.svg

from .webreport import BASE_CSS, THEME_JS

GREEN, RED = "#15781bd9", "#c8281ecc"
# same board colours as the main report
BOARD = {"square light": "#f0d9b5", "square dark": "#b58863", "square light lastmove": "#cdd26a",
         "square dark lastmove": "#aaa23b", "margin": "#2b2b2e", "coord": "#e8e8e8"}

CSS = BASE_CSS + """
main { max-width: 1180px; margin: 0 auto; padding: 1.5rem 1.25rem 4rem; }
h1 { font-size: 1.6rem; margin: .4rem 0 .3rem; }
h2 { margin: 2.6rem 0 .6rem; padding-top: 1.1rem; border-top: 1px solid var(--line); }
.backnav { font-size: .92rem; margin: 0 0 1rem; }
.tag { display:inline-block; font-size:.78rem; padding:.05rem .55rem; border-radius:4px; border:1px solid var(--line);
  margin-right:6px; }
.tag.stable { border-color: var(--good); color: var(--good); }
.feats { margin: .5rem 0 1rem; padding: 0; list-style: none; display:flex; flex-wrap:wrap; gap:6px; max-width:none; }
.feats li { background: var(--surface); border:1px solid var(--line); border-radius:6px; padding:.1rem .6rem;
  font-size:.88rem; max-width:none; }
.up { color: var(--good); font-weight: 600; } .down { color: var(--bad); font-weight: 600; }
.grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 1rem; }
.card { background: var(--surface); border:1px solid var(--line); border-radius: 8px; padding: .7rem;
  font-size: .9rem; line-height: 1.5; box-shadow: var(--shadow); }
.card svg { width: 100%; height: auto; display:block; border-radius: 3px; }
.card > .muted { margin: .45rem 0 .2rem; }
.card .line { font-size: .88rem; word-break: break-word; margin-top: .2rem; }
.best { color: var(--good); } .played { color: var(--bad); }
td, th { text-align: right; } th:first-child, td:first-child { text-align: left; }
details summary { cursor: pointer; color: var(--muted); }
.card.has-ends { position: relative; cursor: zoom-in; outline: none; }
.card .ends { display: none; position: absolute; top: 0; left: 100%; margin-left: 8px; z-index: 20;
  grid-template-columns: 1fr 1fr; gap: 10px; width: 470px; padding: 10px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,.18); }
.card .ends.flip { left: auto; right: 100%; margin-left: 0; margin-right: 8px; }
.card .ends.below { left: 0; right: auto; top: 100%; margin: 6px 0 0; width: min(470px, 92vw); }
.card.has-ends:hover .ends, .card.has-ends:focus-within .ends { display: grid; }
.card .ends figure { margin: 0; }
.card .ends figcaption { font-size: 12px; margin-top: 4px; }
"""

# Hover popups: open the end-of-line boards on whichever side of the card has room.
JS = """
document.querySelectorAll('.card.has-ends').forEach(function (c) {
  function place() {
    var e = c.querySelector('.ends'), r = c.getBoundingClientRect(), w = 490;
    e.classList.remove('flip', 'below');
    if (r.right + w <= window.innerWidth) return;
    if (r.left - w >= 0) e.classList.add('flip'); else e.classList.add('below');
  }
  c.addEventListener('mouseenter', place);
  c.addEventListener('focusin', place);
});
"""


def page(title: str, body: str, back: str | None = "report.html") -> str:
    """Detailed per-module page. back: link to the main report (pages live next to it by default)."""
    nav = (f"<p class='backnav'><a href='{html.escape(back)}'>← Main report</a> · "
           f"<a href='../../../index.html'>All players &amp; comparisons</a></p>" if back else "")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><script>{THEME_JS}</script><style>{CSS}</style></head>"
            f"<body><main>{nav}{body}</main><script>{JS}</script></body></html>")


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
    return chess.svg.board(b, orientation=b.turn, arrows=arrows, size=size, coordinates=True, colors=BOARD)


def end_board_svg(fen: str, line, size: int = 220) -> str:
    """Position after playing `line` from `fen`, from the root mover's side, last move highlighted."""
    b = chess.Board(fen)
    pov = b.turn
    last = None
    try:
        for u in line:
            last = chess.Move.from_uci(u)
            b.push(last)
    except Exception:
        pass
    return chess.svg.board(b, orientation=pov, lastmove=last, size=size, coordinates=True, colors=BOARD)


def _mover(fen: str) -> str:
    return "White" if chess.Board(fen).turn == chess.WHITE else "Black"


def position_card(r, *, puzzle: bool = False, link: str | None = None, note: str | None = None) -> str:
    """r: a row with fen, move, best_move, best_line, played_line, wp_loss, phase."""
    head = (f"{_mover(r.fen)} to move · {r.phase} · loss {r.wp_loss:.0%}"
            if not puzzle else f"{_mover(r.fen)} to move · {r.phase} · find the best move")
    if link:
        head += f" · <a href='{html.escape(link)}' target='_blank'>game</a>"
    if note:
        head += f"<br>{html.escape(note)}"
    if puzzle:
        svg = board_svg(r.fen)
        sol = (f"<details><summary>solution</summary><div class='line best'>"
               f"{html.escape(line_san(r.fen, r.best_line))}</div>"
               f"<div class='muted'>{board_svg(r.fen, r.best_move)}</div></details>")
        return f"<div class='card'>{svg}<div class='muted'>{head}</div>{sol}</div>"
    svg = board_svg(r.fen, r.best_move, r.move)
    best, played = list(r.best_line), list(r.played_line)
    ends = (f"<div class='ends'>"
            f"<figure>{end_board_svg(r.fen, best)}<figcaption class='best'>after the best line "
            f"<span class='muted'>({len(best)} plies)</span></figcaption></figure>"
            f"<figure>{end_board_svg(r.fen, played)}<figcaption class='played'>after the played line "
            f"<span class='muted'>({len(played)} plies)</span></figcaption></figure></div>")
    return (f"<div class='card has-ends' tabindex='0'>"
            f"{svg}<div class='muted'>{head}</div>"
            f"<div class='line best'>best: {html.escape(line_san(r.fen, r.best_line))}</div>"
            f"<div class='line played'>played: {html.escape(line_san(r.fen, r.played_line))}</div>"
            f"{ends}</div>")
