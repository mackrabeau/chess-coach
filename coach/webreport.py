"""Building blocks for the interactive report (20_report.py): one self-contained HTML file.

No external libraries: boards are drawn by a small inline script from python-chess's piece SVGs;
every line is precomputed in Python (FEN after each ply), so the page needs no chess logic beyond
"click two squares" for puzzles.

Readability rules (sources: web.dev typography, GOV.UK type scale / details, NN/g on scanning):
16px body, line-height 1.6, prose capped at ~70 characters; the point of each block first (claim,
then what to do, then numbers, then boards); plain-language titles with the technical category
kept underneath; tags always carry a word, never colour alone. Theme: system / light / dark
toggle (stored per browser); tabs are deep-linkable (#middlegame) and follow the back button.
"""
from __future__ import annotations

import html
import json
import re

import chess
import chess.svg

from .categories import PLAIN

# Tag ids stay the data keys; readers see plain words, with a one-line explanation on hover.
TAG_TEXT = {"solid": "Confirmed weakness", "lead": "Possible weakness", "style": "Your style",
            "strength": "Strength", "info": "Context", "exploratory": "Experimental"}
TAG_HELP = {"solid": "You lose more here than players at your rating, and it holds up on a second check.",
            "lead": "Points the same way, but the evidence is thinner (one measure, or borderline). Worth a look.",
            "style": "You play differently from players at your rating here, without losing points for it.",
            "strength": "You lose less here than players at your rating, and it holds up on a second check.",
            "info": "Background numbers that help you read the rest; nothing to fix.",
            "exploratory": "Built on models of opponents or looser evidence; useful ideas, not proven."}

_SAN = re.compile(r"^(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?[?!]*$")


def esc(x) -> str:
    return html.escape(str(x))


def number_moves(s: str) -> str:
    """'e4 e5 Nc3 Nc6 f4?' -> '1.e4 e5 2.Nc3 Nc6 3.f4?' (only bare SAN sequences from the start position)."""
    toks = s.split()
    if len(toks) < 2 or not all(_SAN.match(t) for t in toks):
        return s
    out = []
    for i, t in enumerate(toks):
        out.append(f"{i // 2 + 1}.{t}" if i % 2 == 0 else t)
    return " ".join(out)


def plain_title(t: str) -> str:
    """Readable title: category ids -> plain words; bare move sequences get move numbers."""
    out = _plain(t)
    return out[:1].upper() + out[1:] if out[:1].islower() and not _SAN.match(out.split()[0]) else out


def _plain(t: str) -> str:
    if t in PLAIN:
        return PLAIN[t]
    parts = t.split(": ")
    for i in range(1, len(parts)):              # "<prefix>: <category or moves>"
        head, rest = ": ".join(parts[:i]), ": ".join(parts[i:])
        if rest in PLAIN:
            return f"{head}: {PLAIN[rest]}"
        numbered = number_moves(rest)
        if numbered != rest:
            return f"{head}: {numbered}"
    return number_moves(t)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "item"


def line_data(fen: str, ucis) -> dict:
    """SAN + FEN after each ply of a line (stops at the first illegal move)."""
    b = chess.Board(fen)
    sans, fens, moves = [], [], []
    for u in list(ucis or []):
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            break
        if mv not in b.legal_moves:
            break
        num = f"{b.fullmove_number}." if b.turn else (f"{b.fullmove_number}..." if not sans else "")
        sans.append(num + b.san(mv))
        b.push(mv)
        fens.append(b.fen())
        moves.append(u)
    return {"sans": sans, "fens": fens, "ucis": moves}


_NUM = re.compile(r"^\s*[~≈±+\-−]?\s*\d")
_TAGS = re.compile(r"<[^>]+>")


class Report:
    """Collects tabs, sections and the example/puzzle data, then writes one HTML file."""

    def __init__(self, title: str, home: str | None = None, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.home = home                         # relative link to data/index.html
        self.tabs: list[tuple[str, str, list[str]]] = []   # (id, label, html parts)
        self.examples: list[dict] = []
        self.game = False                        # a game-entry board is on the page (loads chess.js)

    def tab(self, tid: str, label: str) -> list[str]:
        parts: list[str] = []
        self.tabs.append((tid, label, parts))
        return parts

    # ------------------------------------------------------------ components
    def example(self, fen: str, best=None, played=None, note: str = "", puzzle: bool = False,
                labels=("best line", "your move")) -> str:
        """Interactive board: step through the best line and the played line; puzzle = find the move."""
        b = chess.Board(fen)
        ex = {"fen": fen, "orient": "w" if b.turn else "b", "note": note, "puzzle": puzzle,
              "lines": [{"name": labels[0], **line_data(fen, best)}]}
        if played:
            ex["lines"].append({"name": labels[1], **line_data(fen, played)})
        self.examples.append(ex)
        return f"<div class='ex' data-ex='{len(self.examples) - 1}'></div>"

    def game_board(self, me: str, inbox: str) -> str:
        """Analysis board to enter a game (click moves or paste a PGN), then copy / download the PGN or
        open it on Lichess. me: your name for the White/Black tag (so the import recognises you);
        inbox: the folder the import stage reads (shown to the reader)."""
        self.game = True
        return f"<div class='gameboard' data-me='{esc(me)}' data-inbox='{esc(inbox)}'></div>"

    @staticmethod
    def finding(title: str, claim: str, tag: str, evidence: str = "", examples: str = "",
                todo: str = "", elo=None, elo_note: str = "", link: str | None = None, open_: bool = True,
                sub: str = "", fid: str | None = None) -> str:
        """One finding: headline (tag, title, Elo), then claim -> what to do -> numbers -> boards."""
        gain = (f"<span class='gain' title='{esc(elo_note)}'>≈ +{elo:.0f} Elo</span>" if elo else "")
        more = f" <a class='more' href='{esc(link)}'>Full analysis →</a>" if link else ""
        tagt = TAG_TEXT.get(tag, tag)
        subl = f"<span class='sub'>{esc(sub)}</span>" if sub else ""
        idattr = f" id='{esc(fid)}'" if fid else ""
        todo_html = "" if not todo or todo.strip() == "Nothing to fix." else \
            f"<div class='todo'><span class='todo-k'>What to do</span> {todo}</div>"
        return (f"<details class='finding t-{esc(tag)}'{idattr}{' open' if open_ else ''}><summary>"
                f"<span class='tag t-{esc(tag)}' title='{esc(TAG_HELP.get(tag, ''))}'>{esc(tagt)}</span>"
                f"<span class='ftitle'>{esc(title)}{subl}</span>{gain}</summary><div class='fbody'>"
                f"<p class='claim'>{claim}{more}</p>{todo_html}"
                + (f"<div class='evidence'>{evidence}</div>" if evidence else "")
                + (f"<div class='exs'>{examples}</div>" if examples else "") + "</div></details>")

    @staticmethod
    def table(header, rows, cls: str = "") -> str:
        def td(c):
            txt = _TAGS.sub("", str(c)).strip()
            k = []
            if _NUM.match(txt):
                k.append("num")
            if txt in ("worse", "weaker", "lower"):
                k.append("bad")
            elif txt in ("better", "stronger"):
                k.append("good")
            return f"<td class='{' '.join(k)}'>{c}</td>" if k else f"<td>{c}</td>"
        body = "".join("<tr>" + "".join(td(c) for c in r) + "</tr>" for r in rows)
        numcol = [bool(rows) and all(_NUM.match(_TAGS.sub("", str(r[i])).strip() or "0")
                                     for r in rows if i < len(r)) for i in range(len(header))]
        h = "".join(f"<th class='num'>{esc(c)}</th>" if numcol[i] else f"<th>{esc(c)}</th>"
                    for i, c in enumerate(header))
        return f"<div class='tw'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>"

    @staticmethod
    def kpis(items) -> str:
        """items: (label, value, note)."""
        return "<div class='kpis'>" + "".join(
            f"<div class='kpi'><span class='kl'>{esc(a)}</span><b>{esc(b)}</b><span class='kn'>{esc(c)}</span></div>"
            for a, b, c in items) + "</div>"

    @staticmethod
    def level_chip(rel: float | None, text: str | None = None, scale: float = 300) -> str:
        """Red (below your level) -> green (above), for a relative Elo-like value."""
        if rel is None or rel != rel:
            return "<span class='chip'>–</span>"
        t = max(-1.0, min(1.0, rel / scale))
        hue = 60 + 60 * t                     # 0 = red, 60 = yellow, 120 = green
        return (f"<span class='chip' style='background:hsl({hue:.0f} 60% 38%)'>"
                f"{esc(text if text is not None else f'{rel:+.0f}')}</span>")

    # ------------------------------------------------------------ output
    def html(self) -> str:
        pieces = "".join(f"<symbol id='p-{s}' viewBox='0 0 45 45'>{svg}</symbol>"
                         for s, svg in chess.svg.PIECES.items())
        nav = "".join(f"<a role='tab' href='#{tid}' data-tab='{tid}'>{esc(label)}</a>" for tid, label, _ in self.tabs)
        panes = "".join(f"<section id='{tid}' class='pane' role='tabpanel'>{''.join(parts)}</section>"
                        for tid, _, parts in self.tabs)
        data = json.dumps(self.examples).replace("</", "<\\/")
        home = f"<a class='home' href='{esc(self.home)}'>← All players</a>" if self.home else ""
        sub = f"<p class='subtitle'>{esc(self.subtitle)}</p>" if self.subtitle else ""
        return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
                f"<title>{esc(self.title)}</title><script>{THEME_JS}</script><style>{BASE_CSS}{CSS}</style></head><body>"
                f"<svg width='0' height='0' style='position:absolute' aria-hidden='true'><defs>{pieces}</defs></svg>"
                f"<header class='top'><div class='bar'><div><h1>{esc(self.title)}</h1>{sub}</div>"
                f"<div class='tools'>{home}<button class='theme' type='button'>Theme: auto</button></div></div>"
                f"<nav role='tablist'>{nav}</nav></header><main>{panes}</main>"
                + (f"<script src='{CHESS_JS}'></script>" if self.game else "")
                + f"<script>const EX={data};{JS}{GAME_JS if self.game else ''}</script></body></html>")

    def write(self, path) -> None:
        path.write_text(self.html(), encoding="utf-8")


# Runs in <head> so the stored theme applies before first paint (no flash). Shared with coach/report.py.
THEME_JS = "try{const t=localStorage.getItem('theme');if(t)document.documentElement.dataset.theme=t}catch(e){}"

# Tokens + typography shared by every page (main report, hub, comparisons, detailed pages).
BASE_CSS = """
:root{--bg:#f6f5f1;--surface:#fff;--surface2:#f0eee8;--fg:#1b1b1d;--muted:#55555c;--line:#dedbd3;
--accent:#1f5fc4;--link:#1a5bc0;--link-visited:#6a3cab;--good:#0f7a5a;--bad:#b8322a;--warn:#a35f00;
--good-bg:#e5f3ee;--bad-bg:#fbeceb;--warn-bg:#fdf3e2;--info-bg:#e9f0fb;--focus:#1f5fc4;
--shadow:0 1px 2px rgba(0,0,0,.05),0 2px 8px rgba(0,0,0,.04);color-scheme:light}
:root[data-theme="dark"]{--bg:#151619;--surface:#1e1f23;--surface2:#26272c;--fg:#e9e9ec;--muted:#a3a3ad;
--line:#34353c;--accent:#7eaaff;--link:#8db4ff;--link-visited:#c4a8ff;--good:#4cc79a;--bad:#ff7d72;--warn:#f0b04d;
--good-bg:#173229;--bad-bg:#3a1f1d;--warn-bg:#3a2c15;--info-bg:#1b2840;--focus:#8db4ff;--shadow:none;color-scheme:dark}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151619;--surface:#1e1f23;--surface2:#26272c;
--fg:#e9e9ec;--muted:#a3a3ad;--line:#34353c;--accent:#7eaaff;--link:#8db4ff;--link-visited:#c4a8ff;--good:#4cc79a;
--bad:#ff7d72;--warn:#f0b04d;--good-bg:#173229;--bad-bg:#3a1f1d;--warn-bg:#3a2c15;--info-bg:#1b2840;--focus:#8db4ff;
--shadow:none;color-scheme:dark}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,
"Helvetica Neue",Arial,sans-serif;font-variant-numeric:tabular-nums}
a{color:var(--link);text-underline-offset:2px}a:visited{color:var(--link-visited)}
a:hover{text-decoration-thickness:2px}
:focus-visible{outline:3px solid var(--focus);outline-offset:2px;border-radius:4px}
h1{font-size:1.5rem;line-height:1.25;margin:0;letter-spacing:-.01em}
h2{font-size:1.25rem;line-height:1.3;margin:2.2rem 0 .6rem}
h3{font-size:1.06rem;line-height:1.35;margin:1.4rem 0 .4rem}
p,li{max-width:72ch;text-wrap:pretty}
code{font:.9em ui-monospace,Consolas,monospace;background:var(--surface2);padding:.1em .35em;border-radius:4px}
.muted{color:var(--muted)}
.lede{font-size:1.05rem;color:var(--muted);margin:.2rem 0 1rem}
.tw{overflow-x:auto;margin:.6rem 0;max-width:100%}
table{border-collapse:collapse;font-size:.94rem}
th,td{padding:.4rem .75rem;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:.82rem;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right}td.bad{color:var(--bad);font-weight:600}td.good{color:var(--good);font-weight:600}
tbody tr:hover{background:var(--surface2)}
"""

CSS = """
header.top{position:sticky;top:0;z-index:10;background:var(--bg);border-bottom:1px solid var(--line)}
.bar{max-width:1120px;margin:0 auto;padding:.9rem 1.25rem .3rem;display:flex;gap:1rem;align-items:flex-start;
justify-content:space-between;flex-wrap:wrap}
.subtitle{margin:.15rem 0 0;color:var(--muted);font-size:.92rem}
.tools{display:flex;gap:.9rem;align-items:center;font-size:.92rem}
.theme{font:inherit;font-size:.85rem;color:var(--muted);background:none;border:1px solid var(--line);
border-radius:6px;padding:.15rem .6rem;cursor:pointer}
nav{max-width:1120px;margin:0 auto;padding:0 1.25rem;display:flex;gap:.25rem;overflow-x:auto;scrollbar-width:none}
nav::-webkit-scrollbar{display:none}
nav a{flex:none;padding:.55rem .8rem;color:var(--muted);text-decoration:none;font-weight:550;
border-bottom:3px solid transparent;white-space:nowrap}
nav a:visited{color:var(--muted)}nav a:hover{color:var(--fg)}
nav a[aria-selected="true"]{color:var(--fg);border-bottom-color:var(--accent)}
main{max-width:1120px;margin:0 auto;padding:1.5rem 1.25rem 5rem}
.pane{display:none}.pane.on{display:block}.pane>h2:first-child{margin-top:.4rem}
/* tags: a word plus a colour, never colour alone */
.tag{display:inline-block;flex:none;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
border-radius:4px;padding:.12rem .45rem;line-height:1.4;background:var(--surface2);color:var(--muted)}
.tag.t-solid{background:var(--bad-bg);color:var(--bad)}.tag.t-lead{background:var(--warn-bg);color:var(--warn)}
.tag.t-strength{background:var(--good-bg);color:var(--good)}.tag.t-info{background:var(--info-bg);color:var(--accent)}
.tag.t-exploratory{background:#efe7f8;color:#6a3a9e}
:root[data-theme="dark"] .tag.t-exploratory{background:#2e2340;color:#c9a8f5}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .tag.t-exploratory{background:#2e2340;color:#c9a8f5}}
/* findings */
.finding{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;
margin:.75rem 0;box-shadow:var(--shadow)}
.finding.t-solid{border-left-color:var(--bad)}.finding.t-lead{border-left-color:var(--warn)}
.finding.t-strength{border-left-color:var(--good)}.finding.t-info{border-left-color:var(--accent)}
.finding.t-exploratory{border-left-color:#8a5cc2}
.finding>summary{list-style:none;cursor:pointer;display:flex;gap:.7rem;align-items:baseline;padding:.85rem 1.1rem}
.finding>summary::-webkit-details-marker{display:none}
.finding>summary::after{content:"";flex:none;align-self:center;width:.55rem;height:.55rem;margin-left:.2rem;
border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(45deg) translateY(-2px);
transition:transform .15s}
.finding[open]>summary::after{transform:rotate(-135deg)}
.finding>summary:hover .ftitle{text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px}
.ftitle{flex:1;font-weight:650;font-size:1.04rem;line-height:1.4;min-width:0}
.sub{display:block;font-weight:400;font-size:.8rem;color:var(--muted);font-family:ui-monospace,Consolas,monospace;
margin-top:.1rem}
.gain{flex:none;font-weight:700;color:var(--accent);white-space:nowrap}
.fbody{padding:0 1.1rem 1rem}
.claim{margin:0 0 .7rem}.more{white-space:nowrap;font-weight:550}
.todo{max-width:72ch;background:var(--info-bg);border-radius:6px;padding:.55rem .8rem;margin:.6rem 0}
.todo-k{display:block;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:var(--accent)}
.evidence{margin:.4rem 0}
/* summary */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.8rem;margin:.4rem 0 1rem}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem;box-shadow:var(--shadow)}
.kpi .kl{display:block;font-size:.85rem;color:var(--muted);font-weight:550}
.kpi b{display:block;font-size:1.9rem;line-height:1.2;margin:.15rem 0;letter-spacing:-.01em}
.kpi .kn{display:block;font-size:.8rem;color:var(--muted);line-height:1.4}
.rank{list-style:none;padding:0;margin:.5rem 0;counter-reset:r;max-width:54rem}
.rank>li{max-width:none;counter-increment:r;display:grid;grid-template-columns:2.2rem 1fr;gap:.2rem .6rem;
background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:.85rem 1.1rem .9rem .8rem;
margin:.6rem 0;box-shadow:var(--shadow)}
.rank>li::before{content:counter(r);grid-row:1/span 4;font-size:1.3rem;font-weight:700;color:var(--muted);
text-align:center;line-height:1.5}
.rank .rh{display:flex;gap:.6rem;align-items:baseline;flex-wrap:wrap}
.rank .rt{font-weight:650;font-size:1.06rem;flex:1;min-width:12rem}
.rank .phase{font-size:.8rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.gbar{height:6px;border-radius:3px;background:var(--surface2);overflow:hidden;max-width:32rem;margin:.25rem 0 .2rem;
display:block}
.gbar i{display:block;height:100%;background:var(--accent);border-radius:3px}
.rank p{margin:.25rem 0}.rank .todo{margin:.45rem 0 .1rem}
.okl{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.5rem;list-style:none;padding:0}
.okl li{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--good);border-radius:6px;
padding:.5rem .8rem;font-weight:550}
details.plain{margin:1rem 0}details.plain>summary{cursor:pointer;color:var(--muted);font-weight:550}
.steps{margin:.3rem 0 0;padding-left:1.3rem}.steps li{margin:.15rem 0}
.legend{list-style:none;padding:0}.legend li{margin:.5rem 0;display:grid;grid-template-columns:12rem 1fr;gap:.8rem;
align-items:baseline;max-width:56rem}.legend .tag{text-align:center}
/* boards */
.exs{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:1.1rem;margin-top:1rem}
.ex{min-width:0}
.ex .eh{display:flex;align-items:center;gap:.45rem;font-size:.88rem;font-weight:600;margin-bottom:.3rem}
.turn{width:.85rem;height:.85rem;border-radius:50%;border:1.5px solid #777;flex:none}
.turn.w{background:#fff}.turn.b{background:#222}
.bw{border-radius:4px;overflow:hidden;line-height:0;outline-offset:3px}
.ex svg.board{width:100%;height:auto;display:block;user-select:none}
.ex .ctl{display:flex;gap:.3rem;align-items:center;margin:.45rem 0 .2rem}
.ex .ctl button{font:inherit;font-size:.95rem;line-height:1;border:1px solid var(--line);background:var(--surface);
color:var(--fg);border-radius:6px;padding:.35rem .6rem;cursor:pointer;min-width:2.2rem}
.ex .ctl button:hover{background:var(--surface2)}
.ex .ctl .pos{margin-left:auto;font-size:.8rem;color:var(--muted)}
.ex .ln{display:flex;flex-wrap:wrap;gap:.1rem .15rem;align-items:baseline;font-size:.88rem;line-height:1.5;
margin:.3rem 0;padding:.25rem .4rem;border-radius:6px}
.ex .ln.on{background:var(--surface2)}
.ex .lk{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-right:.3rem;
display:inline-flex;align-items:center;gap:.3rem;cursor:pointer}
.ex .lk::before{content:"";width:.6rem;height:.6rem;border-radius:2px;background:var(--good)}
.ex .ln.l1 .lk::before{background:var(--bad)}
.ex .ln.l0 .lk{color:var(--good)}.ex .ln.l1 .lk{color:var(--bad)}
.ex .mv{font:inherit;border:0;background:none;color:var(--fg);padding:0 .2rem;border-radius:3px;cursor:pointer}
.ex .mv:hover{background:var(--line)}
.ex .mv.cur{background:var(--accent);color:#fff}
.ex .mv.fut{color:var(--muted)}
.ex .note{font-size:.88rem;color:var(--muted);line-height:1.5;margin-top:.35rem}
.ex .note i{font-style:normal;color:var(--fg)}
.ex .res{font-weight:650;font-size:.92rem;min-height:1.4em}
.ex .ask{font-size:.92rem;font-weight:600;margin:.4rem 0 .2rem}
.co{font-size:9.5px;font-weight:700;pointer-events:none;font-family:system-ui,sans-serif}
.ok{color:var(--good)}.no{color:var(--bad)}
.chip{display:inline-block;min-width:3rem;text-align:center;color:#fff;border-radius:4px;padding:0 .4rem;
background:#777;font-weight:600}
/* files */
.pgnrow{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:.7rem 1rem;margin:.6rem 0}
.copy{font:inherit;font-size:.88rem;border:1px solid var(--line);background:var(--surface2);color:var(--fg);
border-radius:6px;padding:.2rem .7rem;cursor:pointer;margin-right:.4rem}
.home{font-weight:550;text-decoration:none}
/* game entry board */
.gb{display:grid;grid-template-columns:minmax(260px,440px) minmax(260px,1fr);gap:1.4rem;align-items:start;
background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:1rem;box-shadow:var(--shadow)}
.gb .ctl{display:flex;gap:.3rem;flex-wrap:wrap;margin:.5rem 0}
.gb button{font:inherit;font-size:.9rem;border:1px solid var(--line);background:var(--surface2);color:var(--fg);
border-radius:6px;padding:.3rem .65rem;cursor:pointer}.gb button:hover{border-color:var(--muted)}
.gb button.pri{background:var(--accent);border-color:var(--accent);color:#fff}
.gml{display:grid;grid-template-columns:3ch 1fr 1fr;gap:.1rem .3rem;max-height:16rem;overflow-y:auto;
font-size:.92rem;border:1px solid var(--line);border-radius:6px;padding:.4rem;min-height:3rem;margin-bottom:.8rem}
.gml .n{color:var(--muted);text-align:right}.gml button{border:0;background:none;padding:.05rem .3rem;text-align:left}
.gml button.cur{background:var(--accent);color:#fff}
.gh{display:grid;grid-template-columns:auto 1fr;gap:.35rem .6rem;align-items:center;font-size:.9rem;margin-bottom:.8rem}
.gh input,.gh select,.gpaste{font:inherit;font-size:.9rem;background:var(--bg);color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:.25rem .45rem;width:100%}
.gbtn{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.6rem}
.gmsg{min-height:1.4em;font-size:.88rem}
@media (max-width:760px){.gb{grid-template-columns:1fr}}
@media (max-width:640px){body{font-size:16px}h1{font-size:1.25rem}.bar{padding:.7rem 1rem .2rem}nav{padding:0 .6rem}
main{padding:1rem 1rem 4rem}.finding>summary{padding:.75rem .85rem;flex-wrap:wrap}.ftitle{flex-basis:70%}
.fbody{padding:0 .85rem .9rem}.kpi b{font-size:1.6rem}.rank>li{grid-template-columns:1.6rem 1fr;padding:.75rem .8rem}}
@media print{header.top{position:static}nav,.theme,.ctl{display:none}.pane{display:block!important}
.finding{break-inside:avoid}}
"""

JS = r"""
const SQ=45,FILES='abcdefgh',NS='http://www.w3.org/2000/svg';
const COL={light:'#f0d9b5',dark:'#b58863',last:'rgba(155,199,0,.45)',pick:'rgba(20,85,180,.5)',
 good:'rgba(21,120,27,.85)',bad:'rgba(200,40,30,.8)'};
function sqXY(sq,o){const f=FILES.indexOf(sq[0]),r=+sq[1]-1;return o==='w'?[f*SQ,(7-r)*SQ]:[(7-f)*SQ,r*SQ];}
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function arrow(s,from,to,o,color){const [x1,y1]=sqXY(from,o),[x2,y2]=sqXY(to,o);
 const ax=x1+SQ/2,ay=y1+SQ/2,bx=x2+SQ/2,by=y2+SQ/2,dx=bx-ax,dy=by-ay,L=Math.hypot(dx,dy);if(!L)return;
 const ux=dx/L,uy=dy/L,px=-uy,py=ux,hl=15,hw=11,sw=4.6,ex=bx-ux*hl,ey=by-uy*hl,sx=ax+ux*8,sy=ay+uy*8;
 const pts=[[sx+px*sw,sy+py*sw],[ex+px*sw,ey+py*sw],[ex+px*hw,ey+py*hw],[bx,by],[ex-px*hw,ey-py*hw],
  [ex-px*sw,ey-py*sw],[sx-px*sw,sy-py*sw]].map(p=>p.join(',')).join(' ');
 s.appendChild(el('polygon',{points:pts,fill:color,'pointer-events':'none'}));}
function pieceAt(fen,sq){const rows=fen.split(' ')[0].split('/'),r=8-+sq[1],tf=FILES.indexOf(sq[0]);let f=0;
 for(const c of rows[r]){if(/\d/.test(c)){f+=+c;continue;}if(f===tf)return c;f++;}return null;}
function boardSVG(fen,o,marks,click){
 const s=el('svg',{viewBox:'0 0 360 360',class:'board',role:'img'});
 for(let r=0;r<8;r++)for(let f=0;f<8;f++){const sq=FILES[f]+(r+1),[x,y]=sqXY(sq,o),light=(f+r)%2===1;
  const e=el('rect',{x,y,width:SQ,height:SQ,fill:light?COL.light:COL.dark});e.dataset.sq=sq;
  if(click){e.style.cursor='pointer';e.addEventListener('click',()=>click(sq));}s.appendChild(e);
  const m=marks.sq[sq];if(m)s.appendChild(el('rect',{x,y,width:SQ,height:SQ,fill:m,'pointer-events':'none'}));}
 fen.split(' ')[0].split('/').forEach((row,i)=>{let f=0;for(const c of row){if(/\d/.test(c)){f+=+c;continue;}
  const [x,y]=sqXY(FILES[f]+(8-i),o),u=el('use',{href:'#p-'+c,x,y,width:SQ,height:SQ});
  u.style.pointerEvents='none';s.appendChild(u);f++;}});
 for(let i=0;i<8;i++){const fl=o==='w'?FILES[i]:FILES[7-i],rk=o==='w'?8-i:i+1;
  const bottomLight=i%2===1, leftLight=((o==='w'?0:7)+(rk-1))%2===1;
  const a=el('text',{x:i*SQ+SQ-2.5,y:357,'text-anchor':'end',class:'co',fill:bottomLight?COL.dark:COL.light});
  a.textContent=fl;s.appendChild(a);
  const b=el('text',{x:2.5,y:i*SQ+10.5,class:'co',fill:leftLight?COL.dark:COL.light});b.textContent=rk;s.appendChild(b);}
 for(const [u,c] of marks.arrows)arrow(s,u.slice(0,2),u.slice(2,4),o,c);
 return s;}
function mount(root){const ex=EX[+root.dataset.ex];let li=0,ply=0,pick=null,solved=!ex.puzzle;
 const side=ex.fen.split(' ')[1]==='w'?'White':'Black';
 const head=document.createElement('div'),bw=document.createElement('div'),ctl=document.createElement('div'),
  lines=document.createElement('div'),res=document.createElement('div'),note=document.createElement('div');
 head.className='eh';head.innerHTML='<span class="turn '+ex.fen.split(' ')[1]+'"></span>'+side+' to move';
 bw.className='bw';bw.tabIndex=0;bw.setAttribute('aria-label','board, use arrow keys to step through the line');
 ctl.className='ctl';res.className='res';res.setAttribute('aria-live','polite');note.className='note';
 const L=()=>ex.lines[li];
 function marks(){const m={sq:{},arrows:[]};
  if(!solved){if(pick)m.sq[pick]=COL.pick;return m;}
  if(ply===0){ex.lines.slice().reverse().forEach((l,j)=>{const i=ex.lines.length-1-j;
    if(l.ucis[0]&&!(i>0&&l.ucis[0]===ex.lines[0].ucis[0]))m.arrows.push([l.ucis[0],i===0?COL.good:COL.bad]);});return m;}
  const u=L().ucis[ply-1];m.sq[u.slice(0,2)]=COL.last;m.sq[u.slice(2,4)]=COL.last;return m;}
 function draw(){const fen=ply===0?ex.fen:L().fens[ply-1];
  bw.replaceChildren(boardSVG(fen,ex.orient,marks(),solved?null:onClick));
  if(!solved){lines.innerHTML='<div class="ask">Find the best move for '+side+'.</div>';ctl.hidden=true;return;}
  ctl.hidden=false;lines.replaceChildren(...ex.lines.map((l,i)=>{const d=document.createElement('div');
   d.className='ln l'+Math.min(i,1)+(i===li?' on':'');const k=document.createElement('span');k.className='lk';
   k.textContent=l.name;k.onclick=()=>{li=i;ply=0;draw();};d.appendChild(k);
   l.sans.forEach((sn,j)=>{const b=document.createElement('button');b.className='mv'+(i===li?(j===ply-1?' cur':(j>=ply?' fut':'')):'');
    b.textContent=sn;b.onclick=()=>{li=i;ply=j+1;draw();};d.appendChild(b);});
   if(!l.sans.length)d.insertAdjacentHTML('beforeend','<span class="muted">–</span>');return d;}));
  pos.textContent=ply?('move '+ply+' / '+L().fens.length):'start';}
 function onClick(sq){const fen=ex.fen,p=pieceAt(fen,sq),mine=p&&((p===p.toUpperCase())===(side==='White'));
  if(!pick||mine){if(mine){pick=sq;res.textContent='';}draw();return;}
  const u=pick+sq,want=ex.lines[0].ucis[0]||'';pick=null;
  if(u===want.slice(0,4)){res.textContent='✓ Correct';res.className='res ok';solved=true;li=0;ply=1;}
  else{res.textContent='✗ Not this one, try again';res.className='res no';}draw();}
 const mk=(t,lab,f)=>{const b=document.createElement('button');b.type='button';b.textContent=t;
  b.setAttribute('aria-label',lab);b.title=lab;b.onclick=()=>{f();draw();};ctl.appendChild(b);};
 mk('⏮','start',()=>{ply=0;});mk('◀','previous move',()=>{ply=Math.max(0,ply-1);});
 mk('▶','next move',()=>{ply=Math.min(L().fens.length,ply+1);});mk('⏭','end of line',()=>{ply=L().fens.length;});
 const pos=document.createElement('span');pos.className='pos';ctl.appendChild(pos);
 bw.addEventListener('keydown',e=>{if(!solved)return;const n=L().fens.length;
  const k={ArrowLeft:()=>ply=Math.max(0,ply-1),ArrowRight:()=>ply=Math.min(n,ply+1),Home:()=>ply=0,End:()=>ply=n,
   ArrowUp:()=>{li=(li+ex.lines.length-1)%ex.lines.length;ply=0;},ArrowDown:()=>{li=(li+1)%ex.lines.length;ply=0;}}[e.key];
  if(k){e.preventDefault();k();draw();}});
 if(ex.puzzle){const sb=document.createElement('button');sb.type='button';sb.className='copy';sb.textContent='Show solution';
  sb.onclick=()=>{solved=true;pick=null;li=0;ply=0;res.textContent='';sb.remove();draw();};
  root.append(head,bw,ctl,res,lines,sb,note);}else root.append(head,bw,ctl,res,lines,note);
 note.innerHTML=ex.note||'';if(!ex.note)note.hidden=true;draw();}
document.querySelectorAll('.ex').forEach(mount);
// tabs: deep links (#tab or #tab/finding-id), back button, last tab remembered
const tabs=[...document.querySelectorAll('nav a[data-tab]')],panes=document.querySelectorAll('.pane');
function show(id,target){if(!document.getElementById(id))return;
 tabs.forEach(t=>t.setAttribute('aria-selected',t.dataset.tab===id?'true':'false'));
 panes.forEach(p=>p.classList.toggle('on',p.id===id));try{localStorage.setItem('tab:'+location.pathname,id)}catch(e){}
 const t=target&&document.getElementById(target);
 if(t){if(t.tagName==='DETAILS')t.open=true;t.scrollIntoView({block:'start'});window.scrollBy(0,-110);}
 else window.scrollTo(0,0);}
function fromHash(){const [id,target]=decodeURIComponent(location.hash.slice(1)).split('/');
 if(id&&document.getElementById(id)&&document.getElementById(id).classList.contains('pane')){show(id,target);return true;}return false;}
window.addEventListener('hashchange',fromHash);window.addEventListener('popstate',fromHash);
document.addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]');if(!a||e.ctrlKey||e.metaKey)return;
 const [id,target]=decodeURIComponent(a.getAttribute('href').slice(1)).split('/'),p=document.getElementById(id);
 if(!p||!p.classList.contains('pane'))return;e.preventDefault();history.pushState(null,'','#'+a.getAttribute('href').slice(1));
 show(id,target);});
if(!fromHash()){let s=null;try{s=localStorage.getItem('tab:'+location.pathname)}catch(e){}
 show(s&&document.getElementById(s)?s:tabs[0].dataset.tab);}
// theme toggle: auto -> light -> dark
const tb=document.querySelector('button.theme'),order=['auto','light','dark'];
function setTheme(t){if(t==='auto')delete document.documentElement.dataset.theme;else document.documentElement.dataset.theme=t;
 try{t==='auto'?localStorage.removeItem('theme'):localStorage.setItem('theme',t)}catch(e){}
 if(tb)tb.textContent='Theme: '+t;}
if(tb){setTheme(document.documentElement.dataset.theme||'auto');
 tb.onclick=()=>setTheme(order[(order.indexOf(document.documentElement.dataset.theme||'auto')+1)%3]);}
document.querySelectorAll('button.copy[data-src]').forEach(b=>b.onclick=async()=>{
 const t=document.getElementById(b.dataset.src).value;
 try{await navigator.clipboard.writeText(t);}catch(e){const a=document.createElement('textarea');a.value=t;
  document.body.appendChild(a);a.select();document.execCommand('copy');a.remove();}
 b.textContent='Copied ✓';setTimeout(()=>b.textContent='Copy PGN',1500);});
"""

# Legal moves / PGN parsing for the game-entry board (exact version, loaded only on pages that have one).
CHESS_JS = "https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"

GAME_JS = r"""
document.querySelectorAll('.gameboard').forEach(root=>{
 if(typeof Chess==='undefined'){root.innerHTML='<p class="no">The board needs an internet connection (it loads '+
  'chess.js). Without it, put a .pgn file in <code>'+root.dataset.inbox+'</code> and run main.py.</p>';return;}
 const me=root.dataset.me,today=new Date(Date.now()-new Date().getTimezoneOffset()*6e4).toISOString().slice(0,10);
 root.innerHTML='<div class="gb"><div><div class="bw" tabindex="0" aria-label="board: click a piece, then a square; '+
  'arrow keys step through the game"></div><div class="ctl">'+
  '<button data-a="start" title="start">⏮</button><button data-a="prev" title="previous move">◀</button>'+
  '<button data-a="next" title="next move">▶</button><button data-a="end" title="last move">⏭</button>'+
  '<button data-a="undo" title="take back the last move">Take back</button><button data-a="flip">Flip</button>'+
  '<button data-a="new">New game</button></div></div><div><div class="gml" aria-label="moves"></div>'+
  '<div class="gh"><label>White</label><input data-h="White"><label>Black</label><input data-h="Black">'+
  '<label>Result</label><select data-h="Result"><option>*</option><option>1-0</option><option>0-1</option>'+
  '<option>1/2-1/2</option></select><label>Date</label><input data-h="Date" type="date">'+
  '<label>Event</label><input data-h="Event" placeholder="e.g. club night, 90+30"></div>'+
  '<div class="gbtn"><button class="pri" data-a="dl">Download PGN</button><button data-a="copy">Copy PGN</button>'+
  '<button data-a="lichess">Analyse on Lichess</button></div>'+
  '<details><summary>Paste a PGN instead</summary><textarea class="gpaste" rows="6"></textarea>'+
  '<button data-a="load">Load</button></details><p class="gmsg muted" aria-live="polite"></p></div></div>';
 const bw=root.querySelector('.bw'),ml=root.querySelector('.gml'),msg=root.querySelector('.gmsg');
 const H=k=>root.querySelector('[data-h="'+k+'"]');
 H('White').value=me;H('Date').value=today;
 let hist=[],ply=0,orient='w',pick=null,g=new Chess();
 const at=n=>{const c=new Chess();for(let i=0;i<n;i++)c.move(hist[i].san);return c;};
 const say=t=>{msg.textContent=t;};
 function draw(){g=at(ply);const m={sq:{},arrows:[]};
  if(ply){const u=hist[ply-1];m.sq[u.from]=COL.last;m.sq[u.to]=COL.last;}if(pick)m.sq[pick]=COL.pick;
  bw.replaceChildren(boardSVG(g.fen(),orient,m,click));
  ml.replaceChildren();hist.forEach((mv,i)=>{if(i%2===0){const n=document.createElement('span');n.className='n';
    n.textContent=(i/2+1)+'.';ml.appendChild(n);}
   const b=document.createElement('button');b.textContent=mv.san;b.className=i===ply-1?'cur':'';
   b.onclick=()=>{ply=i+1;pick=null;draw();};ml.appendChild(b);});
  if(!hist.length)ml.innerHTML='<span></span><span class="muted" style="grid-column:span 2">Click a piece, then a square.</span>';
  const cur=ml.querySelector('.cur');if(cur)cur.scrollIntoView({block:'nearest'});}
 function click(sq){const p=g.get(sq);
  if(p&&p.color===g.turn()){pick=sq;draw();return;}
  if(!pick)return;const mv=g.move({from:pick,to:sq,promotion:'q'});pick=null;
  if(mv){hist=hist.slice(0,ply);hist.push(mv);ply++;}draw();}
 function pgn(){const h=k=>(H(k).value||'?').replace(/"/g,"'"),res=H('Result').value;
  let s='[Event "'+h('Event')+'"]\n[Date "'+h('Date').replace(/-/g,'.')+'"]\n[White "'+h('White')+'"]\n[Black "'+
   h('Black')+'"]\n[Result "'+res+'"]\n\n',t='';
  hist.forEach((mv,i)=>{t+=(i%2===0?(i/2+1)+'. ':'')+mv.san+' ';});return s+t+res+'\n';}
 async function copy(t){try{await navigator.clipboard.writeText(t);}catch(e){const a=document.createElement('textarea');
  a.value=t;document.body.appendChild(a);a.select();document.execCommand('copy');a.remove();}}
 const act={start:()=>{ply=0;},prev:()=>{ply=Math.max(0,ply-1);},next:()=>{ply=Math.min(hist.length,ply+1);},
  end:()=>{ply=hist.length;},undo:()=>{hist=hist.slice(0,Math.max(0,ply-1));ply=hist.length;},
  flip:()=>{orient=orient==='w'?'b':'w';},new:()=>{if(hist.length&&!confirm('Start a new game? The moves are cleared.'))return;
   hist=[];ply=0;say('');},
  dl:()=>{if(!hist.length){say('Play or paste some moves first.');return;}
   const name=(H('Date').value+'_'+H('White').value+'_vs_'+H('Black').value).replace(/[^\w.-]+/g,'_')+'.pgn';
   const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([pgn()],{type:'application/x-chess-pgn'}));
   a.download=name;a.click();URL.revokeObjectURL(a.href);
   say('Saved '+name+'. Move it into '+root.dataset.inbox+' and run main.py: it gets analysed with your other games.');},
  copy:async()=>{await copy(pgn());say('PGN copied.');},
  lichess:async()=>{await copy(pgn());window.open('https://lichess.org/paste','_blank','noopener');
   say('PGN copied: paste it on the Lichess page that opened, then press Import (free engine analysis).');},
  load:()=>{const c=new Chess(),t=root.querySelector('.gpaste').value.trim();
   if(!t||!c.load_pgn(t)){say('Could not read that PGN (check the moves).');return;}
   hist=c.history({verbose:true});ply=hist.length;const hd=c.header();
   for(const k of ['White','Black','Result','Event'])if(hd[k])H(k).value=hd[k];
   if(hd.Date&&/^\d{4}\.\d\d\.\d\d$/.test(hd.Date))H('Date').value=hd.Date.replace(/\./g,'-');
   say('Loaded '+hist.length+' moves.');}};
 root.querySelectorAll('button[data-a]').forEach(b=>b.onclick=async()=>{await act[b.dataset.a]();pick=null;draw();});
 bw.addEventListener('keydown',e=>{const k={ArrowLeft:'prev',ArrowRight:'next',Home:'start',End:'end'}[e.key];
  if(k){e.preventDefault();act[k]();draw();}});
 draw();});
"""
