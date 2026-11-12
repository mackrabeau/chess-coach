"""Navigation: one page (data/index.html) that links every player's report, every comparison and the
latest log, plus the 'Files' section each report gets (detailed reports + Lichess studies with a
copy-PGN button), so nothing needs the file explorer.

Rebuilt at the end of main.py, 20_report.py and 23_compare_players.py.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from .config import data_root
from .webreport import Report, esc


def rel(target: Path, start: Path) -> str:
    return os.path.relpath(target, start).replace("\\", "/")


def _stamp(p: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))


def files_section(page_dir: Path, htmls: list[Path], pgns: list[Path]) -> str:
    """Links to detailed reports and every study (copy / download / Lichess) as HTML."""
    out = []
    if htmls:
        out.append("<h2>Detailed reports</h2><ul>" + "".join(
            f"<li><a href='{esc(rel(p, page_dir))}'>{esc(p.stem.replace('_', ' '))}</a> "
            f"<span class='muted'>· {_stamp(p)}</span></li>" for p in htmls) + "</ul>")
    if pgns:
        rows = []
        for i, p in enumerate(pgns):
            text = p.read_text(encoding="utf-8")
            n = len(re.findall(r'^\[Event ', text, flags=re.M))
            rows.append(f"<div class='pgnrow'><b>{esc(p.stem.replace('_', ' '))}</b> "
                        f"<span class='muted'>{n} chapters · {_stamp(p)}</span><br>"
                        f"<button class='copy' data-src='pgn{i}'>Copy PGN</button> "
                        f"<a href='{esc(rel(p, page_dir))}' download>download</a> · "
                        f"<a href='https://lichess.org/study' target='_blank' rel='noopener'>open Lichess studies</a>"
                        f"<textarea id='pgn{i}' hidden>{esc(text)}</textarea></div>")
        out.append("<h2>Lichess studies</h2><p class='muted'>Copy PGN, then on Lichess: Studies → create a study "
                   "(or open one) → add chapter → <i>PGN</i> tab → paste. A study holds 64 chapters; bigger themes are "
                   "split into _1, _2, ...</p>" + "".join(rows))
    return "".join(out) or "<p class='muted'>No files yet.</p>"


def build_index(cfg) -> Path:
    root = data_root(cfg)
    rep = Report("Chess coach")
    P = rep.tab("players", "Players")
    players = sorted((root / "players").glob("*/player"))
    if not players:
        P.append("<p class='muted'>No players yet: python main.py --config configs/local/&lt;you&gt;.yaml</p>")
    for d in players:
        main = d / "report.html"
        others = sorted(p for p in d.glob("*.html") if p.name != "report.html")
        link = (f"<a href='{esc(rel(main, root))}'><b>main report</b></a> <span class='muted'>· {_stamp(main)}</span>"
                if main.exists() else "<span class='muted'>no main report yet (run main.py)</span>")
        P.append(Report.finding(d.parent.name, link + "<br>" + " · ".join(
            f"<a href='{esc(rel(p, root))}'>{esc(p.stem.replace('_', ' '))}</a>" for p in others), "info"))
    C = rep.tab("compare", "Comparisons")
    comps = sorted((root / "compare").glob("*/report.html"))
    C += [Report.finding(c.parent.name.replace("_vs_", " vs "),
                         f"<a href='{esc(rel(c, root))}'><b>open</b></a> <span class='muted'>· {_stamp(c)}</span>", "info")
          for c in comps] or ["<p class='muted'>None yet: python scripts/23_compare_players.py --config A --other B</p>"]
    L = rep.tab("logs", "Logs")
    logs = sorted((root / "logs").glob("main_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    L.append("<ul>" + "".join(f"<li><a href='{esc(rel(p, root))}'>{esc(p.name)}</a></li>" for p in logs) + "</ul>"
             if logs else "<p class='muted'>No logs yet.</p>")
    out = root / "index.html"
    rep.write(out)
    return out
