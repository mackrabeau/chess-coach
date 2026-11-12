"""The main report: where your rating can grow, by phase, ordered by expected Elo gain.

Reads every module's findings (coach/findings.py, data/findings/*.json) and lays them out:
  Summary     your level (multi-band), the biggest gains across phases, what's working
  Opening · Middlegame · Endgame
              weaknesses (solid / lead) ordered by expected Elo gain; what's working collapsed below
  Exploitation (exploratory) · Games (imported PGNs) · Drills · Method
High level on purpose: each finding links to the detailed per-module report for the full analysis.
Drills: every weakness with puzzles (drill spec) becomes a Lichess study per theme
(data/player/drills_<theme>.pgn), puzzles carry a 'why' from the feature deltas when available.
Add an analysis by writing findings in its module; this script needs no change.

    python scripts/20_report.py --config configs/v0.yaml        -> data/player/report.html
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import pandas as pd

from coach.config import data_dir, data_root, load_config
from coach.hub import build_index, files_section, rel
from coach.findings import ELO_PER_SCORE, load_all
from coach.results import load_analyzed
from coach.study import puzzle_chapter, write_studies
from coach.webreport import Report, esc, plain_title, slugify
from coach.why import why_texts

WEAK = ("solid", "lead")
SECTIONS = [("opening", "Opening"), ("middlegame", "Middlegame"), ("endgame", "Endgame")]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "drills"


def san(fen, uci):
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def gain_key(f):
    """Order by `rank` (Elo value x trainability) when a module sets it, else by Elo."""
    r = f.get("rank")
    return -(r if r is not None else (f.get("elo") or 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--max-puzzles", type=int, default=128, help="per theme (split into 64-chapter studies)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = data_dir(cfg, "player")
    F = load_all(cfg)
    if not F and not any(data_dir(cfg, "findings").glob("*.json")):
        sys.exit("no findings yet: run the analysis stages first (python main.py)")
    if not F:
        print("note: the analysis stages ran but found nothing clear; writing a report that says so")
    me = load_analyzed(cfg, "player")
    pos = me.set_index("pos_id")
    pids = {e.get("pos_id") for f in F for e in f["examples"]} | \
           {p for f in F if f.get("drill") for p in f["drill"].get("pos_ids", [])}
    why = why_texts(cfg, "player", [p for p in pids if p])
    rep = Report(f"Chess report · {cfg['player']['username']}", home=rel(data_root(cfg) / "index.html", P),
                 subtitle=f"Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(F)} findings · "
                          "weaknesses ordered by expected rating gain")

    def fid(f):
        return slugify(f"{f['section']}-{f['title']}")

    def examples(f, seen):
        """Boards for one finding; positions already shown earlier in the same tab are skipped."""
        out, dup = [], 0
        for e in f["examples"]:
            if seen is not None:
                if e["fen"] in seen:
                    dup += 1
                    continue
                seen.add(e["fen"])
            note = esc(e.get("note", ""))
            w = why.get(e.get("pos_id"))
            if w:
                note += f"<br><i>{esc(w)}</i>"
            out.append(rep.example(e["fen"], e["best"], e["played"] or None, note))
        return "".join(out), dup

    def block(f, open_=True, with_examples=True, seen=None):
        ev = ""
        if f.get("evidence"):
            ev = Report.table(f["evidence"]["header"], [[esc(c) for c in r] for r in f["evidence"]["rows"]])
        exs, dup = examples(f, seen) if with_examples else ("", 0)
        if dup:
            ev += (f"<p class='muted'>{'All' if not exs else dup} of its example positions "
                   f"{'are' if dup > 1 or not exs else 'is'} already shown above.</p>")
        title = plain_title(f["title"])
        return Report.finding(title, f["claim"], f["tag"], ev, exs, f.get("todo", ""), f.get("elo"),
                              f.get("elo_note", ""), f.get("link"), open_,
                              sub=f["title"] if title != f["title"] else "", fid=fid(f))

    # ---------------- drills
    themes: dict[str, dict] = {}
    for f in F:
        d = f.get("drill")
        if not d or f["tag"] not in WEAK:
            continue
        t = themes.setdefault(d["theme"], {"elo": 0.0, "puzzles": [], "seen": set()})
        t["elo"] += f.get("elo") or 0
        for p in d.get("pos_ids", []):
            if p in pos.index and p not in t["seen"]:
                r = pos.loc[p]
                t["seen"].add(p)
                c = f"You played {san(r.fen, r.move)} (-{100 * r.wp_loss:.0f})."
                if p in why:
                    c += f" Why: {why[p]}."
                t["puzzles"].append({"fen": r.fen, "solution": list(r.best_line), "comment": c, "date": r.date,
                                     "wrong_line": list(r.played_line)})
        for z in d.get("puzzles", []):
            if z["fen"] not in t["seen"]:
                t["seen"].add(z["fen"])
                t["puzzles"].append({**z, "date": ""})
    drill_files = {}
    for name, t in sorted(themes.items(), key=lambda kv: -kv[1]["elo"]):
        pz = t["puzzles"][: args.max_puzzles]
        if len(pz) < 3:
            continue
        ch = [puzzle_chapter(f"{name} #{i + 1} {z['date']}".strip(), z["fen"], z["solution"], z["comment"],
                             wrong_line=z.get("wrong_line"))
              for i, z in enumerate(pz)]
        drill_files[name] = (write_studies(ch, P / f"drills_{slug(name)}"), pz)

    # ---------------- summary
    S = rep.tab("summary", "Summary")
    mbp = P / "multiband.parquet"
    kp = []
    if mbp.exists():
        mb = pd.read_parquet(mbp)
        for metric, cat, label in (("loss per 100 moves", "all moves", "your level (accuracy)"),
                                   ("loss per 100 moves", "eval: contested (0.2-0.8)", "in contested positions"),
                                   ("blunder rate", "all moves", "by blunders")):
            r = mb[(mb.metric == metric) & (mb.category == cat)]
            if len(r) and pd.notna(r.level.iloc[0]):
                r = r.iloc[0]
                kp.append((label, f"~{r.level:.0f}", f"Lichess rapid, 90% CI {r.level_lo:.0f}–{r.level_hi:.0f}"))
    weak_all = sorted([f for f in F if f["tag"] in WEAK and f["section"] in dict(SECTIONS)], key=gain_key)
    total = sum(f.get("elo") or 0 for f in weak_all[:10])
    kp.append(("Top 10 fixes", f"≈ +{total:.0f}", "Elo if all were fixed (estimates overlap: an upper bound)"))
    if weak_all:
        top = weak_all[0]
        S.append(f"<p class='lede'>Your biggest single gain is in the <b>{esc(dict(SECTIONS)[top['section']].lower())}"
                 f"</b>: {esc(plain_title(top['title']).lower())} (≈ +{top.get('elo') or 0:.0f} Elo).</p>")
    S.append(Report.kpis([(a[:1].upper() + a[1:], b, c) for a, b, c in kp]))
    S.append("<h2>Where your rating can grow</h2><p class='muted'>Expected rating gain if the weakness were fixed "
             "(to the next rating band's level, or the engine's move in openings). Estimates overlap, so don't add "
             "them up. Click a title for the evidence and example positions.</p><ol class='rank'>")
    top_elo = max([f.get("elo") or 0 for f in weak_all[:6]] + [1e-9])
    for f in weak_all[:6]:
        elo = f.get("elo") or 0
        todo = f.get("todo", "")
        S.append(f"<li><div class='rh'><span class='rt'><span class='phase'>{esc(dict(SECTIONS)[f['section']])}</span><br>"
                 f"<a href='#{f['section']}/{fid(f)}'>{esc(plain_title(f['title']))}</a></span>"
                 f"<span class='tag t-{esc(f['tag'])}'>{esc(f['tag'])}</span>"
                 f"<span class='gain'>≈ +{elo:.0f} Elo</span></div>"
                 f"<span class='gbar'><i style='width:{100 * elo / top_elo:.0f}%'></i></span>"
                 f"<p>{f['claim']}</p>"
                 + (f"<div class='todo'><span class='todo-k'>What to do</span> {todo}</div>" if todo else "")
                 + "</li>")
    S.append("</ol>")
    good = [f for f in F if f["tag"] == "strength"]
    style = [f for f in F if f["tag"] == "style"]
    if good:
        S.append("<h2>What's working</h2><ul class='okl'>" + "".join(
            f"<li><a href='#{f['section']}/{fid(f)}'>{esc(plain_title(f['title']))}</a></li>"
            if f["section"] in dict(SECTIONS) else f"<li>{esc(plain_title(f['title']))}</li>" for f in good) + "</ul>")
    if style:
        S.append(f"<details class='plain'><summary>Style: {len(style)} areas where you play differently from peers "
                 "without losing points</summary><ul>"
                 + "".join(f"<li>{esc(plain_title(f['title']))}</li>" for f in style) + "</ul></details>")

    # ---------------- phase tabs
    for sec, label in SECTIONS:
        T = rep.tab(sec, label)
        weak = sorted([f for f in F if f["section"] == sec and f["tag"] in WEAK], key=gain_key)
        rest = [f for f in F if f["section"] == sec and f["tag"] not in WEAK]
        seen: set = set()
        T.append(f"<h2>{label}: {len(weak)} area{'s' if len(weak) != 1 else ''} to improve</h2>"
                 "<p class='muted'>Largest expected gain first. Each block: the finding, what to do, the numbers, then "
                 "positions from your games (arrows: <span class='ok'>green</span> = best move, "
                 "<span class='no'>red</span> = your move; ← → step through the line).</p>")
        T += [block(f, open_=i < 3, seen=seen) for i, f in enumerate(weak)]
        if not weak:
            T.append("<p>No weaknesses passed the rules here.</p>")
        if rest:
            T.append("<h2>What's working and context</h2>" + "".join(block(f, open_=False, seen=seen) for f in rest))

    # ---------------- exploitation, games, drills, method
    X = rep.tab("exploit", "Exploitation")
    X.append("<p class='muted'>Exploratory: models of your opponents (Lichess book at their level, Maia where thin). "
             "Looser evidence than the phase tabs.</p>")
    xseen: set = set()
    X += [block(f, open_=i < 3, seen=xseen)
          for i, f in enumerate(sorted([f for f in F if f["section"] == "exploit"], key=gain_key))]
    G = rep.tab("games", "Games")
    games = [f for f in F if f["section"] == "games"]
    G.append("<p class='muted'>Imported games (e.g. over-the-board): <code>python scripts/22_import_games.py --pgn "
             "games.pgn --name \"Your Name\"</code>. Game ratings are rough (one game is a small sample).</p>")
    G += [block(f, open_=i < 2) for i, f in enumerate(games)] or ["<p>No imported games yet.</p>"]
    D = rep.tab("drills", "Drills")
    D.append("<p class='muted'>Find-the-move puzzles from your own games, one Lichess study per theme (import at "
             "lichess.org/study → Import PGN). Themes are ordered by expected gain. Do a short set, come back in a few "
             "days (retrieval + spacing).</p>")
    for name, (files, pz) in drill_files.items():
        ex = "".join(rep.example(z["fen"], z["solution"], None, esc(z["comment"]), puzzle=True) for z in pz[:6])
        D.append(Report.finding(plain_title(name), f"{len(pz)} puzzles · " + ", ".join(esc(p.name) for p in files), "lead",
                                examples=ex, open_=False))
    rf = sorted(p.name for p in P.glob("repertoire_*.pgn"))
    if rf:
        D.append(Report.finding("Opening repertoire", "Your lines extended with the follow-ups: "
                                + ", ".join(esc(f) for f in rf), "info", open_=False))
    M = rep.tab("method", "Method")
    M.append("<h2>How to read this</h2><ul>"
             "<li><span class='tag t-solid'>solid</span> the engine finds more errors than peers make and a second "
             "measure agrees.</li>"
             "<li><span class='tag t-lead'>lead</span> one measure only, or a borderline interval.</li>"
             "<li><span class='tag t-strength'>strength</span> fewer errors than peers, confirmed twice.</li>"
             "<li><span class='tag t-style'>style</span> different from peers, not costing points.</li>"
             "<li><span class='tag t-info'>info</span> context. <span class='tag t-exploratory'>exploratory</span> "
             "opponent models, looser evidence.</li>"
             "<li>Titles are in plain words; the small grey line under a title is the exact category used in the "
             "data.</li>"
             f"<li><b>≈ +N Elo</b>: expected score gained per game if fixed × {ELO_PER_SCORE:.0f} (rating points per "
             "point of score at an even score). Win-probability loss stands in for score.</li></ul>")
    M.append(f"<p class='muted'>Generated {time.strftime('%Y-%m-%d %H:%M')} from {len(F)} findings.</p>")
    Fi = rep.tab("files", "Files")
    me_name = cfg["player"]["username"].lower()
    comps = [c for c in sorted((data_root(cfg) / "compare").glob("*/report.html")) if me_name in c.parent.name.split("_vs_")]
    if comps:
        Fi.append("<h2>Head-to-head</h2><ul>" + "".join(
            f"<li><a href='{esc(rel(c, P))}'>{esc(c.parent.name.replace('_vs_', ' vs '))}</a></li>" for c in comps) + "</ul>")
    Fi.append(files_section(P, sorted(p for p in P.glob("*.html") if p.name != "report.html")
                            + sorted(data_dir(cfg, "ideas").glob("*.html")), sorted(P.glob("*.pgn"))))
    out = P / "report.html"
    rep.write(out)
    print(f"index: {build_index(cfg)}")
    print(f"{len(F)} findings ({len(weak_all)} weaknesses), {len(drill_files)} drill themes, {len(rep.examples)} boards")
    for f in weak_all[:8]:
        print(f"   {f['section']:10s} {f.get('elo') or 0:6.1f} Elo  {f['title']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
