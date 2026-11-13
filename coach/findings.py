"""Findings registry: the contract between analysis modules and the report.

Each module writes data/findings/<module>.json (a list of findings) and nothing else needs to know
about it: 20_report.py loads every file and orders them. To add an analysis, write findings here;
don't edit the report. (This keeps parallel work on separate modules from colliding.)

A finding (dict):
  id        unique within the module                       section  opening | middlegame | endgame | exploit
  title     short name                                     claim    one or two sentences (HTML allowed)
  tag       solid | lead | strength | style | info | exploratory
  elo       expected rating gain if fixed (see elo_gain), or None   elo_note  how it was computed
  rank      optional ordering key (Elo x trainability); the report sorts by rank, else elo
  detail    True = "in depth" item: tables / boards specific to a section, shown at the bottom of its tab
            (replaces the old standalone detail pages); use detail() for these
  evidence  {"header": [...], "rows": [[...]]} or None
  examples  [{"fen", "best": [uci], "played": [uci], "note"}]  (your positions; boards in the report)
  drill     {"theme": str, "pos_ids": [player pos_ids]} or None  (puzzles; the report builds the study)
  todo      what to do (HTML allowed)                     link     detailed report file name, or None
"""
from __future__ import annotations

import json
import math

from .config import data_dir

SECTIONS = ("opening", "middlegame", "endgame", "exploit", "games")
# d(rating)/d(score) of the Elo logistic at a 50% score: 400 / (ln 10 * 0.25)
ELO_PER_SCORE = 400 / (math.log(10) * 0.25)


def elo_gain(win_prob_per_game: float) -> float:
    """Expected score gained per game (win-prob points, 0..1) -> rating points at an even score.
    Win-prob loss is a reasonable proxy for expected score (it is fitted on game outcomes)."""
    return ELO_PER_SCORE * max(0.0, float(win_prob_per_game))


def finding(id, section, title, claim, tag, elo=None, elo_note="", evidence=None, examples=None,
            drill=None, todo="", link=None, rank=None, detail=False) -> dict:
    """rank: optional ordering key (e.g. Elo x trainability); the report sorts by it, else by elo.
    detail: an "in depth" item (section-specific tables / boards) shown at the bottom of its tab."""
    assert section in SECTIONS, section
    num = lambda x: None if x is None or x != x else round(float(x), 1)
    return {"id": str(id), "section": section, "title": title, "claim": claim, "tag": tag,
            "elo": num(elo), "elo_note": elo_note, "rank": num(rank), "detail": bool(detail),
            "evidence": evidence, "examples": examples or [], "drill": drill, "todo": todo, "link": link}


def detail(id, section, title, claim, evidence=None, examples=None) -> dict:
    """Shorthand for an "in depth" item (tag info, no Elo)."""
    return finding(id, section, title, claim, "info", evidence=evidence, examples=examples, detail=True)


def table(header, rows) -> dict:
    """Evidence table in the findings format (cells are HTML-escaped by the report)."""
    return {"header": [str(h) for h in header], "rows": [[str(c) for c in r] for r in rows]}


def example(fen, best, played=None, note="", pos_id=None, labels=None) -> dict:
    """labels: names of the two lines on the board (default "best line" / "your move")."""
    as_list = lambda x: [] if x is None else [str(u) for u in list(x)]
    out = {"fen": fen, "best": as_list(best), "played": as_list(played), "note": note, "pos_id": pos_id}
    if labels:
        out["labels"] = list(labels)
    return out


def write(cfg, module: str, items: list[dict], expand: bool = True) -> None:
    """Save a module's findings. expand: one-move example lines / puzzle solutions get the engine's
    follow-up (coach/lines.py, cached), so every opening finding shows the idea, not just a move."""
    if expand:
        try:
            from .lines import expand_examples
            expand_examples(cfg, items)
        except Exception as e:                           # never lose findings over a follow-up failure
            print(f"  (follow-up lines skipped: {e})")
    d = data_dir(cfg, "findings")
    for f in items:
        f["module"] = module
    tmp = d / f"{module}.json.tmp"
    tmp.write_text(json.dumps(items, indent=1, default=float), encoding="utf-8")
    tmp.replace(d / f"{module}.json")


def load_all(cfg) -> list[dict]:
    out = []
    for p in sorted(data_dir(cfg, "findings").glob("*.json")):
        try:
            out += json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  skipping unreadable findings file {p.name}")
    return out
