"""The report: one interactive HTML file + drill studies, built from what the other stages saved.

Tabs: Summary · Opening · Middlegame · Endgame · Across phases · Exploitation (exploratory) · Drills · Method.
Every finding is generated from the data by fixed rules (nothing written for one player):

  engine verdict  your error rate vs peers in that category: worse / better / par (90% CI of the ratio, 16)
  Maia verdict    your Maia level there vs your overall level (16): lower / higher / par
  multi-band      your Lichess level there vs your all-moves level (17, error rate)
  tag             solid    = engine worse AND (Maia lower OR multi-band <= -100)
                  lead     = engine worse only
                  strength = engine better AND (Maia higher OR multi-band >= +100), unless inflated (below)
                  style    = engine par but Maia clearly different (not costing points)
                  info     = looks better but may be inflated: if most of your lost games end by mate or
                             flag rather than resignation (detected from game endings), error rates in
                             losing / endgame / low-clock positions are flattered
Examples are your own error positions in the category (games >= player.min_player_rating), with the
best line and your line to step through. Drills: one Lichess study per weakness theme (puzzles from
your games where only one move was good), written to data/player/drills_<theme>.pgn.

    python scripts/20_report.py --config configs/v0.yaml

Output: data/player/report.html, data/player/drills_*.pgn
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.categories import annotate, categories
from coach.config import data_dir, load_config
from coach.outcomes import mover_results
from coach.results import load_analyzed
from coach.study import puzzle_chapter, write_studies
from coach.webreport import Report, esc

ORDER = {"solid": 0, "lead": 1, "strength": 2, "info": 3, "style": 4}
INFLATABLE = ("losing", "endgame", "clock: < 1 min")


def read(path):
    return pd.read_parquet(path) if path.exists() else None


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def san(fen, uci):
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def phase_of(cat: str) -> str:
    c = cat.lower()
    if "endgame" in c:
        return "endgame"
    if "middlegame" in c or c.startswith("critical"):
        return "middlegame"
    if "opening" in c:
        return "opening"
    return "across"


def boot_ci(x, B=1000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    m = x[np.random.default_rng(seed).integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, 5), np.percentile(m, 95)


def never_resigns(sa) -> tuple[bool, str]:
    if sa is None or "termination" not in sa:
        return False, ""
    lost = sa[sa.S == 0]
    if len(lost) < 20:
        return False, ""
    resigned = (lost.termination == "resigned").mean()
    return resigned < 0.3, f"{resigned:.0%} of your lost games ended by resignation"


def category_findings(skill, mb, inflated):
    """One row per category with verdicts and a tag (None = nothing to say)."""
    lv = {}
    if mb is not None:
        m = mb[mb.metric == "error rate"]
        base = m[m.category == "all moves"].level
        base = float(base.iloc[0]) if len(base) and base.notna().iloc[0] else np.nan
        for r in m.itertuples():
            lv[r.category] = (r.level, r.level - base if np.isfinite(base) and np.isfinite(r.level) else np.nan,
                              r.level_lo, r.level_hi)
    out = []
    for r in skill.itertuples():
        if r.category == "all moves" or not np.isfinite(r.ratio):
            continue
        eng = "worse" if r.ratio_lo > 1 else "better" if r.ratio_hi < 1 else "par"
        maia = "lower" if r.hi < 0 else "higher" if r.lo > 0 else "par"
        level, rel_mb, llo, lhi = lv.get(r.category, (np.nan, np.nan, np.nan, np.nan))
        infl = inflated and any(k in r.category for k in INFLATABLE)
        if eng == "worse":
            tag = "solid" if maia == "lower" or (np.isfinite(rel_mb) and rel_mb <= -100) else "lead"
        elif eng == "better":
            tag = "info" if infl else ("strength" if maia == "higher" or (np.isfinite(rel_mb) and rel_mb >= 100) else None)
        else:
            tag = "style" if maia != "par" and abs(r.relative) >= 60 else None
        if tag:
            out.append({"category": r.category, "tag": tag, "eng": eng, "maia": maia, "ratio": r.ratio,
                        "ratio_lo": r.ratio_lo, "ratio_hi": r.ratio_hi, "err_you": r.err_you, "err_peers": r.err_peers,
                        "rel": r.relative, "rel_lo": r.lo, "rel_hi": r.hi, "level": level, "rel_mb": rel_mb,
                        "level_lo": llo, "level_hi": lhi, "phase": phase_of(r.category), "inflated": infl})
    out.sort(key=lambda f: (ORDER[f["tag"]], -abs(f["ratio"] - 1)))
    return out


def claim_text(f):
    s = (f"Your error rate here is <b>{f['err_you']:.0%}</b> vs {f['err_peers']:.0%} for peers "
         f"({f['ratio']:.2f}×, 90% CI {f['ratio_lo']:.2f}–{f['ratio_hi']:.2f}). Maia rates your move choices "
         f"{f['rel']:+.0f} vs your overall level")
    if np.isfinite(f["level"]):
        s += f"; on the Lichess band scale this plays like ~{f['level']:.0f} ({f['rel_mb']:+.0f} vs your overall)"
    s += "."
    if f["tag"] == "style":
        s += " The engine sees no difference in errors, so this is a difference in style, not a leak."
    if f["inflated"]:
        s += " Caution: you rarely resign, so lost positions get played out and rarely count as errors here."
    return s


def examples_for(rep, df, mask, n=3, puzzle=False, note_prefix=""):
    sub = df[mask & df.err.to_numpy()].sort_values(["critical", "wp_loss"], ascending=False).head(n)
    out = []
    for r in sub.itertuples():
        note = (f"{note_prefix}{esc(r.date)} · you played {esc(san(r.fen, r.move))}, losing "
                f"{100 * r.wp_loss:.0f} points · best {esc(san(r.fen, r.best_move))}")
        out.append(rep.example(r.fen, list(r.best_line), list(r.played_line), note, puzzle=puzzle))
    return "".join(out)


def conversion(df, results, cut=0.8):
    """Per game: first endgame position with wp_best > cut -> final score."""
    d = df.assign(res=results)
    d = d[(d.ph == "endgame") & (d.wp_best > cut) & d.res.notna()].sort_values("ply")
    g = d.groupby("game_id").res.first()
    return g.to_numpy(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--max-puzzles", type=int, default=128, help="per theme (split into 64-chapter studies)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cmp_min = cfg["player"]["min_player_rating"]
    P, O = data_dir(cfg, "player"), data_dir(cfg, "openings")
    me = annotate(load_analyzed(cfg, "player"))
    recent = me[me.mover_rating >= cmp_min].reset_index(drop=True)
    cats = categories(recent)
    skill, mb, sa = read(P / "skill_profile.parquet"), read(P / "multiband.parquet"), read(P / "score_accounting.parquet")
    inflated, why_infl = never_resigns(sa)
    F = category_findings(skill, mb, inflated) if skill is not None else []
    rep = Report(f"Chess report · {cfg['player']['username']}")
    stamp = time.strftime("%Y-%m-%d")

    # ---------------- drills (built first so findings can link to them)
    drill_files = {}
    pz = {}
    for f in F:
        if f["tag"] not in ("solid", "lead") or f["category"] not in cats:
            continue
        m = cats[f["category"]] if f["category"] in cats else None
        allc = categories(me)[f["category"]]
        cand = me[allc & me.err.to_numpy() & me.critical.astype(bool).to_numpy()]
        cand = cand.assign(recent=(cand.mover_rating >= cmp_min)).sort_values(["recent", "wp_loss"], ascending=False)
        cand = cand.head(args.max_puzzles)
        if len(cand) < 5:
            continue
        chapters = [puzzle_chapter(f"{f['category']} #{i + 1} ({r.date})", r.fen, list(r.best_line),
                                   f"You played {san(r.fen, r.move)} (-{100 * r.wp_loss:.0f}). "
                                   f"Only one move keeps it: {san(r.fen, r.best_move)}.")
                    for i, r in enumerate(cand.itertuples())]
        drill_files[f["category"]] = write_studies(chapters, P / f"drills_{slug(f['category'])}")
        pz[f["category"]] = cand.head(6)
        del m

    def todo(f):
        if f["tag"] in ("solid", "lead"):
            files = drill_files.get(f["category"])
            return ("Drill your own positions: " + ", ".join(esc(p.name) for p in files)
                    + " (Drills tab; import at lichess.org/study).") if files else "Review the examples below."
        if f["tag"] == "style":
            return "Nothing to fix: different from peers, not costing points."
        return ""

    def finding_html(f, n_ex=3):
        mask = cats.get(f["category"], np.zeros(len(recent), bool))
        ex = examples_for(rep, recent, mask, n_ex) if f["tag"] in ("solid", "lead") else ""
        ev = Report.table(["measure", "you", "peers / reference", "verdict"], [
            ["engine error rate", f"{f['err_you']:.1%}", f"{f['err_peers']:.1%}", f["eng"]],
            ["Maia level vs your overall", f"{f['rel']:+.0f} [{f['rel_lo']:+.0f}, {f['rel_hi']:+.0f}]", "0", f["maia"]],
            ["Lichess band level", f"{f['level']:.0f}" if np.isfinite(f["level"]) else "–",
             f"{f['rel_mb']:+.0f} vs overall" if np.isfinite(f["rel_mb"]) else "–", ""]])
        return Report.finding(f["category"], claim_text(f), f["tag"], ev, ex, todo(f))

    # ---------------- summary
    S = rep.tab("summary", "Summary")
    kp = []
    if mb is not None:
        def lvl(metric, cat):
            r = mb[(mb.metric == metric) & (mb.category == cat)]
            return r.iloc[0] if len(r) else None
        for metric, label in (("loss per 100 moves", "level by loss"), ("error rate", "level by error rate"),
                              ("blunder rate", "level by blunders")):
            r = lvl(metric, "all moves")
            if r is not None and np.isfinite(r.level):
                kp.append((label, f"~{r.level:.0f}", f"Lichess, 90% CI {r.level_lo:.0f}–{r.level_hi:.0f}"))
        r = lvl("loss per 100 moves", "eval: contested (0.2-0.8)")
        if r is not None and np.isfinite(r.level):
            kp.append(("contested positions", f"~{r.level:.0f}", "Lichess; the cleanest single number"))
    if sa is not None:
        rs = sa[sa.rating >= cmp_min]
        kp.append(("games analysed", f"{len(rs)}", f"score {rs.S.mean():.0%} (games >= {cmp_min})"))
    S.append("<div class='kpis'>" + "".join(f"<div class='kpi'><span class='muted'>{esc(a)}</span><b>{esc(b)}</b>"
                                            f"<span class='muted'>{esc(c)}</span></div>" for a, b, c in kp) + "</div>")
    S.append("<p class='muted'>Levels are on the Lichess rapid scale from five rating bands (rough: ±100). "
             + (esc(why_infl) + ", so endgame / losing / low-clock numbers are flattered. " if inflated else "")
             + f"Generated {stamp}.</p>")
    weak = [f for f in F if f["tag"] in ("solid", "lead")][:3]
    strong = [f for f in F if f["tag"] == "strength"][:3]
    style = [f for f in F if f["tag"] == "style"]
    S.append("<h2>Top fixes</h2>")
    S += [Report.finding(f["category"], claim_text(f), f["tag"], todo=todo(f)) for f in weak]
    oi = json.loads((O / "report_items.json").read_text()) if (O / "report_items.json").exists() else None
    if oi and oi["costly"]:
        c = max(oi["costly"], key=lambda x: x["count"] * x["loss"])
        S.append(Report.finding("Opening line", f"{esc(c['line'])}: you played this {c['count']}x and it costs "
                                f"{100 * c['loss']:.0f} points; play {esc(san(c['fen'], c['best']))}.", "solid",
                                todo="See the Opening tab and the repertoire study."))
    S.append("<h2>Strengths</h2>" + ("".join(Report.finding(f["category"], claim_text(f), "strength") for f in strong)
                                      or "<p class='muted'>None clear on both engine and Maia/bands.</p>"))
    if style:
        S.append("<h2>Style (not costing points)</h2><p>" + ", ".join(esc(f["category"]) for f in style) + "</p>")

    # ---------------- phase tabs
    T = {"opening": rep.tab("opening", "Opening"), "middlegame": rep.tab("middlegame", "Middlegame"),
         "endgame": rep.tab("endgame", "Endgame"), "across": rep.tab("across", "Across phases")}
    # opening specifics
    Op = T["opening"]
    if oi:
        rows = [[c, f"{v['you']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]", f"{v['peers']:.3f}", v["n"]]
                for c, v in oi["out"].items()]
        Op.append(Report.finding("Out of the opening", "Your win probability at move 11 (side to move = you), vs peers. "
                                 "Engine-judged; 0.5 = equal.", "info",
                                 Report.table(["as", "you (90% CI)", "peers", "games"], rows)))
        if oi["costly"]:
            ex = "".join(rep.example(c["fen"], [c["best"]], [c["move"]],
                                     f"{esc(c['line'])} · {c['count']}x · score {c['score']:.2f}") for c in oi["costly"][:6])
            Op.append(Report.finding("Moves you repeat that cost points",
                                     "Ranked by how often × how much (engine only; under 5 points is theory noise).",
                                     "solid", examples=ex, todo="Learn the green move; the repertoire study has the follow-ups."))
        if oi.get("missed_punish"):
            ex = "".join(rep.example(c["fen"], [c["best"]], [c["move"]], f"{esc(c['line'])} · missed {c['count']}x")
                         for c in oi["missed_punish"][:4])
            Op.append(Report.finding("Opponents' opening mistakes you didn't punish",
                                     f"You punished {oi.get('punish_rate', float('nan')):.0%} of opponents' opening "
                                     f"mistakes ({oi.get('punish_n', 0)} cases).", "lead", examples=ex))
    rp = [read(P / f"repertoire_points_{c}.parquet") for c in ("white", "black")]
    rp = pd.concat([d for d in rp if d is not None], ignore_index=True) if any(d is not None for d in rp) else None
    if rp is not None and len(rp):
        top = rp.sort_values("value", ascending=False).head(8)
        ex = "".join(rep.example(r.fen, [r.best], [r.move], f"{esc(r.line)} · you: {r.count}x, score {r.score:.2f}")
                     for r in top.itertuples())
        files = sorted(p.name for p in P.glob("repertoire_*.pgn"))
        Op.append(Report.finding("Your lines, extended", "Where a move you play often has a better alternative, "
                                 "ranked by how often × cost. The full trees are in the repertoire studies.", "info",
                                 examples=ex, todo="Import " + ", ".join(esc(f) for f in files) + " at lichess.org/study."))
    if sa is not None and "loss opening" in sa:
        rs = sa[sa.rating >= cmp_min]
        net = rs["loss opening"] + rs["gift opening"]
        lo, hi = boot_ci(net)
        Op.append(Report.finding("Points won/lost in moves ~10–15", f"Net {net.mean():+.3f} points per game vs your "
                                 f"opponents (90% CI {lo:+.3f} to {hi:+.3f}).", "info"))
    for f in F:
        if f["phase"] in T:
            T[f["phase"]].append(finding_html(f))
    # endgame conversion by outcome
    res_me = mover_results(cfg, recent, "player")
    pop = annotate(load_analyzed(cfg, "population"))
    res_pop = mover_results(cfg, pop, "population")
    cm, cp = conversion(recent, res_me), conversion(pop, res_pop)
    if len(cm) >= 10 and len(cp) >= 10:
        lo, hi = boot_ci(cm)
        tag = "strength" if lo > cp.mean() else ("solid" if hi < cp.mean() else "info")
        T["endgame"].insert(0, Report.finding("Converting winning endgames (by result)",
                                              f"From the first winning endgame position (> 80%) you scored <b>{cm.mean():.2f}</b> "
                                              f"[{lo:.2f}, {hi:.2f}] (n {len(cm)}) vs peers {cp.mean():.2f} (n {len(cp)}).", tag))
    for k, label in (("middlegame", "Middlegame"), ("endgame", "Endgame")):
        if len(T[k]) == 0:
            T[k].append(f"<p class='muted'>No {label.lower()} findings passed the rules.</p>")
    # across phases: tables
    A = T["across"]
    if sa is not None:
        rs = sa[sa.rating >= cmp_min]
        rows = []
        for name, col in (("opening (moves 1–10)", ["opening (moves 1-10)"]),
                          ("moves ~10–15", ["loss opening", "gift opening"]),
                          ("middlegame", ["loss middlegame", "gift middlegame"]),
                          ("endgame", ["loss endgame", "gift endgame"]), ("end of game", ["end of game"])):
            if all(c in rs for c in col):
                x = rs[col].sum(axis=1)
                lo, hi = boot_ci(x)
                rows.append([name, Report.level_chip(100 * x.mean() / 0.05, f"{x.mean():+.3f}"), f"[{lo:+.3f}, {hi:+.3f}]"])
        A.append(Report.finding("Where your points come from", "Per game, net vs your actual opponents "
                                "(telescoping engine accounting; sums to your score − 0.5).", "info",
                                Report.table(["phase", "net points/game", "90% CI"], rows)))
    if mb is not None:
        m = mb[mb.metric == "error rate"].copy()
        base = m[m.category == "all moves"].level
        base = float(base.iloc[0]) if len(base) else np.nan
        rows = [[esc(r.category), f"{r.level:.0f}" if np.isfinite(r.level) else "style (flat)",
                 Report.level_chip(r.level - base if np.isfinite(r.level) else None)]
                for r in m.itertuples()]
        A.append(Report.finding("Your level by category (multi-band)", "Lichess level whose error rate matches yours "
                                "in each kind of position; colour = vs your overall level.", "info",
                                Report.table(["category", "level", "vs overall"], rows)))
    # ---------------- exploitation
    X = rep.tab("exploit", "Exploitation")
    X.append("<p class='muted'>Exploratory: models of your opponents (Lichess book at their level, Maia where thin). "
             "Looser evidence than the analysis tabs.</p>")
    ta = read(O / "traps_A.parquet")
    if ta is not None and len(ta):
        ta = ta.assign(v=ta.visits * ta.exp_loss).sort_values("v", ascending=False).head(6)
        ex = "".join(rep.example(r.fen, [r.best], [r.trap], f"{esc(r.path)} · {r.err_mass:.0%} go wrong (book n {r.book_n})",
                                 labels=("right move", "common mistake")) for r in ta.itertuples())
        X.append(Report.finding("Positions you reach where opponents often go wrong", "Know the punishment.",
                                "exploratory", examples=ex))
    tb = read(O / "traps.parquet")
    if tb is not None and len(tb):
        rows = []
        for e, g in tb.dropna(subset=["ev"]).groupby("epd"):
            sf = g[g.is_best]
            if sf.empty:
                continue
            sf, tr = sf.iloc[0], g.loc[g.ev.idxmax()]
            if tr.move != sf.move and tr.ev - sf.ev >= 0.02:
                rows.append((tr.visits * (tr.ev - sf.ev), tr, sf))
        rows.sort(key=lambda x: -x[0])
        ex = "".join(rep.example(tr.fen, [tr.move], [sf.move],
                                 f"{esc(tr.path)} · expected +{100 * (tr.ev - sf.ev):.1f}, cost {100 * (sf.wp - tr.wp):.1f} · "
                                 f"{esc(tr.src)}", labels=("trap move", "engine move")) for _, tr, sf in rows[:6])
        if ex:
            X.append(Report.finding("Trap moves within a small margin", "Moves almost as good as the engine's that give "
                                    "your opponents more chances to go wrong.", "exploratory", examples=ex))
    # ---------------- drills
    D = rep.tab("drills", "Drills")
    D.append("<p class='muted'>Find-the-move puzzles from your own games, one Lichess study per weakness "
             "(import at lichess.org/study → Import PGN). Retrieval + spacing: do a short set, come back in a few days.</p>")
    for cat, files in drill_files.items():
        sub = pz[cat]
        ex = "".join(rep.example(r.fen, list(r.best_line), None, f"{esc(r.date)}", puzzle=True) for r in sub.itertuples())
        D.append(Report.finding(cat, f"{sum(1 for _ in files)} study file(s): " + ", ".join(esc(p.name) for p in files),
                                "lead", examples=ex))
    files = sorted(p.name for p in P.glob("repertoire_*.pgn"))
    if files:
        D.append(Report.finding("Opening repertoire", "Your lines extended with the engine's follow-ups: " +
                                ", ".join(esc(f) for f in files), "info"))
    # ---------------- method
    M = rep.tab("method", "Method")
    M.append("<h2>How to read this</h2><ul>"
             "<li><b>solid</b>: the engine finds more errors than peers make AND Maia or the rating bands agree.</li>"
             "<li><b>lead</b>: only the engine sees it (or the interval is borderline).</li>"
             "<li><b>strength</b>: fewer errors than peers, confirmed by a second measure.</li>"
             "<li><b>style</b>: your move choices differ (Maia) but the engine finds no extra errors.</li>"
             "<li><b>info</b>: context, or a 'strength' that may be inflated by never resigning.</li></ul>"
             f"<p>Peers: Lichess rapid players around your level; comparisons use your games at chess.com ≥ {cmp_min}. "
             "Error = mistake or blunder (losing ≥ 3 win-probability points). Engine: Stockfish; human model: Maia-3. "
             "Details per measure: the per-script reports below and the repository docs.</p>")
    old = sorted(p.name for p in P.glob("*.html") if p.name != "report.html")
    M.append("<h3>Detailed reports</h3><ul>" + "".join(f"<li><a href='{esc(n)}'>{esc(n)}</a></li>" for n in old) + "</ul>")
    out = P / "report.html"
    rep.write(out)
    print(f"{len(F)} findings ({sum(f['tag'] in ('solid', 'lead') for f in F)} weaknesses), "
          f"{len(drill_files)} drill themes, {len(rep.examples)} boards")
    print(f"wrote {out}" + "".join(f"\n      {p}" for fs in drill_files.values() for p in fs))


if __name__ == "__main__":
    main()
