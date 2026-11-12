"""Findings per kind of position (middlegame / endgame categories), ranked by expected Elo gain.

Combines the skill profile (16: engine error ratio vs peers + Maia level) and the multi-band fits (17)
into findings for the report (coach/findings.py), one per category that says something:

  tag        solid = engine worse than peers AND (Maia lower OR band level <= -100 vs your overall)
             lead = engine worse only · strength = engine better AND a second measure agrees
             style = engine par, Maia clearly different · info = 'better' but inflated by never resigning
  elo        expected rating gain if you played this category like the next rating band above you:
             ELO_PER_SCORE x max(0, your loss/move - that band's loss/move) x your moves/game in it.
             Categories overlap (critical positions are also middlegame), so gains don't add up.
  examples   your worst errors in the category; drill = critical errors (one good move) as puzzles.
Also: converting winning endgames, judged by game results (you vs peers).

    python scripts/17b_category_findings.py --config configs/v0.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coach.categories import annotate, categories
from coach.config import data_dir, load_config
from coach.findings import ELO_PER_SCORE, elo_gain, example, finding, write
from coach.outcomes import mover_results
from coach.results import load_analyzed

INFLATABLE = ("losing", "endgame", "clock: < 1 min")


def section_of(cat: str) -> str:
    c = cat.lower()
    if "endgame" in c:
        return "endgame"
    if "opening" in c:
        return "opening"
    return "middlegame"


def never_resigns(cfg) -> tuple[bool, str]:
    p = data_dir(cfg, "player") / "score_accounting.parquet"
    if not p.exists():
        return False, ""
    sa = pd.read_parquet(p)
    lost = sa[sa.S == 0]
    if "termination" not in sa or len(lost) < 20:
        return False, ""
    r = (lost.termination == "resigned").mean()
    return r < 0.3, f"only {r:.0%} of your lost games ended by resignation"


def boot_ci(x, B=1000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    m = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, 5), np.percentile(m, 95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--max-drill", type=int, default=128)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cmp_min = cfg["player"]["min_player_rating"]
    P = data_dir(cfg, "player")
    skill = pd.read_parquet(P / "skill_profile.parquet") if (P / "skill_profile.parquet").exists() else None
    mb = pd.read_parquet(P / "multiband.parquet") if (P / "multiband.parquet").exists() else None
    if skill is None:
        sys.exit("run 16_skill_profile.py first")
    me_all = annotate(load_analyzed(cfg, "player"))
    me = me_all[me_all.mover_rating >= cmp_min].reset_index(drop=True)
    cats, cats_all = categories(me), categories(me_all)
    n_games = me.game_id.nunique()
    inflated, why_infl = never_resigns(cfg)

    # target band: the first band above your all-moves level (loss metric)
    target, lv = {}, {}
    if mb is not None:
        loss = mb[mb.metric == "loss per 100 moves"]
        allm = loss[loss.category == "all moves"]
        my_level = float(allm.level.iloc[0]) if len(allm) and np.isfinite(allm.level.iloc[0]) else np.nan
        rcols = sorted(c for c in mb.columns if c.startswith("rating_"))
        ratings = np.array([float(loss[c].iloc[0]) for c in rcols]) if rcols and len(loss) else np.array([])
        above = [i for i, r in enumerate(ratings) if np.isfinite(my_level) and r > my_level + 50]
        t = above[0] if above else (len(ratings) - 1 if len(ratings) else None)
        tgt_rating = ratings[t] if t is not None else np.nan
        for r in loss.itertuples():
            if t is not None:
                target[r.category] = getattr(r, f"band_{t}")
        err = mb[mb.metric == "error rate"]
        base = err[err.category == "all moves"].level
        base = float(base.iloc[0]) if len(base) else np.nan
        for r in err.itertuples():
            lv[r.category] = (r.level, r.level - base if np.isfinite(base) and np.isfinite(r.level) else np.nan)
    else:
        tgt_rating = np.nan

    items = []
    for r in skill.itertuples():
        if r.category == "all moves" or not np.isfinite(r.ratio) or r.category not in cats:
            continue
        m = cats[r.category]
        eng = "worse" if r.ratio_lo > 1 else "better" if r.ratio_hi < 1 else "par"
        maia = "lower" if r.hi < 0 else "higher" if r.lo > 0 else "par"
        level, rel_mb = lv.get(r.category, (np.nan, np.nan))
        infl = inflated and any(k in r.category for k in INFLATABLE)
        if eng == "worse":
            tag = "solid" if maia == "lower" or (np.isfinite(rel_mb) and rel_mb <= -100) else "lead"
        elif eng == "better":
            tag = "info" if infl else ("strength" if maia == "higher" or (np.isfinite(rel_mb) and rel_mb >= 100) else None)
        else:
            tag = "style" if maia != "par" and abs(r.relative) >= 60 else None
        if not tag:
            continue
        you_loss = float(me.wp_loss[m].mean()) if m.any() else np.nan
        per_game = m.sum() / max(n_games, 1)
        gain = None
        if r.category in target and np.isfinite(you_loss) and tag in ("solid", "lead"):
            gain = elo_gain((you_loss - target[r.category]) * per_game)
        claim = (f"Your error rate here is <b>{r.err_you:.0%}</b> vs {r.err_peers:.0%} for peers ({r.ratio:.2f}×). "
                 f"Maia rates your choices {r.relative:+.0f} vs your overall level"
                 + (f"; on the band scale this plays like ~{level:.0f} ({rel_mb:+.0f} vs your overall)" if np.isfinite(level) else "")
                 + f". About {per_game:.1f} of your moves per game fall here.")
        if tag == "style":
            claim += " The engine sees no extra errors: style, not a leak."
        if infl:
            claim += f" Caution: {why_infl}, so lost positions rarely count as errors."
        ev = {"header": ["measure", "you", "reference", "verdict"],
              "rows": [["engine error rate", f"{r.err_you:.1%}", f"{r.err_peers:.1%} (peers)", eng],
                       ["Maia level vs your overall", f"{r.relative:+.0f} [{r.lo:+.0f}, {r.hi:+.0f}]", "0", maia],
                       ["loss per move", f"{100 * you_loss:.2f}",
                        f"{100 * target[r.category]:.2f} (band ~{tgt_rating:.0f})" if r.category in target else "–", ""]]}
        exs, drill = [], None
        if tag in ("solid", "lead"):
            sub = me[m & me.err.to_numpy()].sort_values(["critical", "wp_loss"], ascending=False).head(3)
            exs = [example(x.fen, x.best_line, x.played_line, f"{x.date} · you lost {100 * x.wp_loss:.0f} points",
                           x.pos_id)
                   for x in sub.itertuples()]
            ma = cats_all[r.category]
            cand = me_all[ma & me_all.err.to_numpy() & me_all.critical.astype(bool).to_numpy()]
            cand = cand.assign(rec=cand.mover_rating >= cmp_min).sort_values(["rec", "wp_loss"], ascending=False)
            drill = {"theme": r.category, "pos_ids": cand.pos_id.head(args.max_drill).tolist()}
        items.append(finding(f"cat:{r.category}", section_of(r.category), r.category, claim, tag, gain,
                             f"vs the ~{tgt_rating:.0f} band's loss per move × {per_game:.1f} moves/game" if gain else "",
                             ev, exs, drill,
                             "Drill your own positions (Drills tab)." if drill else
                             ("Nothing to fix." if tag == "style" else ""), "player_report.html"))

    # converting winning endgames, by result
    pop = annotate(load_analyzed(cfg, "population"))
    def conv(d, res):
        d = d.assign(res=res)
        d = d[(d.ph == "endgame") & (d.wp_best > 0.8) & d.res.notna()].sort_values("ply")
        return d.groupby("game_id").res.first().to_numpy(float)
    cm, cp = conv(me, mover_results(cfg, me, "player")), conv(pop, mover_results(cfg, pop, "population"))
    if len(cm) >= 10 and len(cp) >= 10:
        lo, hi = boot_ci(cm)
        worse = hi < cp.mean()
        tag = "strength" if lo > cp.mean() else ("solid" if worse else "info")
        share = len(cm) / max(n_games, 1)
        gain = elo_gain((cp.mean() - cm.mean()) * share) if worse else None
        items.append(finding("endgame:conversion", "endgame", "Converting winning endgames",
                             f"From the first winning endgame position (> 80%) you scored <b>{cm.mean():.2f}</b> "
                             f"[{lo:.2f}, {hi:.2f}] (n {len(cm)}) vs peers {cp.mean():.2f}.", tag, gain,
                             "score gap × share of games reaching a winning endgame" if gain else "",
                             link="endgame_report.html"))
    write(cfg, "categories", items)
    print(f"{len(items)} category findings ({sum(f['tag'] in ('solid', 'lead') for f in items)} weaknesses); "
          f"target band ~{tgt_rating:.0f}; ELO_PER_SCORE {ELO_PER_SCORE:.0f}")
    for f in sorted(items, key=lambda f: -(f["elo"] or 0))[:8]:
        print(f"   {f['tag']:8s} {f['elo'] or 0:6.1f} Elo  {f['title']}")


if __name__ == "__main__":
    main()
