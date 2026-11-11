"""Project your games onto the learned ideas and compare with the peer population.

Loss breakdown (per 100 moves): blunders, and mistakes split into the same tracks 06 uses
(decided position / tactical / positional).

Per positional idea k:
  leak_k     = 100 / N * sum over your positional mistakes of share_k(x) * winprob_loss(x)
               (win probability lost to idea k per 100 moves; N = all your analyzed moves)
  rel_k      = leak_k / sum_j leak_j: idea k's share of your positional losses. Comparing
               rel_k with the peers' removes overall strength, so it shows your *profile*:
               rel_you > rel_peers = weakness, < = strength.
  miss_k     = in *critical* positions where idea k is what separates the best move from the
               2nd best, how often you didn't play the best move (shrunk toward the peer rate)
Bootstrap intervals resample your games.

    python scripts/05_build_deltas.py --set player     (first)
    python scripts/08_player_report.py
Output: data/player/player_report.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.difficulty import p_best_own, player_maia_elo
from coach.ideas import IdeaModel, move_tracks, shares, track, track_cfg
from coach.motifs import tactic_labels
from coach.report import feats_html, page, position_card
from coach.results import load_deep, load_with_deltas
from coach.stats import ci, game_bootstrap


def mistake_codes(model: IdeaModel, df: pd.DataFrame, D: dict, tr: np.ndarray):
    """Codes for positional mistakes. Returns (rows_index, W, S)."""
    m = tr == "positional"
    W, keep, _ = model.encode((D["dB"] - D["dP"])[m])
    rows = np.flatnonzero(m)[keep]
    return rows, W, shares(W)


def leak(model, df, rows, S) -> np.ndarray:
    contrib = np.zeros(model.k)
    if len(rows):
        contrib = (S * df.wp_loss.to_numpy()[rows, None]).sum(0)
    return 100 * contrib / max(len(df), 1)


TRACKS = ["blunder", "tactical", "positional", "decided", "ok", "unusable"]
TRACK_TEXT = {
    "blunder": "blunders (> 20% win prob)",
    "tactical": "tactical mistakes (material ends ≥ 2 apart)",
    "positional": "positional mistakes (→ ideas below)",
    "decided": "mistakes in already-decided positions",
    "ok": "small inaccuracies (< 3% each)",
    "unusable": "mistakes with unusable lines",
}


def breakdown(df, tr) -> np.ndarray:
    """Win prob lost per 100 moves, per track."""
    loss = df.wp_loss.to_numpy()
    return np.array([100 * loss[tr == t].sum() / max(len(df), 1) for t in TRACKS])


def per100(df, rows: np.ndarray, cats: pd.Series, names: list[str]) -> np.ndarray:
    loss = df.wp_loss.to_numpy()
    return np.array([100 * loss[rows[cats.to_numpy() == c]].sum() / max(len(df), 1) for c in names])




def progress_section(cfg, rcfg, model, me_all, Dm_all, names, B):
    """Whole history, you vs yourself, per quarter. Returns (html, console lines)."""
    when = pd.to_datetime(me_all.date.astype(str).str.replace(".", "-", regex=False), errors="coerce")
    if when.notna().mean() < 0.5:
        return "", ["(progress skipped: games have no dates)"]
    df = me_all.assign(q=when.dt.to_period("Q").astype(str))
    tr = move_tracks(model, cfg, me_all, Dm_all, names)
    d = p_best_own(cfg, me_all, "player")
    have_d = d is not None and np.isfinite(d).mean() > 0.9
    loss = df.wp_loss.to_numpy()
    err = (df.label.isin(["mistake", "blunder"]) & ~df.played_is_best).to_numpy()
    best = df.played_is_best.to_numpy()
    min_games = rcfg.get("progress_min_games", 20)
    rows, lines = [], []
    for q, idx in df.groupby("q").indices.items():
        sub = df.iloc[idx]
        if q == "NaT" or sub.game_id.nunique() < min_games:
            continue
        vals = [loss[idx], loss[idx] * (tr[idx] == "blunder"), loss[idx] * (tr[idx] == "positional")]
        if have_d:
            dd = d[idx]
            vals += [(dd < 0.4), (dd < 0.4) & err[idx], (dd >= 0.6), (dd >= 0.6) & err[idx],
                     best[idx], np.nan_to_num(dd)]
        bt = game_bootstrap(sub, np.arange(len(sub)), np.stack(vals, 1).astype(float), B)
        lo, hi = ci(bt)
        pt = 100 * np.stack(vals, 1).astype(float).sum(0) / len(sub)
        r = sub.mover_rating.median()
        maia_elo = float(player_maia_elo(r, cfg["maia"]))
        cells = [f"{q}", f"{sub.game_id.nunique()}", f"{r:.0f}", f"{maia_elo:.0f}"]
        cells += [f"{pt[i]:.2f} <span class='muted'>[{lo[i]:.2f}, {hi[i]:.2f}]</span>" for i in range(3)]
        line = (f"{q}: {sub.game_id.nunique():3d} games, chess.com {r:.0f} (Maia {maia_elo:.0f}), loss/100 "
                f"{pt[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}], blunders {pt[1]:.2f}, positional {pt[2]:.2f}")
        if have_d:
            hard = pt[4] / max(pt[3], 1e-9)
            easy = pt[6] / max(pt[5], 1e-9)
            hard_ci = ci(bt[:, 4] / np.maximum(bt[:, 3], 1e-9))
            easy_ci = ci(bt[:, 6] / np.maximum(bt[:, 5], 1e-9))
            gap = (pt[7] - pt[8]) / 100
            gap_ci = ci((bt[:, 7] - bt[:, 8]) / 100)
            cells += [f"{hard:.1%} <span class='muted'>[{hard_ci[0]:.1%}, {hard_ci[1]:.1%}]</span>",
                      f"{easy:.1%} <span class='muted'>[{easy_ci[0]:.1%}, {easy_ci[1]:.1%}]</span>",
                      f"{gap:+.1%} <span class='muted'>[{gap_ci[0]:+.1%}, {gap_ci[1]:+.1%}]</span>"]
            line += f", errors hard {hard:.1%} / easy {easy:.1%}, best vs Maia {gap:+.1%}"
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        lines.append(line)
    if not rows:
        return "", ["(progress skipped: no quarter with enough games)"]
    head = ["quarter", "games", "chess.com rating", "Maia level", "loss / 100 moves",
            "blunders / 100", "positional / 100"]
    if have_d:
        head += ["error rate, hard positions", "error rate, easy positions", "best move vs Maia-expected"]
    note = ("<p class='muted'>Your whole history, compared only with yourself. Loss = win probability lost "
            "per 100 moves. Maia level = your rating then mapped onto Maia's scale. Hard / easy = Maia at "
            "your level finds the best move &lt; 40% / ≥ 60% of the time; error = mistake or blunder. "
            "Brackets are 90% intervals over games.</p>")
    return ("<h2>Progress over time</h2><table><tr>" + "".join(f"<th>{h}</th>" for h in head)
            + "</tr>" + "".join(rows) + "</table>" + note), lines


def compare_categories(me, pop, cat_me, cat_pop, cats, B, rcfg, title):
    """Loss per 100 moves per category, you vs peers, with the usual verdict rules.
    cat_*: category label per row ('' = not counted). Returns (html table, console lines)."""
    min_ratio, min_exp = rcfg.get("min_ratio", 1.25), rcfg.get("min_expected", 5)
    lm, lp = me.wp_loss.to_numpy(), pop.wp_loss.to_numpy()
    oh = np.stack([(cat_me == c) * lm for c in cats], 1)
    lo, hi = ci(game_bootstrap(me, np.arange(len(me)), oh, B))
    rows, lines = [], [title + ":"]
    for j, c in enumerate(cats):
        v_me = 100 * lm[cat_me == c].sum() / len(me)
        v_pop = 100 * lp[cat_pop == c].sum() / len(pop)
        n_me, n_pop = int((cat_me == c).sum()), int((cat_pop == c).sum())
        exp_n = n_pop * len(me) / max(len(pop), 1)
        r = v_me / max(v_pop, 1e-9)
        if exp_n < min_exp:
            v = "too few"
        elif (lo[j] > v_pop and r >= min_ratio) or (hi[j] < v_pop and r <= 1 / min_ratio):
            v = "clear"
        else:
            v = "within noise"
        cls = "muted" if v != "clear" else ("down" if r > 1 else "up")
        rows.append(f"<tr><td>{html.escape(c)}</td><td>{v_me:.2f} <span class='muted'>[{lo[j]:.2f}, {hi[j]:.2f}]"
                    f"</span></td><td>{v_pop:.2f}</td><td>{r:.2f}×</td><td><span class='{cls}'>{v}</span></td>"
                    f"<td>{n_me} <span class='muted'>(exp. {exp_n:.0f})</span></td><td>{n_pop}</td></tr>")
        lines.append(f"  {c:34s} {v_me:5.2f} [{lo[j]:.2f}, {hi[j]:.2f}] vs {v_pop:5.2f}  {r:.2f}x  {v:12s} "
                     f"n {n_me} (exp {exp_n:.0f}) / {n_pop}")
    table = (f"<h3>{html.escape(title)}</h3><table><tr><th></th><th>you, per 100 moves (90% CI)</th><th>peers</th>"
             "<th>ratio</th><th>vs peers</th><th>your n</th><th>peer n</th></tr>" + "".join(rows) + "</table>")
    return table, lines


def anatomy_section(cfg, rcfg, pop, me, me_all, B):
    """Error anatomy (09c): depth of the punishment, kind of reply, was it on a human's radar,
    and (you vs yourself) think time. Returns (html, console lines)."""
    from coach.anatomy import depth_bucket
    paths = {n: data_dir(cfg, n) / "anatomy.parquet" for n in ("player", "population")}
    if not all(p.exists() for p in paths.values()):
        return ("<h2>Error anatomy</h2><p class='muted'>Run 09c_error_anatomy.py to get this section.</p>",
                ["(anatomy skipped: run 09c)"])
    an_me = pd.read_parquet(paths["player"]).drop_duplicates("pos_id")
    an_pop = pd.read_parquet(paths["population"]).drop_duplicates("pos_id")
    A_me = me[["pos_id"]].merge(an_me, on="pos_id", how="left")
    A_pop = pop[["pos_id"]].merge(an_pop, on="pos_id", how="left")
    html_parts, lines = ["<h2>Error anatomy: how deep was the punishment, and did a human see it coming?</h2>"], []

    def cats_allowed(A, kind):
        is_err = A.best_kind.notna().to_numpy()
        refuted = is_err & A.refute_ply.notna().to_numpy()
        out = np.full(len(A), "", dtype=object)
        if kind == "depth":
            out[is_err] = [depth_bucket(x) for x in A.refute_ply.to_numpy(float)[is_err]]
        elif kind == "reply":
            out[refuted] = ["punished by a " + k for k in A.reply_kind.to_numpy()[refuted]]
        elif kind == "radar":
            p = A.p_reply.to_numpy(float)
            lab = np.where(p < 0.05, "reply off the radar (Maia < 5%)",
                           np.where(p < 0.20, "reply unlikely (5-20%)", "reply expected (>= 20%)"))
            ok = refuted & np.isfinite(p)
            out[ok] = lab[ok]
        elif kind == "missed":
            g = A.gain_ply.to_numpy(float)
            ok = is_err & np.isfinite(g)
            out[ok] = [("missed win, " + ("1 move (ply 1)" if x <= 1 else "2 moves (ply 2-3)" if x <= 3
                        else "deeper (ply 4+)") + ", starts with a " + k)
                       for x, k in zip(g[ok], A.best_kind.to_numpy()[ok])]
        return out

    specs = [("depth", "Allowed: how many moves until the material went (played line)",
              ["punished next move (ply 2)", "2-move combination (ply 3-4)", "deeper (ply 5+)", "no material swing"]),
             ("reply", "Allowed: what kind of reply punished you", None),
             ("radar", "Allowed: would a player at your level even consider that reply? (Maia)",
              ["reply off the radar (Maia < 5%)", "reply unlikely (5-20%)", "reply expected (>= 20%)"]),
             ("missed", "Missed wins: how deep, and how it starts (best line)", None)]
    for kind, title, cats in specs:
        cm, cp = cats_allowed(A_me, kind), cats_allowed(A_pop, kind)
        if cats is None:
            cats = sorted((set(cm) | set(cp)) - {""})
        t, l = compare_categories(me, pop, cm, cp, cats, B, rcfg, title)
        html_parts.append(t)
        lines += l
    html_parts.append("<p class='muted'>Errors = mistakes and blunders. Ply 1 = your move, 2 = their reply, "
                      "and so on; a loss counts once material is ≥ 2 pawns down and stays there to the end "
                      "of the settled line. Maia's probability of the reply is computed from the position "
                      "after your move, at your level.</p>")

    # think time, you vs yourself, whole history
    T = me_all[["pos_id"]].merge(an_me[["pos_id", "think_s"]], on="pos_id", how="left").think_s.to_numpy(float)
    ok = np.isfinite(T)
    if ok.mean() > 0.5:
        err = (me_all.label.isin(["mistake", "blunder"]) & ~me_all.played_is_best).to_numpy()
        blu = ((me_all.label == "blunder") & ~me_all.played_is_best).to_numpy()
        loss = me_all.wp_loss.to_numpy()
        edges = [0, 10, 30, 90, 1e9]
        names = ["< 10 s", "10-30 s", "30-90 s", "90 s +"]
        rows = []
        lines.append("think time (you, whole history):")
        for a, b, nm in zip(edges[:-1], edges[1:], names):
            m = ok & (T >= a) & (T < b)
            vals = np.stack([m, m & err, m & blu], 1).astype(float)
            bt = game_bootstrap(me_all, np.arange(len(me_all)), vals, B)
            e_lo, e_hi = ci(bt[:, 1] / np.maximum(bt[:, 0], 1e-9))
            b_lo, b_hi = ci(bt[:, 2] / np.maximum(bt[:, 0], 1e-9))
            er, br = (m & err).sum() / max(m.sum(), 1), (m & blu).sum() / max(m.sum(), 1)
            share_loss = loss[m & err].sum() / max(loss[err & ok].sum(), 1e-9)
            rows.append(f"<tr><td>{nm}</td><td>{m.sum() / ok.sum():.0%}</td>"
                        f"<td>{er:.1%} <span class='muted'>[{e_lo:.1%}, {e_hi:.1%}]</span></td>"
                        f"<td>{br:.1%} <span class='muted'>[{b_lo:.1%}, {b_hi:.1%}]</span></td>"
                        f"<td>{share_loss:.0%}</td></tr>")
            lines.append(f"  {nm:8s} moves {m.sum() / ok.sum():4.0%}  error rate {er:5.1%} [{e_lo:.1%}, {e_hi:.1%}]  "
                         f"blunder rate {br:5.1%} [{b_lo:.1%}, {b_hi:.1%}]  share of error loss {share_loss:4.0%}")
        html_parts.append("<h3>Think time (you, whole history)</h3><table><tr><th>time spent on the move</th>"
                          "<th>share of moves</th><th>error rate (90% CI)</th><th>blunder rate</th>"
                          "<th>share of your error loss</th></tr>" + "".join(rows) + "</table>"
                          "<p class='muted'>Fast errors suggest the move wasn't searched; slow ones suggest it "
                          "was searched and misjudged. Peers' think times aren't available (their positions "
                          "are sampled, so consecutive clocks are missing).</p>")
    return "".join(html_parts), lines


DIFF_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def maia_section(cfg, rcfg, pop, me, tr_pop, tr_me, tl_pop, tl_me, links, B):
    """Difficulty via Maia (09b): returns (html, console lines), or ("", []) if 09b hasn't run.

    d = P(engine best move) for a player at the mover's own Maia-scale Elo (peers: Lichess
    rating; you: chess.com rating then + maia.player_offset)."""
    dpop, dme = p_best_own(cfg, pop, "population"), p_best_own(cfg, me, "player")
    if dpop is None or dme is None or np.isnan(dme).mean() > 0.1 or np.isnan(dpop).mean() > 0.1:
        return ("<h2>Difficulty (Maia)</h2><p class='muted'>Run 09b_maia_difficulty.py for both sets "
                "to get this section.</p>", ["(Maia section skipped: run 09b for both sets)"])
    mcfg = cfg["maia"]
    thr = mcfg.get("surprise", 0.5)
    min_ratio, min_exp, min_n = rcfg.get("min_ratio", 1.25), rcfg.get("min_expected", 5), rcfg.get("min_tactic_n", 10)
    lines = []

    def err(df):
        return (df.label.isin(["mistake", "blunder"]) & ~df.played_is_best).to_numpy()

    hit_me, hit_pop = me.played_is_best.to_numpy(), pop.played_is_best.to_numpy()
    ok_me, ok_pop = np.isfinite(dme), np.isfinite(dpop)
    rows_me = np.flatnonzero(ok_me)

    # 1. best-move hit rate vs what Maia expects at the mover's level
    bt = game_bootstrap(me, rows_me, np.stack([hit_me[ok_me], dme[ok_me]], 1).astype(float), B)
    gap_lo, gap_hi = ci((bt[:, 0] - bt[:, 1]) / 100)
    act_me, exp_me = hit_me[ok_me].mean(), dme[ok_me].mean()
    act_pop, exp_pop = hit_pop[ok_pop].mean(), dpop[ok_pop].mean()
    t1 = ("<table><tr><th></th><th>found the engine's best move</th><th>Maia at the same level expects</th>"
          "<th>difference</th></tr>"
          f"<tr><td>you</td><td>{act_me:.1%}</td><td>{exp_me:.1%}</td><td>{act_me - exp_me:+.1%} "
          f"<span class='muted'>[{gap_lo:+.1%}, {gap_hi:+.1%}]</span></td></tr>"
          f"<tr><td>peers</td><td>{act_pop:.1%}</td><td>{exp_pop:.1%}</td><td>{act_pop - exp_pop:+.1%}</td></tr></table>")
    lines.append(f"best move found: you {act_me:.1%} vs Maia-expected {exp_me:.1%} "
                 f"({act_me - exp_me:+.1%} [{gap_lo:+.1%}, {gap_hi:+.1%}]); peers {act_pop:.1%} vs {exp_pop:.1%}")

    # 2. by difficulty band: error rate and loss
    e_me, e_pop = err(me), err(pop)
    loss_me, loss_pop = me.wp_loss.to_numpy(), pop.wp_loss.to_numpy()
    rows2, labels = [], []
    for a, b in zip(DIFF_BINS[:-1], DIFF_BINS[1:]):
        in_me, in_pop = ok_me & (dme >= a) & (dme < b), ok_pop & (dpop >= a) & (dpop < b)
        labels.append(f"{a:.0%}–{min(b, 1):.0%}")
        vals = np.stack([in_me, in_me & e_me, (in_me & e_me) * loss_me], 1).astype(float)
        bb = game_bootstrap(me, np.arange(len(me)), vals, B)
        rate_lo, rate_hi = ci(bb[:, 1] / np.maximum(bb[:, 0], 1e-9))
        l_lo, l_hi = ci(bb[:, 2])
        r_me = (in_me & e_me).sum() / max(in_me.sum(), 1)
        r_pop = (in_pop & e_pop).sum() / max(in_pop.sum(), 1)
        lm = 100 * ((in_me & e_me) * loss_me).sum() / len(me)
        lp = 100 * ((in_pop & e_pop) * loss_pop).sum() / len(pop)
        exp_n = r_pop * in_me.sum()
        if exp_n < min_exp:
            v = "<span class='muted'>too few</span>"
        elif rate_lo > r_pop and r_me / max(r_pop, 1e-9) >= min_ratio:
            v = "<span class='down'>clear</span>"
        elif rate_hi < r_pop and r_me / max(r_pop, 1e-9) <= 1 / min_ratio:
            v = "<span class='up'>clear</span>"
        else:
            v = "<span class='muted'>within noise</span>"
        rows2.append(f"<tr><td>{labels[-1]}</td><td>{in_me.mean():.0%} / {in_pop.mean():.0%}</td>"
                     f"<td>{r_me:.1%} <span class='muted'>[{rate_lo:.1%}, {rate_hi:.1%}]</span></td>"
                     f"<td>{r_pop:.1%}</td><td>{r_me / max(r_pop, 1e-9):.2f}×</td><td>{v}</td>"
                     f"<td>{lm:.2f} <span class='muted'>[{l_lo:.2f}, {l_hi:.2f}]</span></td><td>{lp:.2f}</td></tr>")
        lines.append(f"  Maia finds best {labels[-1]:>9s}: share {in_me.mean():4.0%}/{in_pop.mean():4.0%}  "
                     f"error rate {r_me:5.1%} [{rate_lo:.1%}, {rate_hi:.1%}] vs {r_pop:5.1%}  "
                     f"loss/100 {lm:.2f} vs {lp:.2f}  {('CLEAR' if 'clear' in v else 'few' if 'few' in v else 'noise')}")
    t2 = ("<table><tr><th>Maia at your level finds the best move</th><th>share of moves (you / peers)</th>"
          "<th>your error rate (90% CI)</th><th>peers</th><th>ratio</th><th>vs peers</th>"
          "<th>your loss / 100 moves</th><th>peers</th></tr>" + "".join(rows2) + "</table>"
          "<p class='muted'>Error = mistake or blunder. Each peer is measured against Maia at their own "
          "Lichess rating; you at your chess.com rating at the time + "
          f"{mcfg.get('player_offset')}.</p>")

    # 3. surprising misses: errors where Maia at your level usually finds the best move
    def categories(df, tr, tl):
        cat = np.array([f"{t} mistake" if t in ("positional", "decided") else "" for t in tr], dtype=object)
        if len(tl):
            cat[tl.index.to_numpy()] = tl.category.to_numpy()
        return cat

    c_me, c_pop = categories(me, tr_me, tl_me), categories(pop, tr_pop, tl_pop)
    s_me = e_me & ok_me & (dme >= thr)
    s_pop = e_pop & ok_pop & (dpop >= thr)
    cats = sorted((set(c_pop[s_pop]) | set(c_me[s_me])) - {""})
    n_pop = np.array([(s_pop & (c_pop == c)).sum() for c in cats])
    n_me = np.array([(s_me & (c_me == c)).sum() for c in cats])
    v_me = np.array([100 * loss_me[s_me & (c_me == c)].sum() / len(me) for c in cats])
    v_pop = np.array([100 * loss_pop[s_pop & (c_pop == c)].sum() / len(pop) for c in cats])
    oh = np.stack([(s_me & (c_me == c)) * loss_me for c in cats], 1) if cats else np.zeros((len(me), 0))
    lo3, hi3 = ci(game_bootstrap(me, np.arange(len(me)), oh, B)) if cats else ([], [])
    expected = n_pop * len(me) / max(len(pop), 1)
    order = [j for j in np.argsort(-(v_me - v_pop)) if n_pop[j] >= min_n]
    rows3 = []
    lines.append(f"surprising misses (Maia at your level finds the best move >= {thr:.0%}), loss per 100 moves:")
    tot_me = 100 * loss_me[s_me].sum() / len(me)
    tot_pop = 100 * loss_pop[s_pop].sum() / len(pop)
    lines.append(f"  total {tot_me:.2f} vs {tot_pop:.2f}  (n {s_me.sum()} / {s_pop.sum()})")
    for j in order:
        r = v_me[j] / max(v_pop[j], 1e-9)
        if expected[j] < min_exp:
            v = "too few"
        elif lo3[j] > v_pop[j] and r >= min_ratio:
            v = "clear"
        elif hi3[j] < v_pop[j] and r <= 1 / min_ratio:
            v = "clear"
        else:
            v = "within noise"
        cls = "muted" if v != "clear" else ("down" if r > 1 else "up")
        rows3.append(f"<tr><td>{html.escape(cats[j])}</td><td>{v_me[j]:.2f} <span class='muted'>"
                     f"[{lo3[j]:.2f}, {hi3[j]:.2f}]</span></td><td>{v_pop[j]:.2f}</td><td>{r:.2f}×</td>"
                     f"<td><span class='{cls}'>{v}</span></td><td>{n_me[j]} <span class='muted'>"
                     f"(exp. {expected[j]:.0f})</span></td><td>{n_pop[j]}</td></tr>")
        lines.append(f"  {cats[j]:34s} {v_me[j]:5.2f} [{lo3[j]:.2f}, {hi3[j]:.2f}] vs {v_pop[j]:5.2f}  "
                     f"{r:.2f}x  {v:12s} n {n_me[j]} (exp {expected[j]:.0f}) / {n_pop[j]}")
    t3 = ("<table><tr><th>surprising miss</th><th>you, per 100 moves (90% CI)</th><th>peers</th><th>ratio</th>"
          "<th>vs peers</th><th>your n</th><th>peer n</th></tr>" + "".join(rows3) + "</table>"
          f"<p class='muted'>Surprising miss = a mistake or blunder in a position where Maia at your level "
          f"finds the engine's best move at least {thr:.0%} of the time. Total: you {tot_me:.2f} vs peers "
          f"{tot_pop:.2f} per 100 moves.</p>")

    # your worst surprising misses
    ids = np.flatnonzero(s_me)
    ids = ids[np.argsort(-loss_me[ids])][: rcfg["player_examples"]]
    cards = "".join(position_card(me.iloc[i], link=links.get(me.game_id.iat[i]),
                                  note=f"Maia at your level finds the best move {dme[i]:.0%} of the time")
                    for i in ids)
    body = ("<h2>Difficulty: how your misses compare with what players at your level find (Maia-3)</h2>"
            + t1 + t2 + "<h3>Surprising misses</h3>" + t3
            + (f"<h3>Your worst surprising misses</h3><div class='grid'>{cards}</div>" if cards else ""))
    return body, lines


def critical_stats(model, cfg, df, D, names):
    """Per idea: (n critical positions where it's dominant, misses). Only positions routed
    'positional' when comparing the best line with the 2nd-best line."""
    tr = track(df.wp_best.to_numpy(), D["dB"] - D["dA"], names, track_cfg(model, cfg))
    m = df.critical.to_numpy() & (tr == "positional")
    W, keep, _ = model.encode((D["dB"] - D["dA"])[m])
    rows = np.flatnonzero(m)[keep]
    n = np.zeros(model.k, int)
    miss = np.zeros(model.k, int)
    if len(rows):
        dom = W.argmax(1)
        missed = ~df.played_is_best.to_numpy()[rows]
        np.add.at(n, dom, 1)
        np.add.at(miss, dom, missed.astype(int))
    return n, miss


def chesscom_links(cfg) -> dict[str, str]:
    links = {}
    for p in data_dir(cfg, "player", "archives").glob("*.json"):
        for g in json.loads(p.read_text(encoding="utf-8")).get("games", []):
            gid = g.get("uuid") or g.get("url", "").rstrip("/").split("/")[-1]
            if g.get("url"):
                links[gid] = g["url"]
    return links


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rcfg = cfg["report"]
    model = IdeaModel.load(data_dir(cfg, "ideas") / "model.pkl")

    pop, Dp, names = load_with_deltas(cfg, "population")
    me_all, Dm_all, _ = load_with_deltas(cfg, "player")
    # Comparisons with the peer band only use games at >= min_player_rating; the whole
    # history goes to "progress over time".
    cmp_min = cfg["player"]["min_player_rating"]
    keep = (me_all.mover_rating >= cmp_min).to_numpy()
    me = me_all[keep].reset_index(drop=True)
    Dm = {k: v[keep] for k, v in Dm_all.items()}
    tr_pop, tr_me = move_tracks(model, cfg, pop, Dp, names), move_tracks(model, cfg, me, Dm, names)
    pr, _, pS = mistake_codes(model, pop, Dp, tr_pop)
    mr, mW, mS = mistake_codes(model, me, Dm, tr_me)
    B = rcfg["bootstrap"]

    # Loss breakdown by track
    bd_pop, bd_me = breakdown(pop, tr_pop), breakdown(me, tr_me)
    onehot = np.stack([(tr_me == t) for t in TRACKS], 1) * me.wp_loss.to_numpy()[:, None]
    bd_lo, bd_hi = ci(game_bootstrap(me, np.arange(len(me)), onehot, B))

    # Per idea: absolute leak and relative share of positional losses
    leak_pop = leak(model, pop, pr, pS)
    leak_me = leak(model, me, mr, mS)
    boots = game_bootstrap(me, mr, mS * me.wp_loss.to_numpy()[mr, None], B)
    lo, hi = ci(boots)
    rel_pop = leak_pop / max(leak_pop.sum(), 1e-12)
    rel_me = leak_me / max(leak_me.sum(), 1e-12)
    rel_lo, rel_hi = ci(boots / np.maximum(boots.sum(1, keepdims=True), 1e-12))
    diff = rel_me - rel_pop

    n_pop, miss_pop = critical_stats(model, cfg, pop, Dp, names)
    n_me, miss_me = critical_stats(model, cfg, me, Dm, names)
    rate_pop = (miss_pop + 1) / (n_pop + 2)
    s = rcfg["prior_strength"]
    rate_me = (miss_me + s * rate_pop) / (n_me + s)

    # Tactical / blunder track: direction (missed vs allowed) x motif
    tl_pop = tactic_labels(pop, tr_pop, load_deep(cfg, "population"))
    tl_me = tactic_labels(me, tr_me, load_deep(cfg, "player"))
    cats = sorted(set(tl_pop.category) | set(tl_me.category))
    n_cat_pop = tl_pop.category.value_counts().reindex(cats, fill_value=0).to_numpy()
    n_cat_me = tl_me.category.value_counts().reindex(cats, fill_value=0).to_numpy()
    t_pop = per100(pop, tl_pop.index.to_numpy(), tl_pop.category, cats)
    t_me = per100(me, tl_me.index.to_numpy(), tl_me.category, cats)
    oh = np.stack([(tl_me.category == c).to_numpy() for c in cats], 1) * me.wp_loss.to_numpy()[tl_me.index, None]
    t_lo, t_hi = ci(game_bootstrap(me, tl_me.index.to_numpy(), oh, B))
    t_ratio = t_me / np.maximum(t_pop, 1e-9)
    min_n = rcfg.get("min_tactic_n", 10)   # peer count needed before a category is compared

    # Blunders still without a material swing: by game phase
    phases = ["opening", "middlegame", "endgame"]
    nm_pop = tl_pop[tl_pop.direction == "no_material"]
    nm_me = tl_me[tl_me.direction == "no_material"]
    ph_pop = per100(pop, nm_pop.index.to_numpy(), nm_pop.phase, phases)
    ph_me = per100(me, nm_me.index.to_numpy(), nm_me.phase, phases)
    oh_ph = np.stack([(nm_me.phase == p_).to_numpy() for p_ in phases], 1) * me.wp_loss.to_numpy()[nm_me.index, None]
    ph_lo, ph_hi = ci(game_bootstrap(me, nm_me.index.to_numpy(), oh_ph, B))

    links = chesscom_links(cfg)
    # Stable ideas only, ranked by share ratio you / peers. "clear" needs the peer share
    # outside your 90% interval AND a ratio of at least min_ratio (or at most 1/min_ratio).
    stable = [k for k in range(model.k) if model.is_stable(k)]
    n_show = rcfg["player_top_ideas"]
    min_ratio = rcfg.get("min_ratio", 1.25)
    ratio = rel_me / np.maximum(rel_pop, 1e-12)
    weak = sorted([k for k in stable if diff[k] > 0], key=lambda k: -ratio[k])[:n_show]
    strong = sorted([k for k in stable if diff[k] < 0], key=lambda k: ratio[k])[:n_show]

    def is_clear(k):
        if rel_lo[k] > rel_pop[k] and ratio[k] >= min_ratio:
            return "weak"
        if rel_hi[k] < rel_pop[k] and ratio[k] <= 1 / min_ratio:
            return "strong"
        return ""

    def verdict(k):
        c = is_clear(k)
        if c:
            return f"<span class='{'down' if c == 'weak' else 'up'}'>clear</span>"
        return "<span class='muted'>within noise</span>"

    def title(k):
        return html.escape(model.name(k))

    def name(k):
        return html.escape(", ".join(f"{a}{n}" for a, n, _ in model.describe(k, 2)))

    def stat_table(ks):
        rows = "".join(
            f"<tr><td><a href='#idea-{k}'>{title(k)}</a> <span class='muted'>{name(k)}</span></td>"
            f"<td>{rel_me[k]:.1%} <span class='muted'>[{rel_lo[k]:.1%}, {rel_hi[k]:.1%}]</span></td>"
            f"<td>{rel_pop[k]:.1%}</td><td>{ratio[k]:.2f}×</td><td>{verdict(k)}</td>"
            f"<td>{leak_me[k]:.2f} <span class='muted'>[{lo[k]:.2f}, {hi[k]:.2f}]</span></td>"
            f"<td>{leak_pop[k]:.2f}</td>"
            f"<td>{rate_me[k]:.0%} <span class='muted'>(n={n_me[k]})</span></td><td>{rate_pop[k]:.0%}</td></tr>"
            for k in ks)
        return ("<table><tr><th>idea</th><th>share of your positional loss (90% CI)</th><th>peers</th>"
                "<th>ratio</th><th>vs peers</th><th>leak / 100 moves</th><th>peers</th>"
                "<th>critical miss rate</th><th>peers</th></tr>" + rows + "</table>")

    # Your expected count at the peer rate: below min_expected, a bootstrap over a handful of
    # events gives a falsely narrow interval (0 events -> [0, 0]), so no verdict.
    expected = n_cat_pop * len(me) / max(len(pop), 1)
    min_exp = rcfg.get("min_expected", 5)

    def tactic_verdict(j):
        if expected[j] < min_exp:
            return "<span class='muted'>too few</span>"
        if t_lo[j] > t_pop[j] and t_ratio[j] >= min_ratio:
            return "<span class='down'>clear</span>"
        if t_hi[j] < t_pop[j] and t_ratio[j] <= 1 / min_ratio:
            return "<span class='up'>clear</span>"
        return "<span class='muted'>within noise</span>"

    shown = [j for j in np.argsort(-(t_me - t_pop)) if n_cat_pop[j] >= min_n]

    def tactic_table():
        rows = "".join(
            f"<tr><td>{html.escape(cats[j])}</td><td>{t_me[j]:.2f} <span class='muted'>"
            f"[{t_lo[j]:.2f}, {t_hi[j]:.2f}]</span></td><td>{t_pop[j]:.2f}</td>"
            f"<td>{t_ratio[j]:.2f}×</td><td>{tactic_verdict(j)}</td>"
            f"<td>{n_cat_me[j]} <span class='muted'>(exp. {expected[j]:.0f})</span></td>"
            f"<td>{n_cat_pop[j]}</td></tr>" for j in shown)
        return ("<table><tr><th>what happened</th><th>you, per 100 moves (90% CI)</th><th>peers</th>"
                "<th>ratio</th><th>vs peers</th><th>your n</th><th>peer n</th></tr>" + rows + "</table>"
                f"<p class='muted'>Blunders and tactical mistakes. “missed” = the best line won ≥ 2 pawns of "
                f"material you didn't take; “allowed” = your move let the opponent win ≥ 2. The motif is the "
                f"first one detected in the relevant side's moves (heuristic). Categories with fewer than "
                f"{min_n} peer examples are hidden; no verdict when you'd expect fewer than {min_exp} at the "
                f"peer rate.</p>")

    def phase_table():
        rows = "".join(
            f"<tr><td>{p_}</td><td>{ph_me[i]:.2f} <span class='muted'>[{ph_lo[i]:.2f}, {ph_hi[i]:.2f}]"
            f"</span></td><td>{ph_pop[i]:.2f}</td><td>{ph_me[i] / max(ph_pop[i], 1e-9):.2f}×</td></tr>"
            for i, p_ in enumerate(phases))
        return ("<h3>Blunders with no material swing, by phase</h3><table><tr><th>phase</th>"
                "<th>you, per 100 moves (90% CI)</th><th>peers</th><th>ratio</th></tr>" + rows + "</table>"
                "<p class='muted'>No ≥ 2-pawn material swing even after the deep pass: let a win slip, "
                "equal to lost, or a large positional drop.</p>")

    def tactic_examples():
        weak_t = [j for j in shown if t_me[j] > t_pop[j]][:2]
        out = []
        for j in weak_t:
            ids = tl_me.index[tl_me.category == cats[j]]
            ids = me.loc[ids].sort_values("wp_loss", ascending=False).index[: rcfg["player_examples"]]
            cards = "".join(position_card(me.loc[i], link=links.get(me.game_id.at[i])) for i in ids)
            out.append(f"<h3>{html.escape(cats[j])}: your positions</h3><div class='grid'>{cards}</div>")
        return "".join(out)

    def breakdown_table():
        rows = "".join(
            f"<tr><td>{TRACK_TEXT[t]}</td><td>{bd_me[i]:.2f} <span class='muted'>"
            f"[{bd_lo[i]:.2f}, {bd_hi[i]:.2f}]</span></td><td>{bd_pop[i]:.2f}</td>"
            f"<td>{bd_me[i] / max(bd_pop[i], 1e-9):.2f}×</td></tr>"
            for i, t in enumerate(TRACKS))
        return ("<table><tr><th>where the win probability goes (per 100 moves)</th>"
                "<th>you (90% CI)</th><th>peers</th><th>ratio</th></tr>" + rows + "</table>")

    def idea_section(k):
        own = np.argsort(-(mW[:, k] * mS[:, k]))[: rcfg["player_examples"]]
        own = [i for i in own if mS[i, k] > 0.2]
        cards = []
        for i in own:
            r = me.iloc[mr[i]]
            cards.append(position_card(r, link=links.get(r.game_id)))
        drill_idx = np.argsort(-(pS[:, k]))[: rcfg["drills"]]
        drills = [position_card(pop.iloc[pr[i]], puzzle=True) for i in drill_idx if pS[i, k] > 0.3]
        return (
            f"<h2 id='idea-{k}'>{title(k)}</h2>"
            f"<p class='muted'>You lose {leak_me[k]:.2f}% win probability per 100 moves to this idea "
            f"vs {leak_pop[k]:.2f}% for peers. In critical positions where it decides the best move, "
            f"you miss it {rate_me[k]:.0%} of the time (peers {rate_pop[k]:.0%}, your n = {n_me[k]}).</p>"
            + feats_html(model.describe(k))
            + "<h3>Your games</h3><div class='grid'>" + "".join(cards) + "</div>"
            + "<h3>Practice positions (peers' games)</h3><div class='grid'>" + "".join(drills) + "</div>"
        )

    maia_html, maia_lines = maia_section(cfg, rcfg, pop, me, tr_pop, tr_me, tl_pop, tl_me, links, B)
    prog_html, prog_lines = progress_section(cfg, rcfg, model, me_all, Dm_all, names, B)
    anat_html, anat_lines = anatomy_section(cfg, rcfg, pop, me, me_all, B)
    calib = (me.wp_loss.mean(), pop.wp_loss.mean())
    body = [
        "<h1>Your ideas report</h1>",
        f"<p class='muted'>{len(me):,} of your moves ({me.game_id.nunique():,} games at chess.com ≥ {cmp_min}) compared with "
        f"{len(pop):,} peer moves (Lichess rapid ≈ {model.meta.get('rating_center')}). "
        f"Mean win-prob loss per move: you {calib[0]:.2%}, peers {calib[1]:.2%}. "
        "These are patterns in where your losses concentrate relative to peers, not proven causes.</p>",
        "<h2>Where your losses come from</h2>", breakdown_table(),
        "<p class='muted'>Decided = the best move's win probability was outside "
        f"{track_cfg(model, cfg)['contested']}. Only positional mistakes feed the ideas below.</p>",
        "<h2>Weaknesses: ideas that take a bigger share of your losses than peers'</h2>",
        stat_table(weak) if weak else "<p>None among stable ideas.</p>",
        "<h2>Strengths: ideas where you can trust your instincts</h2>",
        stat_table(strong) if strong else "<p>None among stable ideas.</p>",
        "<p class='muted'>Share = this idea's part of your positional losses, so it compares your "
        "profile with peers' independent of overall strength. “clear” = the peer share is outside "
        f"your 90% interval and the ratio is at least {min_ratio}× (or at most 1/{min_ratio}); "
        "“within noise” rows are leads, not conclusions.</p>",
        "<h2>Tactics: what your blunders and tactical mistakes are made of</h2>", tactic_table(),
        phase_table(),
        tactic_examples(),
        maia_html,
        anat_html,
        prog_html,
        *[idea_section(k) for k in weak],
    ]
    path = data_dir(cfg, "player") / "player_report.html"
    path.write_text(page("Your ideas report", "".join(body)), encoding="utf-8")
    print(f"your positional mistakes encoded: {len(mr):,}; wrote {path}")
    print("loss per 100 moves, you vs peers:")
    for i, t in enumerate(TRACKS):
        print(f"  {t:10s} {bd_me[i]:5.2f} [{bd_lo[i]:.2f}, {bd_hi[i]:.2f}]  vs {bd_pop[i]:5.2f}")
    print("tactics (loss per 100 moves, you vs peers; peer n >= %d):" % min_n)
    for j in shown:
        v = "CLEAR" if "clear" in tactic_verdict(j) else "few" if "too few" in tactic_verdict(j) else "noise"
        print(f"  {cats[j]:34s} {t_me[j]:5.2f} [{t_lo[j]:.2f}, {t_hi[j]:.2f}] vs {t_pop[j]:5.2f}  "
              f"{t_ratio[j]:.2f}x  {v:5s}  n {n_cat_me[j]} (exp {expected[j]:.0f}) / {n_cat_pop[j]}")
    print("no-material blunders by phase (per 100 moves, you vs peers):")
    for i, p_ in enumerate(phases):
        print(f"  {p_:10s} {ph_me[i]:5.2f} [{ph_lo[i]:.2f}, {ph_hi[i]:.2f}] vs {ph_pop[i]:5.2f}")
    print("error anatomy:")
    for line in anat_lines:
        print("  " + line)
    print("progress over time (whole history, you vs yourself):")
    for line in prog_lines:
        print("  " + line)
    print("Maia difficulty:")
    for line in maia_lines:
        print("  " + line)
    for title, ks in (("weaknesses", weak), ("strengths", strong)):
        print(f"{title} (share of positional loss, you vs peers):")
        for k in ks:
            v = "CLEAR" if is_clear(k) else "noise"
            print(f"  {model.name(k):28s} {rel_me[k]:5.1%} [{rel_lo[k]:.1%}, {rel_hi[k]:.1%}] vs "
                  f"{rel_pop[k]:5.1%}  {ratio[k]:.2f}x  {v:5s}  "
                  f"{', '.join(f'{a}{n}' for a, n, _ in model.describe(k, 3))}")


if __name__ == "__main__":
    main()
