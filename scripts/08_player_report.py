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
Output: findings/tactics.json (tactics findings + the Middlegame tab's "In depth" tables and boards)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coach.findings import detail, elo_gain, example, finding, table, write as write_findings
from coach.config import data_dir, load_config
from coach.difficulty import p_best_own, player_maia_elo
from coach.ideas import IdeaModel, move_tracks, shares, track, track_cfg
from coach.motifs import tactic_labels
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
    """Whole history, you vs yourself, per quarter. Returns (in-depth items, console lines)."""
    when = pd.to_datetime(me_all.date.astype(str).str.replace(".", "-", regex=False), errors="coerce")
    if when.notna().mean() < 0.5:
        return [], ["(progress skipped: games have no dates)"]
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
        cells += [f"{pt[i]:.2f} [{lo[i]:.2f}, {hi[i]:.2f}]" for i in range(3)]
        line = (f"{q}: {sub.game_id.nunique():3d} games, chess.com {r:.0f} (Maia {maia_elo:.0f}), loss/100 "
                f"{pt[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}], blunders {pt[1]:.2f}, positional {pt[2]:.2f}")
        if have_d:
            hard = pt[4] / max(pt[3], 1e-9)
            easy = pt[6] / max(pt[5], 1e-9)
            hard_ci = ci(bt[:, 4] / np.maximum(bt[:, 3], 1e-9))
            easy_ci = ci(bt[:, 6] / np.maximum(bt[:, 5], 1e-9))
            gap = (pt[7] - pt[8]) / 100
            gap_ci = ci((bt[:, 7] - bt[:, 8]) / 100)
            cells += [f"{hard:.1%} [{hard_ci[0]:.1%}, {hard_ci[1]:.1%}]",
                      f"{easy:.1%} [{easy_ci[0]:.1%}, {easy_ci[1]:.1%}]",
                      f"{gap:+.1%} [{gap_ci[0]:+.1%}, {gap_ci[1]:+.1%}]"]
            line += f", errors hard {hard:.1%} / easy {easy:.1%}, best vs Maia {gap:+.1%}"
        rows.append(cells)
        lines.append(line)
    if not rows:
        return [], ["(progress skipped: no quarter with enough games)"]
    head = ["quarter", "games", "chess.com rating", "Maia level", "loss / 100 moves",
            "blunders / 100", "positional / 100"]
    if have_d:
        head += ["error rate, hard positions", "error rate, easy positions", "best move vs Maia-expected"]
    note = ("Your whole history, compared only with yourself. Loss = win probability lost per 100 moves. Maia "
            "level = your rating then, on Maia's scale. Hard / easy = Maia at your level finds the best move < 40% / "
            "≥ 60% of the time; error = mistake or blunder. Brackets are 90% intervals over games.")
    return [detail("progress", "middlegame", "Your progress over time (all phases)", note, table(head, rows))], lines


_FINDINGS: list = []      # clear weaknesses for the report (coach/findings.py), filled as tables are built


def compare_categories(me, pop, cat_me, cat_pop, cats, B, rcfg, title):
    """Loss per 100 moves per category, you vs peers, with the usual verdict rules.
    cat_*: category label per row ('' = not counted). Returns (table, console lines)."""
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
        if v == "clear" and r > 1:
            mpg = len(me) / max(me.game_id.nunique(), 1)
            _FINDINGS.append(finding(f"anatomy:{title}:{c}", "middlegame", f"{title}: {c}",
                                     f"{v_me:.2f} win-prob points lost per 100 moves vs {v_pop:.2f} for peers ({r:.2f}×).",
                                     "solid", elo_gain((v_me - v_pop) / 100 * mpg), "extra loss per move × moves per game"))
        rows.append([c, f"{v_me:.2f} [{lo[j]:.2f}, {hi[j]:.2f}]", f"{v_pop:.2f}", f"{r:.2f}×",
                     verdict_word(v, r), f"{n_me} (exp. {exp_n:.0f})", n_pop])
        lines.append(f"  {c:34s} {v_me:5.2f} [{lo[j]:.2f}, {hi[j]:.2f}] vs {v_pop:5.2f}  {r:.2f}x  {v:12s} "
                     f"n {n_me} (exp {exp_n:.0f}) / {n_pop}")
    return table(["", "you, per 100 moves (90% CI)", "peers", "ratio", "vs peers", "your n", "peer n"], rows), lines


def verdict_word(v: str, r: float) -> str:
    """'clear' -> worse / better (coloured by the report's table), else the verdict as is."""
    return ("worse" if r > 1 else "better") if v == "clear" else v


def anatomy_section(cfg, rcfg, pop, me, me_all, B):
    """Error anatomy (09c): depth of the punishment, kind of reply, was it on a human's radar,
    and (you vs yourself) think time. Returns (in-depth items, console lines)."""
    from coach.anatomy import depth_bucket
    paths = {n: data_dir(cfg, n) / "anatomy.parquet" for n in ("player", "population")}
    if not all(p.exists() for p in paths.values()):
        return [], ["(anatomy skipped: run 09c)"]
    an_me = pd.read_parquet(paths["player"]).drop_duplicates("pos_id")
    an_pop = pd.read_parquet(paths["population"]).drop_duplicates("pos_id")
    A_me = me[["pos_id"]].merge(an_me, on="pos_id", how="left")
    A_pop = pop[["pos_id"]].merge(an_pop, on="pos_id", how="left")
    items, lines = [], []

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
        items.append(detail(f"anatomy:{kind}", "middlegame", "Error anatomy. " + title,
                            "Win probability lost per 100 moves, you vs peers. Errors = mistakes and blunders. Ply 1 = "
                            "your move, 2 = their reply, and so on; a loss counts once material is ≥ 2 pawns down and "
                            "stays there to the end of the settled line. Maia's probability of the reply is computed "
                            "after your move, at your level.", t))
        lines += l

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
            rows.append([nm, f"{m.sum() / ok.sum():.0%}", f"{er:.1%} [{e_lo:.1%}, {e_hi:.1%}]",
                         f"{br:.1%} [{b_lo:.1%}, {b_hi:.1%}]", f"{share_loss:.0%}"])
            lines.append(f"  {nm:8s} moves {m.sum() / ok.sum():4.0%}  error rate {er:5.1%} [{e_lo:.1%}, {e_hi:.1%}]  "
                         f"blunder rate {br:5.1%} [{b_lo:.1%}, {b_hi:.1%}]  share of error loss {share_loss:4.0%}")
        items.append(detail("anatomy:think", "middlegame", "Think time and errors (all phases, whole history)",
                            "Fast errors suggest the move wasn't searched; slow ones that it was searched and misjudged. "
                            "Peers' think times aren't available (their positions are sampled).",
                            table(["time spent on the move", "share of moves", "error rate (90% CI)", "blunder rate",
                                   "share of your error loss"], rows)))
    return items, lines


DIFF_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def maia_section(cfg, rcfg, pop, me, tr_pop, tr_me, tl_pop, tl_me, B):
    """Difficulty via Maia (09b): returns (in-depth items, console lines), empty if 09b hasn't run.

    d = P(engine best move) for a player at the mover's own Maia-scale Elo (peers: Lichess
    rating; you: chess.com rating then + maia.player_offset)."""
    dpop, dme = p_best_own(cfg, pop, "population"), p_best_own(cfg, me, "player")
    if dpop is None or dme is None or np.isnan(dme).mean() > 0.1 or np.isnan(dpop).mean() > 0.1:
        return [], ["(Maia section skipped: run 09b for both sets)"]
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
    t1 = table(["", "found the engine's best move", "Maia at the same level expects", "difference"],
               [["you", f"{act_me:.1%}", f"{exp_me:.1%}", f"{act_me - exp_me:+.1%} [{gap_lo:+.1%}, {gap_hi:+.1%}]"],
                ["peers", f"{act_pop:.1%}", f"{exp_pop:.1%}", f"{act_pop - exp_pop:+.1%}"]])
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
            v = "too few"
        elif rate_lo > r_pop and r_me / max(r_pop, 1e-9) >= min_ratio:
            v = "worse"
        elif rate_hi < r_pop and r_me / max(r_pop, 1e-9) <= 1 / min_ratio:
            v = "better"
        else:
            v = "within noise"
        rows2.append([labels[-1], f"{in_me.mean():.0%} / {in_pop.mean():.0%}", f"{r_me:.1%} [{rate_lo:.1%}, {rate_hi:.1%}]",
                      f"{r_pop:.1%}", f"{r_me / max(r_pop, 1e-9):.2f}×", v, f"{lm:.2f} [{l_lo:.2f}, {l_hi:.2f}]",
                      f"{lp:.2f}"])
        lines.append(f"  Maia finds best {labels[-1]:>9s}: share {in_me.mean():4.0%}/{in_pop.mean():4.0%}  "
                     f"error rate {r_me:5.1%} [{rate_lo:.1%}, {rate_hi:.1%}] vs {r_pop:5.1%}  "
                     f"loss/100 {lm:.2f} vs {lp:.2f}  {v}")
    t2 = table(["Maia at your level finds the best move", "share of moves (you / peers)", "your error rate (90% CI)",
                "peers", "ratio", "vs peers", "your loss / 100 moves", "peers"], rows2)

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
        rows3.append([cats[j], f"{v_me[j]:.2f} [{lo3[j]:.2f}, {hi3[j]:.2f}]", f"{v_pop[j]:.2f}", f"{r:.2f}×",
                      verdict_word(v, r), f"{n_me[j]} (exp. {expected[j]:.0f})", n_pop[j]])
        lines.append(f"  {cats[j]:34s} {v_me[j]:5.2f} [{lo3[j]:.2f}, {hi3[j]:.2f}] vs {v_pop[j]:5.2f}  "
                     f"{r:.2f}x  {v:12s} n {n_me[j]} (exp {expected[j]:.0f}) / {n_pop[j]}")
    t3 = table(["surprising miss", "you, per 100 moves (90% CI)", "peers", "ratio", "vs peers", "your n", "peer n"],
               rows3)
    ids = np.flatnonzero(s_me)
    ids = ids[np.argsort(-loss_me[ids])][: rcfg["player_examples"]]
    exs = [example(me.fen.iat[i], me.best_line.iat[i], me.played_line.iat[i],
                   f"{me.date.iat[i]} · Maia at your level finds the best move {dme[i]:.0%} of the time · lost "
                   f"{100 * loss_me[i]:.0f} points", me.pos_id.iat[i]) for i in ids]
    off = mcfg.get("player_offset")
    items = [
        detail("maia:found", "middlegame", "How often you find the engine's move, vs Maia at your level",
               "Each peer is measured against Maia at their own Lichess rating; you at your chess.com rating then "
               f"+ {off}.", t1),
        detail("maia:difficulty", "middlegame", "Your errors by how hard the position was (Maia)",
               "Error = mistake or blunder; difficulty = how often Maia at your level finds the best move.", t2),
        detail("maia:surprising", "middlegame", "Surprising misses: errors in positions players at your level solve",
               f"A mistake or blunder where Maia at your level finds the engine's move at least {thr:.0%} of the time. "
               f"Total: you {tot_me:.2f} vs peers {tot_pop:.2f} win-prob points per 100 moves. Boards: your worst ones.",
               t3, exs)]
    return items, lines


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

    def idea_rows(ks):
        return [[f"{model.name(k)} ({', '.join(f'{a}{n}' for a, n, _ in model.describe(k, 2))})",
                 f"{rel_me[k]:.1%} [{rel_lo[k]:.1%}, {rel_hi[k]:.1%}]", f"{rel_pop[k]:.1%}", f"{ratio[k]:.2f}×",
                 {"weak": "worse", "strong": "better"}.get(is_clear(k), "within noise"),
                 f"{leak_me[k]:.2f} [{lo[k]:.2f}, {hi[k]:.2f}]", f"{leak_pop[k]:.2f}",
                 f"{rate_me[k]:.0%} (n={n_me[k]})", f"{rate_pop[k]:.0%}"] for k in ks]

    # Your expected count at the peer rate: below min_expected, a bootstrap over a handful of
    # events gives a falsely narrow interval (0 events -> [0, 0]), so no verdict.
    expected = n_cat_pop * len(me) / max(len(pop), 1)
    min_exp = rcfg.get("min_expected", 5)

    def tactic_verdict(j):
        if expected[j] < min_exp:
            return "too few"
        if t_lo[j] > t_pop[j] and t_ratio[j] >= min_ratio:
            return "worse"
        if t_hi[j] < t_pop[j] and t_ratio[j] <= 1 / min_ratio:
            return "better"
        return "within noise"

    shown = [j for j in np.argsort(-(t_me - t_pop)) if n_cat_pop[j] >= min_n]

    def tactic_examples():
        exs = []
        for j in [j for j in shown if t_me[j] > t_pop[j]][:2]:
            ids = tl_me.index[tl_me.category == cats[j]]
            for i in me.loc[ids].sort_values("wp_loss", ascending=False).index[:3]:
                r = me.loc[i]
                exs.append(example(r.fen, r.best_line, r.played_line,
                                   f"{r.date} · {cats[j]} · lost {100 * r.wp_loss:.0f} points", r.pos_id))
        return exs

    maia_items, maia_lines = maia_section(cfg, rcfg, pop, me, tr_pop, tr_me, tl_pop, tl_me, B)
    prog_items, prog_lines = progress_section(cfg, rcfg, model, me_all, Dm_all, names, B)
    anat_items, anat_lines = anatomy_section(cfg, rcfg, pop, me, me_all, B)
    calib = (me.wp_loss.mean(), pop.wp_loss.mean())
    scope = (f"{len(me):,} of your moves ({me.game_id.nunique():,} games at chess.com ≥ {cmp_min}) vs {len(pop):,} peer "
             f"moves (Lichess rapid ≈ {model.meta.get('rating_center')}); mean win-prob loss per move: you "
             f"{calib[0]:.2%}, peers {calib[1]:.2%}.")
    details = [
        detail("loss:breakdown", "middlegame", "Where your win probability goes (all phases)",
               scope + f" Decided = the best move's win probability was outside {track_cfg(model, cfg)['contested']}.",
               table(["where the win probability goes (per 100 moves)", "you (90% CI)", "peers", "ratio"],
                     [[TRACK_TEXT[t], f"{bd_me[i]:.2f} [{bd_lo[i]:.2f}, {bd_hi[i]:.2f}]", f"{bd_pop[i]:.2f}",
                       f"{bd_me[i] / max(bd_pop[i], 1e-9):.2f}×"] for i, t in enumerate(TRACKS)])),
        detail("tactics:table", "middlegame", "What your blunders and tactical mistakes are made of",
               "“missed” = the best line won ≥ 2 pawns you didn't take; “allowed” = your move let the opponent win ≥ 2. "
               f"The motif is the first one detected (heuristic). Categories with fewer than {min_n} peer examples are "
               f"hidden; no verdict below {min_exp} expected. Boards: your worst examples of the categories where you "
               "lose more than peers.",
               table(["what happened", "you, per 100 moves (90% CI)", "peers", "ratio", "vs peers", "your n", "peer n"],
                     [[cats[j], f"{t_me[j]:.2f} [{t_lo[j]:.2f}, {t_hi[j]:.2f}]", f"{t_pop[j]:.2f}", f"{t_ratio[j]:.2f}×",
                       tactic_verdict(j), f"{n_cat_me[j]} (exp. {expected[j]:.0f})", n_cat_pop[j]] for j in shown]),
               tactic_examples()),
        detail("tactics:no-swing", "middlegame", "Blunders with no material swing, by phase",
               "No ≥ 2-pawn material swing even after the deep pass: a win let slip, equal to lost, or a large "
               "positional drop.",
               table(["phase", "you, per 100 moves (90% CI)", "peers", "ratio"],
                     [[p_, f"{ph_me[i]:.2f} [{ph_lo[i]:.2f}, {ph_hi[i]:.2f}]", f"{ph_pop[i]:.2f}",
                       f"{ph_me[i] / max(ph_pop[i], 1e-9):.2f}×"] for i, p_ in enumerate(phases)])),
        *maia_items, *anat_items, *prog_items]
    if weak or strong:
        f_ideas = detail("ideas:table", "middlegame", "Positional ideas (learned groups)",
                         "Groups the model learned from how positions change after a mistake (NMF), with their "
                         "strongest features. Share = this idea's part of your positional losses, so it compares your "
                         "profile with peers' independent of strength. These groups are hard to name and are being "
                         "reworked; treat them as hints, not findings.",
                         table(["idea (top features)", "share of your positional loss (90% CI)", "peers", "ratio",
                                "vs peers", "leak / 100 moves", "peers", "critical miss rate", "peers"],
                               idea_rows(weak + strong)))
        f_ideas["tag"] = "exploratory"
        details.append(f_ideas)
    print(f"your positional mistakes encoded: {len(mr):,}; {len(details)} in-depth items")
    print("loss per 100 moves, you vs peers:")
    for i, t in enumerate(TRACKS):
        print(f"  {t:10s} {bd_me[i]:5.2f} [{bd_lo[i]:.2f}, {bd_hi[i]:.2f}]  vs {bd_pop[i]:5.2f}")
    mpg = len(me) / max(me.game_id.nunique(), 1)
    for j in shown:
        if tactic_verdict(j) == "worse":
            _FINDINGS.append(finding(f"tactic:{cats[j]}", "middlegame", f"Tactics: {cats[j]}",
                                     f"{t_me[j]:.2f} win-prob points lost per 100 moves vs {t_pop[j]:.2f} for peers "
                                     f"({t_ratio[j]:.2f}×).", "solid", elo_gain((t_me[j] - t_pop[j]) / 100 * mpg),
                                     "extra loss per move × moves per game", todo="See the examples under In depth."))
    write_findings(cfg, "tactics", _FINDINGS + details)
    print("tactics (loss per 100 moves, you vs peers; peer n >= %d):" % min_n)
    for j in shown:
        v = tactic_verdict(j)
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
    for heading, ks in (("weaknesses", weak), ("strengths", strong)):
        print(f"{heading} (share of positional loss, you vs peers):")
        for k in ks:
            v = "CLEAR" if is_clear(k) else "noise"
            print(f"  {model.name(k):28s} {rel_me[k]:5.1%} [{rel_lo[k]:.1%}, {rel_hi[k]:.1%}] vs "
                  f"{rel_pop[k]:5.1%}  {ratio[k]:.2f}x  {v:5s}  "
                  f"{', '.join(f'{a}{n}' for a, n, _ in model.describe(k, 3))}")


if __name__ == "__main__":
    main()
