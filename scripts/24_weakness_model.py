"""Weakness model: your excess errors per concept vs a peer model at your own rating (docs/weakness_model.md).

  1. peers   every Lichess band with engine results (17's bands + your peer set), positions already lost
             (wp_best < 0.15) dropped; motif tags (coach/motifs.line_motifs) cached per set (<set>/motif_tags.parquet).
  2. stage 1 gradient boosting: P(err | x, rating), E[win-prob loss | x, rating], P(event | err, x).
             Validated on held-out peer games (20%, by game): Brier, AUC, observed/expected per band and
             per concept (should be ~1; printed and saved).
  3. stage 2 your games mapped to the Lichess scale (fitted Maia offset), kept inside the bands' rating
             range. Per concept: observed / expected, relative to your overall ratio, shrunk (empirical
             Bayes across concepts), 90% game-bootstrap CI, and a temporal holdout (older 70% of games
             find it, newest 30% must show it too).
  tags       solid = CI above 1 and the newest games agree · lead = CI above 1 only ·
             strength = CI below 1 · diagnostic concepts (all positions per phase) are context only.
  elo        695 x excess win-prob loss per game (shrunk; upper bound); rank = elo x trainability.

    python scripts/24_weakness_model.py --config configs/v0.yaml

Output: player/weakness_model.parquet, player/weakness_validation.json, findings/weakness.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coach.bands import resolve_sets, settings
from coach.config import data_dir, load_config
from coach.findings import ELO_PER_SCORE, example, finding, write
from coach.results import load_analyzed
from coach.weakness import (TRAIN, PeerModel, calibration, concepts, crossfit, eb_alpha, expected, per_game,
                            prepare, shrink)

B = 500
MIN_EXP = 3.0       # expected (relative) count below this: too rare to judge
MIN_PEERS = 2000    # peer positions needed to fit stage 1 (5 folds of gradient boosting)


def load(cfg, which, workers):
    try:
        df = load_analyzed(cfg, which, raw_lines=True, refined=False)   # same depth for everyone
    except FileNotFoundError:
        return None
    return prepare(cfg, df, which, workers)


def stats(df, pred, cs, cal):
    """games x (4 per concept + 4 overall) sums, and game ids."""
    cols = []
    for c in cs:
        cols += list(expected(df, pred, c, cal))
    cols += [df.err.to_numpy(float), pred.p_err.to_numpy(), df.wp_loss.to_numpy(float), pred.l_hat.to_numpy()]
    return per_game(df, cols)


def summarize(S, K, n_games, alpha=None, fit=None):
    """S: (..., 4K+4) sums -> dict of arrays (relative ratio raw/shrunk, elo, E').
    alpha None: fitted on the concepts where `fit` is True (non-diagnostic; they overlap less)."""
    S = np.atleast_2d(S)
    O, E, Lo, Le = (S[:, j:4 * K:4] for j in range(4))
    Oa, Ea, La, Lea = (S[:, 4 * K + j][:, None] for j in range(4))
    th, la = Oa / np.maximum(Ea, 1e-9), La / np.maximum(Lea, 1e-9)
    Er, Ler = th * E, la * Le
    fit = np.ones(K, bool) if fit is None else fit
    a = eb_alpha(O[0][fit], Er[0][fit]) if alpha is None else alpha
    ratio = shrink(O, Er, a)
    w = Er / (Er + a)
    rho_l = 1 + w * (Lo / np.maximum(Ler, 1e-12) - 1)
    elo = ELO_PER_SCORE * Ler * (rho_l - 1) / max(n_games, 1)
    return {"O": O, "Er": Er, "raw": O / np.maximum(Er, 1e-9), "ratio": ratio, "elo": elo,
            "theta": th[:, 0], "alpha": a}


def validate(peers, oof):
    """Out-of-fold quality of stage 1 (before the per-concept calibration)."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    y = peers.err.to_numpy(bool)
    out = {"n_peers": int(len(peers)),
           "brier": float(brier_score_loss(y, oof.p_err)), "brier_base": float(brier_score_loss(y, np.full(len(y), y.mean()))),
           "auc": float(roc_auc_score(y, oof.p_err)), "oe_band": {}, "oe_concept": {}}
    for b, g in peers.groupby("band"):
        out["oe_band"][b] = round(float(g.err.sum() / max(oof.p_err[g.index].sum(), 1e-9)), 3)
    for c in concepts(peers):
        o, e, _, _ = expected(peers, oof, c)
        if e.sum() >= 10:
            out["oe_concept"][c["name"]] = round(float(o.sum() / e.sum()), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-drill", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(0)

    # ---- peers
    sets = resolve_sets(cfg, settings(cfg))
    parts = []
    for name, (lo, hi) in sets:
        d = load(cfg, name, args.workers)
        if d is None or not len(d):
            print(f"  {name}: no engine results, skipped")
            continue
        parts.append(d.assign(band=name))
        print(f"  {name}: {len(d):,} positions, {d.game_id.nunique():,} games, error rate {d.err.mean():.1%}")
    if not parts:
        sys.exit("no peer positions with engine results")
    peers = pd.concat(parts, ignore_index=True)
    if len(peers) < MIN_PEERS:
        write(cfg, "weakness", [], expand=False)
        print(f"only {len(peers):,} peer positions (< {MIN_PEERS:,}): too few to fit the peer model; no findings")
        return
    r_lo, r_hi = min(lo for _, (lo, _h) in sets), max(hi for _, (_l, hi) in sets)

    # ---- stage 1: 5-fold cross-fit on peer games (quality + per-concept calibration), then fit on all
    oof = crossfit(peers)
    val = validate(peers, oof)
    cal = calibration(peers, oof)
    print(f"\nstage 1 (out-of-fold peer games): Brier {val['brier']:.4f} vs {val['brier_base']:.4f} constant, "
          f"AUC {val['auc']:.3f}")
    print("  observed/expected by band: " + ", ".join(f"{k} {v:.2f}" for k, v in val["oe_band"].items()))
    bad = {k: v for k, v in val["oe_concept"].items() if abs(np.log(v)) > np.log(1.25)}
    print(f"  concepts with out-of-fold O/E outside [0.8, 1.25] (corrected by per-concept calibration): "
          f"{len(bad)} of {len(val['oe_concept'])}" + "".join(f"\n     {k}: {v:.2f}" for k, v in bad.items()))
    model = PeerModel().fit(peers)

    # ---- player
    me = load(cfg, "player", args.workers)
    if me is None:
        sys.exit("no player engine results")
    keep = (me.r >= r_lo) & (me.r <= r_hi)
    print(f"\nyou: {keep.sum():,} of {len(me):,} positions inside the peer rating range [{r_lo}, {r_hi}] "
          f"(Lichess scale)")
    me = me[keep].reset_index(drop=True)
    pred = model.predict(me)
    cs = concepts(me)
    K = len(cs)
    gids, M = stats(me, pred, cs, cal)
    G = len(gids)
    pt = summarize(M.sum(0), K, G, fit=np.array([c["kind"] != "diagnostic" for c in cs]))
    alpha = pt["alpha"]
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)])
    bs = summarize(W @ M, K, G, alpha)

    # temporal holdout: older 70% of games vs newest 30%
    gdate = me.groupby("game_id").date.min().reindex(gids).to_numpy()
    order = np.argsort(gdate, kind="stable")
    cut = int(0.7 * G)
    old, new = np.zeros(G, bool), np.zeros(G, bool)
    old[order[:cut]], new[order[cut:]] = True, True
    tr = summarize(M[old].sum(0), K, old.sum(), alpha)
    ts = summarize(M[new].sum(0), K, new.sum(), alpha)

    print(f"\noverall: you make {pt['theta'][0]:.2f}x the errors of a Lichess player at your mapped rating "
          f"in the same positions (90% CI {np.percentile(bs['theta'], 5):.2f}-{np.percentile(bs['theta'], 95):.2f}); "
          f"concepts below are relative to that. Shrinkage alpha {alpha:.0f}"
          + (" (= no detectable spread between concepts beyond noise)." if alpha > 5e4 else "."))
    rows = []
    for k, c in enumerate(cs):
        lo_, hi_ = np.percentile(bs["ratio"][:, k], [5, 95])
        elo_lo, elo_hi = np.percentile(bs["elo"][:, k], [5, 95])
        rows.append({"concept": c["name"], "section": c["section"], "kind": c["kind"],
                     "n_pos": int(c["mask"].sum()), "observed": float(pt["O"][0, k]),
                     "expected": float(pt["Er"][0, k]), "ratio_raw": float(pt["raw"][0, k]),
                     "ratio": float(pt["ratio"][0, k]), "ratio_lo": lo_, "ratio_hi": hi_,
                     "elo": float(pt["elo"][0, k]), "elo_lo": elo_lo, "elo_hi": elo_hi,
                     "per_game": float(pt["O"][0, k] / G),
                     "old_ratio": float(tr["raw"][0, k]), "old_exp": float(tr["Er"][0, k]),
                     "new_ratio": float(ts["raw"][0, k]), "new_exp": float(ts["Er"][0, k])})
    res = pd.DataFrame(rows)
    P = data_dir(cfg, "player")
    res.to_parquet(P / "weakness_model.parquet", index=False)
    val.update({"alpha": alpha, "theta": float(pt["theta"][0]), "games": int(G), "positions": int(len(me))})
    judged = res[(res.old_exp >= MIN_EXP) & (res.new_exp >= MIN_EXP) & (res.kind != "diagnostic")]
    if len(judged) >= 5:
        val["holdout_corr_log_ratio"] = float(np.corrcoef(np.log(judged.old_ratio.clip(0.05)),
                                                         np.log(judged.new_ratio.clip(0.05)))[0, 1])
        print(f"holdout: correlation of log ratios, older vs newest games, over {len(judged)} concepts: "
              f"{val['holdout_corr_log_ratio']:+.2f} (0 = the concept profile doesn't replicate)")
    (P / "weakness_validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")

    # ---- findings
    items = []
    for k, (c, r) in enumerate(zip(cs, res.itertuples())):
        if r.expected < MIN_EXP:
            continue
        weak, strong = r.ratio_lo > 1, r.ratio_hi < 1
        if not (weak or strong):
            continue
        replic = r.new_exp >= MIN_EXP and r.new_ratio > 1 and r.old_ratio > 1
        diag = c["kind"] == "diagnostic"
        tag = "info" if diag else ("solid" if weak and replic else "lead") if weak else "strength"
        elo = max(r.elo, 0.0) if weak and not diag else None     # more but smaller errors: no Elo cost
        rank = elo * TRAIN[c["kind"]] if elo is not None else None
        title = c["name"].split(": ", 1)[1].capitalize() + f" ({c['section']})"
        claim = (f"You {'made' if c['event'] else 'erred'} <b>{r.observed:.0f}</b> times here vs "
                 f"<b>{r.expected:.1f}</b> expected for a Lichess player at your rating in the same positions, "
                 f"after adjusting for your overall level: <b>{r.ratio:.2f}×</b> [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}] "
                 f"(shrunk). Older games {r.old_ratio:.2f}×, newest 30% {r.new_ratio:.2f}×"
                 + (" — holds up." if replic else " — not confirmed in your newest games." if weak else ".")
                 + (" Context only: too broad to train directly." if diag else ""))
        ev = {"header": ["measure", "value"],
              "rows": [["kind", c["kind"]], ["positions (opportunities)", f"{r.n_pos:,}"],
                       ["observed / expected", f"{r.observed:.0f} / {r.expected:.1f}"],
                       ["ratio (raw → shrunk)", f"{r.ratio_raw:.2f} → {r.ratio:.2f} [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}]"],
                       ["Elo value (90% CI)", f"{r.elo:+.1f} [{r.elo_lo:+.1f}, {r.elo_hi:+.1f}]"],
                       ["older 70% / newest 30% of games", f"{r.old_ratio:.2f} / {r.new_ratio:.2f}"]]}
        exs, drill = [], None
        if weak and not diag:
            hit = c["mask"] & (me[c["event"]].to_numpy(bool) if c["event"] else me.err.to_numpy(bool))
            sub = me[hit].sort_values(["date", "wp_loss"], ascending=False)
            exs = [example(x.fen, x.best_line, x.played_line, f"{x.date} · lost {100 * x.wp_loss:.0f} points",
                           x.pos_id) for x in sub.sort_values("wp_loss", ascending=False).head(3).itertuples()]
            drill = {"theme": title, "pos_ids": sub.pos_id.head(args.max_drill).tolist()}
        items.append(finding(f"wm:{c['name']}", c["section"], title, claim, tag, elo,
                             f"695 × excess win-prob loss per game (shrunk), trainability {TRAIN[c['kind']]:.1f}"
                             if elo is not None else "", ev, exs, drill,
                             "Drill these positions (Drills tab)." if drill else "", None, rank))
    write(cfg, "weakness", items)

    show = res[res.kind != "diagnostic"].assign(rank=lambda d: d.elo * d.kind.map(TRAIN))
    show = show[(show.ratio_lo > 1) & (show.expected >= MIN_EXP)].sort_values("rank", ascending=False)
    print(f"\n{len(items)} findings; weaknesses by Elo x trainability:")
    for r in show.head(12).itertuples():
        print(f"   {r.elo:+6.1f} Elo  {r.ratio:4.2f}x [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}]  "
              f"old {r.old_ratio:4.2f} new {r.new_ratio:4.2f}  {r.concept}  ({r.observed:.0f}/{r.expected:.1f})")


if __name__ == "__main__":
    mp.freeze_support()
    main()
