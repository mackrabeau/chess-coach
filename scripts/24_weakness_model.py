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

  controls   + Maia-3 difficulty at the mover's own Elo (P(engine best move), policy entropy), cached per set
             in <set>/maia_difficulty.parquet (--no-maia: left out). Lines are the settled ones (03).
  4. line patterns (coach/linepatterns.py; <set>/linepatterns.parquet): opportunity = best line has a named
             pattern, 2nd-best doesn't (gap >= 0.05); peer model P(took | controls, rating, pattern), 5-fold
             cross-fit + per-pattern calibration with a peer-game bootstrap; your declines observed / expected
             relative to your overall decline ratio, shrunk family -> pattern; 90% CI (your games x peer
             calibration draws), MDE, BH q over families, older/newest games, peer rating slope (skill vs style).
             Gate: family finding = CI excludes 1, q < 0.10, both halves agree, label precision >= 85% and (for a
             weakness) a positive rating slope; else a lead. Patterns below 70% precision are exploratory; the
             negative control must come out ~1. --null-sims: simulated null players for the holdout correlation
             (shared peer-calibration noise), the gate's false-positive rate and an injected x1.5 effect.

    python scripts/24_weakness_model.py --config configs/v0.yaml

Output: player/weakness_model.parquet, player/weakness_validation.json, player/linepatterns_uptake.parquet,
player/linepatterns_validation.json, findings/weakness.json
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
from coach.findings import example, finding, write
from coach.linepatterns import (CLAIM_PRECISION, CONTROL, FAMILIES, LEAD_PRECISION, PATTERNS, PRECISION,
                                pattern_table)
from coach.weakness import (TRAIN, PeerModel, UptakeModel, calibration, concepts, crossfit, excess, expected,
                            load_set, null_checks, opportunities, rating_slopes, up_calibration, up_crossfit,
                            uptake_gate)

B = 500
MIN_EXP = 3.0       # expected (relative) count below this: too rare to judge
MIN_PEERS = 2000    # peer positions needed to fit stage 1 (5 folds of gradient boosting)
Q_MAX = 0.10        # Benjamini-Hochberg level for line-pattern family claims


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


# ------------------------------------------------------------------ line patterns (coach/linepatterns.py)
def gate_labels(res, slopes, info):
    """label per row: finding (weakness / strength) | lead | noise | exploratory | control.
    Tests use the RAW ratio's 90% CI (the shrunk CI is degenerate when alpha hits the grid top; shrunk ratios
    give the size and the Elo). Family finding = raw CI excludes 1 AND BH q < Q_MAX AND older / newest games agree AND precision
    >= CLAIM_PRECISION AND (for a weakness) peer uptake rises with rating (skill; else style, no Elo).
    A family missing any of these but with a CI excluding 1 is a lead; patterns are at most leads."""
    labels, notes = [], []
    for r in res.itertuples():
        weak, strong = r.raw_lo > 1, r.raw_hi < 1
        prec = PRECISION.get(r.pattern) if r.level == "pattern" else r.precision
        sl = slopes.get(r.family if r.level == "family" else r.pattern, (np.nan,) * 4)
        why = []
        if r.pattern == CONTROL:              # outside theta's set: judged on the absolute scale (raw x theta)
            lo, hi = r.raw_lo * info["theta"], r.raw_hi * info["theta"]
            labels.append("control FAILED" if (lo > 1 or hi < 1) else "control")
            notes.append(f"negative control, absolute ratio {r.ratio_raw * info['theta']:.2f} [{lo:.2f}, {hi:.2f}] "
                         f"must include 1")
            continue
        if r.level == "pattern" and (prec is None or prec < LEAD_PRECISION):
            labels.append("exploratory")
            notes.append(f"precision {prec:.0%} < {LEAD_PRECISION:.0%}")
            continue
        if not (weak or strong):
            labels.append("noise")
            notes.append(f"CI includes 1; MDE {r.mde:.2f}")
            continue
        up = 1 if weak else -1
        holds = r.old_exp >= MIN_EXP and r.new_exp >= MIN_EXP and np.sign(r.old_ratio - 1) == up \
            and np.sign(r.new_ratio - 1) == up
        if not (r.q < Q_MAX):
            why.append(f"BH q {r.q:.2f}")
        if not holds:
            why.append("not in both older and newest games")
        if prec < CLAIM_PRECISION:
            why.append(f"precision {prec:.0%}")
        if weak and not (sl[1] > 0):
            why.append("no rating slope (style)")
        if r.level == "pattern":
            why.append("pattern level (n too small for a claim)")
        labels.append(("weakness" if weak else "strength") if not why else "lead")
        notes.append("; ".join(why))
    res["label"], res["note"] = labels, notes
    return res


def line_patterns(cfg, peers, me, args):
    """Uptake of named line patterns vs a peer model; writes player/linepatterns_uptake.parquet +
    linepatterns_validation.json, returns findings."""
    parts = []
    for name, d in peers.groupby("band"):
        parts.append(pattern_table(cfg, name, d, args.workers))
    o_peer = opportunities(peers, pd.concat(parts, ignore_index=True))
    o_me = opportunities(me, pattern_table(cfg, "player", me, args.workers))
    print(f"\nline patterns: {len(o_peer):,} peer opportunities, {len(o_me):,} yours")
    if len(o_peer) < 500 or len(o_me) < 50:
        print("  too few opportunities for the uptake model; skipped")
        return []
    patterns = list(PATTERNS)
    good = [q for q in patterns if q != CONTROL and PRECISION.get(q, 0) >= LEAD_PRECISION]
    families = {f: [q for q in good if PATTERNS[q][0] == f] for f in FAMILIES}
    families = {f: qs for f, qs in families.items() if qs}
    inc = [patterns.index(q) for q in good]
    p_oof = up_crossfit(o_peer, patterns)
    c_point, c_boot = up_calibration(o_peer, p_oof, patterns)
    model = UptakeModel().fit(o_peer, patterns)
    p_me = model.predict(o_me)
    dec = o_peer[~o_peer.took.astype(bool)]
    peer_cost = dec.groupby("pattern").cost.mean().to_dict()
    peer_cost.update({f: dec[dec.pattern.isin(qs)].cost.mean() for f, qs in families.items()})
    res, info = uptake_gate(o_me, p_me, patterns, families, c_point, c_boot, inc, B, peer_cost=peer_cost)
    prec_f = {f: float(np.average([PRECISION[q] for q in qs],
                                  weights=[max(1, (o_me.pattern == q).sum()) for q in qs])) for f, qs in families.items()}
    res["precision"] = [prec_f.get(r.family) if r.level == "family" else PRECISION.get(r.pattern, np.nan)
                        for r in res.itertuples()]
    slopes = rating_slopes(o_peer, {**families, **{q: [q] for q in patterns}}, B=args.slope_boot)
    res["peer_uptake"] = [o_peer[o_peer.pattern.isin(families[r.family] if r.level == "family" else [r.pattern])]
                          .took.mean() for r in res.itertuples()]
    res["peer_n"] = [int(o_peer.pattern.isin(families[r.family] if r.level == "family" else [r.pattern]).sum())
                     for r in res.itertuples()]
    key = [r.family if r.level == "family" else r.pattern for r in res.itertuples()]
    res["slope"], res["slope_lo"], res["slope_hi"] = ([slopes.get(k, (np.nan,) * 4)[j] for k in key] for j in range(3))
    res = gate_labels(res, slopes, info)
    if args.null_sims:
        nc = null_checks(o_me, p_me, patterns, families, c_point, c_boot, inc, info["theta"], S=args.null_sims)
        hn = np.array(nc["holdout_null"])
        info.update({"holdout_null_mean": float(hn.mean()), "holdout_null_p95": float(np.percentile(hn, 95)),
                     "holdout_null_p": float((hn >= info["holdout_corr"]).mean()) if info["holdout_corr"] == info["holdout_corr"] else None,
                     "false_positive_rate": nc["false_positive_rate"], "inject": nc["inject"],
                     "detection": nc["detection"]})
    P = data_dir(cfg, "player")
    res.to_parquet(P / "linepatterns_uptake.parquet", index=False)
    info.update({"precision": PRECISION, "families": families, "peer_opportunities": int(len(o_peer)),
                 "peer_oof_decline_factors": dict(zip(patterns, map(float, c_point)))})
    (P / "linepatterns_validation.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")

    print(f"  overall decline ratio theta {info['theta']:.2f} [{info['theta_lo']:.2f}, {info['theta_hi']:.2f}] "
          f"(you vs peers at your rating on the same opportunities; rows below are relative to it); "
          f"shrinkage alpha family {info['alpha_family']:.0f}, pattern {info['alpha_pattern']:.0f}")
    hc = info.get("holdout_corr")
    if hc == hc and hc is not None:
        print(f"  holdout correlation (older vs newest games, {info['holdout_n']} patterns): {hc:+.2f}"
              + (f"; simulated null mean {info['holdout_null_mean']:+.2f}, 95th pct {info['holdout_null_p95']:+.2f}"
                 if "holdout_null_mean" in info else ""))
    if "detection" in info:
        print(f"  gate false-positive rate per family (null players) {info['false_positive_rate']:.1%}; "
              f"injected x{info['inject']} decline rate found: "
              + ", ".join(f"{f} {v:.0%}" for f, v in info["detection"].items()))
    print(f"  theta older 70% of games {info['theta_old']:.2f}, newest 30% {info['theta_new']:.2f}")
    print(f"  {'':10s}{'family / pattern':24s} {'opp':>5s} {'peer':>6s} {'raw [90% CI]':>18s} {'shrunk':>6s} "
          f"{'MDE':>5s} {'slope':>6s} {'Elo':>6s}  label")
    for r in res.itertuples():
        nm = r.family if r.level == "family" else "  " + r.pattern
        print(f"  {r.level:10s}{nm:24s} {r.n_opp:5d} {r.peer_uptake:6.0%} "
              f"{r.ratio_raw:4.2f} [{r.raw_lo:4.2f}, {r.raw_hi:4.2f}] {r.ratio:6.2f} {r.mde:5.2f} {r.slope:+6.2f} "
              f"{r.elo:+6.1f}  "
              f"{r.label}{' (' + r.note + ')' if r.note else ''}")

    items = []
    for r in res[(res.level == "family") & res.label.isin(["weakness", "strength", "lead"])].itertuples():
        weak = r.ratio > 1
        qs = families[r.family]
        sub = o_me[o_me.pattern.isin(qs)]
        section = sub.ph.mode().iat[0] if len(sub) else "middlegame"
        tag = {"weakness": "solid", "strength": "strength", "lead": "lead"}[r.label]
        elo = max(r.elo, 0.0) if r.label == "weakness" else None
        advice = " ".join(PATTERNS[q][1] for q in qs)
        claim = (f"When the engine's best line {'/'.join(q.replace('_', ' ') for q in qs)}, you "
                 f"{'pass it up more' if weak else 'take it more'} than a Lichess player at your rating in the "
                 f"same spots: <b>{r.ratio_raw:.2f}×</b> the expected declines [{r.raw_lo:.2f}, {r.raw_hi:.2f}] "
                 f"({r.declined:.0f} vs {r.expected:.1f}; raw uptake gap {100 * (r.declined - r.expected) / r.n_opp:+.0f} "
                 f"points, detectable ≥ {100 * r.mde:.0f}; shrunk {r.ratio:.2f}×). Older games {r.old_ratio:.2f}×, newest 30% {r.new_ratio:.2f}×."
                 + (f" Lead only: {r.note}." if r.label == "lead" else ""))
        ev = {"header": ["measure", "value"],
              "rows": [["opportunities (per game)", f"{r.n_opp} ({r.per_game:.2f})"],
                       ["peer uptake / rating slope per 100", f"{r.peer_uptake:.0%} / {r.slope:+.2f} [{r.slope_lo:+.2f}, {r.slope_hi:+.2f}]"],
                       ["declines observed / expected", f"{r.declined:.0f} / {r.expected:.1f}"],
                       ["ratio raw (90% CI) → shrunk, BH q", f"{r.ratio_raw:.2f} [{r.raw_lo:.2f}, {r.raw_hi:.2f}] → {r.ratio:.2f}, q {r.q:.2f}"],
                       ["Elo (upper bound, 90% CI)", f"{r.elo:+.1f} [{r.elo_lo:+.1f}, {r.elo_hi:+.1f}]"],
                       ["label precision by eye", f"{r.precision:.0%}"]]}
        exs, drill = [], None
        cand = sub[~sub.took.astype(bool)].assign(wp_loss=lambda d: d.cost)
        if weak and len(cand):
            try:
                from coach.clusters import pick_puzzles
                pick = pick_puzzles(cand, k=min(args.max_drill, 12))
            except ImportError:
                pick = cand.sort_values("cost", ascending=False).head(12)
            exs = [example(x.fen, x.best_line, x.played_line, f"{x.date} · {x.pattern.replace('_', ' ')} · "
                           f"lost {100 * x.cost:.0f} points", x.pos_id) for x in pick.head(3).itertuples()]
            drill = {"theme": f"Line pattern: {r.family}", "pos_ids": pick.pos_id.tolist()}
        items.append(finding(f"lp:{r.family}", section, f"Line patterns: {r.family}", claim, tag, elo,
                             "695 × opportunities/game × shrunk uptake deficit × mean cost when declined (upper bound)"
                             if elo is not None else "", ev, exs, drill, advice, None,
                             elo * min(1.0, r.per_game * 10) if elo is not None else None))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-drill", type=int, default=64)
    ap.add_argument("--no-maia", action="store_true", help="no Maia difficulty control (no GPU pass)")
    ap.add_argument("--null-sims", type=int, default=200,
                    help="simulated null players for the line-pattern gate checks (0 = skip)")
    ap.add_argument("--slope-boot", type=int, default=200, help="bootstrap draws for peer rating slopes")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # ---- peers
    sets = resolve_sets(cfg, settings(cfg))
    parts = []
    for name, (lo, hi) in sets:
        d = load_set(cfg, name, args.workers, not args.no_maia)
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
    me = load_set(cfg, "player", args.workers, not args.no_maia)
    if me is None:
        sys.exit("no player engine results")
    keep = (me.r >= r_lo) & (me.r <= r_hi)
    print(f"\nyou: {keep.sum():,} of {len(me):,} positions inside the peer rating range [{r_lo}, {r_hi}] "
          f"(Lichess scale)")
    me = me[keep].reset_index(drop=True)
    pred = model.predict(me)
    cs = concepts(me)
    res, info = excess(me, pred, cs, cal, B, MIN_EXP)
    alpha = info["alpha"]
    print(f"\noverall: you make {info['theta']:.2f}x the errors of a Lichess player at your mapped rating "
          f"in the same positions (90% CI {info['theta_lo']:.2f}-{info['theta_hi']:.2f}); "
          f"concepts below are relative to that. Shrinkage alpha {alpha:.0f}"
          + (" (= no detectable spread between concepts beyond noise)." if alpha > 5e4 else "."))
    P = data_dir(cfg, "player")
    res.to_parquet(P / "weakness_model.parquet", index=False)
    val.update({k: info[k] for k in ("alpha", "theta", "games", "positions")})
    if "holdout_corr_log_ratio" in info:
        val["holdout_corr_log_ratio"] = info["holdout_corr_log_ratio"]
        print(f"holdout: correlation of log ratios, older vs newest games, over {info['holdout_n']} concepts: "
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
    items += line_patterns(cfg, peers, me, args)
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
