"""Puzzle sets from your own games (exploratory): which positions to drill, not what to train.

  1. pool     your critical errors over your whole history (mistake or blunder where the best move beat the
              2nd best by >= 0.10: one clear solution) + peers' critical positions from every Lichess band.
  2. embed    Maia-3 hidden states, one fixed Elo (clusters.elo) for both sides; default pool "move" =
              [square mean, best move's from-square, to-square] of block clusters.layer (coach/clusters.py).
  3. sets     k-means on the pooled positions; your errors per cluster = one puzzle set, ranked by
              size x mean win prob lost. Named by the tags that dominate it (phase / swing type / motif /
              kind of best move / queens); coherence = within-set tag agreement vs size-matched random sets.
  4. puzzles  coach.clusters.pick_puzzles: representative (near the centre), recent, costly, no duplicates
              (one per game, one per opening line, embedding distance). "More like this": peer positions in
              the same cluster, picked the same way (cost = how much the only move matters).
  5. output   one "in depth" item per top set (exploratory, no Elo) with boards + drill, and a study PGN per set
              (player/clusters_drills_<nn>_<name>.pgn: your positions, then the Lichess ones).
No weakness or Elo claim: clusters were ruled out as a concept source (docs/future_work.md "Concept groups").

    python scripts/25_position_clusters.py --config configs/v0.yaml [--k 64] [--layer 3] [--pool move]
    python scripts/25_position_clusters.py --config ... --variants   # coherence of embedding variants
    python scripts/25_position_clusters.py --config ... --validate   # the concept-source research check:
        held-out peer log-loss gain of clusters over 24's features vs permuted / readable-feature baselines,
        and your observed / expected per cluster (player/clusters.parquet, clusters_validation.json)

Output: player/clusters_puzzles.parquet, player/clusters_sets.parquet, player/clusters_coherence.json,
        player/clusters_model.pkl, player/clusters_drills_*.pgn, findings/clusters.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.bands import resolve_sets, settings
from coach.clusters import (PositionClusters, coherence, embeddings, name_set, pick_puzzles, puzzle_labels,
                            readable, rule)
from coach.config import data_dir, load_config
from coach.engine import win_prob
from coach.findings import example, finding, table, write
from coach.results import load_analyzed
from coach.study import puzzle_chapter, write_studies
from coach.weakness import PeerModel, _clf, calibration, crossfit, excess, features, load_set

B = 500
MIN_EXP = 3.0
NULL_REPS = 3
MIN_SET = 8          # puzzles (and >= 5 games) for a set to be shown

def fold_of(peers, folds=5, seed=0):
    games = peers.game_id.unique()
    return peers.game_id.map(dict(zip(games, np.random.default_rng(seed).integers(0, folds, len(games))))).to_numpy()


def fold_labels(X, fold, k, dims):
    """Per fold: (train labels, test labels) from clusters fitted on the training games only."""
    out = []
    for f in np.unique(fold):
        te = fold == f
        pc = PositionClusters(k, dims).fit(X[~te])
        out.append((pc.km.labels_, pc.assign(X[te])))
    return out


def permuted(labels, fold, strata, rng):
    """The same fold labels shuffled within strata (band x phase), train and test separately."""
    out = []
    for f, (ltr, lte) in zip(np.unique(fold), labels):
        te = fold == f
        pair = []
        for lab, s in ((ltr, strata[~te]), (lte, strata[te])):
            lab = lab.copy()
            for v in np.unique(s):
                i = np.flatnonzero(s == v)
                lab[i] = lab[rng.permutation(i)]
            pair.append(lab)
        out.append(tuple(pair))
    return out


def oof_p(peers, fold, k, labels=None):
    """Out-of-fold P(err) from 24's features, plus the cluster as a categorical feature if labels given."""
    F, y = features(peers), peers.err.to_numpy(bool)
    cat = pd.CategoricalDtype(range(k))
    p = np.zeros(len(peers))
    for j, f in enumerate(np.unique(fold)):
        te = fold == f
        Ftr, Fte = F[~te].copy(), F[te].copy()
        if labels is not None:
            Ftr["cluster"] = pd.Categorical(labels[j][0], dtype=cat)
            Fte["cluster"] = pd.Categorical(labels[j][1], dtype=cat)
        clf = _clf().set_params(categorical_features="from_dtype")
        p[te] = clf.fit(Ftr, y[~te]).predict_proba(Fte)[:, 1]
    return p


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def gain(peers, base_ll, p, rng):
    """Mean log-loss gain per position vs the base model, 90% game-bootstrap CI, per band."""
    d = base_ll - logloss(peers.err.to_numpy(float), p)
    gids, g = np.unique(peers.game_id.to_numpy(), return_inverse=True)
    s, n = np.bincount(g, d), np.bincount(g)
    bs = [(s[i].sum() / n[i].sum()) for i in (rng.integers(0, len(gids), len(gids)) for _ in range(200))]
    band = pd.Series(d).groupby(peers.band.to_numpy()).mean()
    return {"gain": float(d.mean()), "lo": float(np.percentile(bs, 5)), "hi": float(np.percentile(bs, 95)),
            "by_band": {k: float(v) for k, v in band.items()}}


# ------------------------------------------------------------------ research check (--validate)
def validate(cfg, args, cc):
    """Clusters as a concept source (ruled out 2026-11-12): held-out peer log-loss gain over 24's features
    vs permuted / readable-feature baselines, and your observed / expected per cluster. No findings."""
    K = args.k or 32
    layer = args.layer if args.layer is not None else 3
    layer = None if layer < 0 else layer
    elo = args.elo or int(cc.get("elo", 1900))
    dims = int(cc.get("dims", 32))
    rng = np.random.default_rng(0)

    # ---- peers + embeddings
    sets = resolve_sets(cfg, settings(cfg))
    parts, Xs, maia, label = [], [], None, {}
    for name, (lo, hi) in sets:
        d = load_set(cfg, name, args.workers)
        if d is None or not len(d):
            print(f"  {name}: no engine results, skipped")
            continue
        X, maia = embeddings(cfg, name, d, elo, layer, maia=maia)
        parts.append(d.assign(band=name))
        Xs.append(X)
        label[name] = f"{lo}-{hi}"
        print(f"  {name}: {len(d):,} positions")
    if not parts:
        sys.exit("no peer positions with engine results")
    peers = pd.concat(parts, ignore_index=True)
    Xp = np.concatenate(Xs)
    r_lo, r_hi = min(lo for _, (lo, _h) in sets), max(hi for _, (_l, hi) in sets)
    if len(peers) < 20 * K:
        sys.exit(f"only {len(peers):,} peer positions for {K} clusters: too few")

    # ---- test on held-out peer games
    print(f"\npredictive test (held-out peer games, k={K}, Maia block {layer if layer is not None else 'final'}, "
          f"Elo {elo}):")
    fold = fold_of(peers)
    y = peers.err.to_numpy(float)
    base_p = oof_p(peers, fold, K)
    base_ll = logloss(y, base_p)
    lab_maia = fold_labels(Xp, fold, K, dims)
    R = readable(peers)
    lab_hand = fold_labels(R.to_numpy(), fold, K, dims)
    strata = (peers.band.astype(str) + "|" + peers.ph.astype(str)).to_numpy()
    res = {"base_logloss": float(base_ll.mean()), "n_peers": int(len(peers)), "k": K, "layer": layer, "elo": elo,
           "maia": gain(peers, base_ll, oof_p(peers, fold, K, lab_maia), rng),
           "readable_kmeans": gain(peers, base_ll, oof_p(peers, fold, K, lab_hand), rng),
           "null": [gain(peers, base_ll, oof_p(peers, fold, K, permuted(lab_maia, fold, strata, rng)), rng)["gain"]
                    for _ in range(NULL_REPS)]}
    g = res["maia"]
    same_sign = sum(v > 0 for v in g["by_band"].values())
    # a useless label of this size costs log loss (the null is negative), so beating it only says "not noise";
    # adding information beyond 24's features needs the gain's lower bound above 0 as well
    res["verdict"] = {"beats_null": g["lo"] > max(res["null"]) and g["lo"] > 0,"beats_readable": g["gain"] > res["readable_kmeans"]["gain"],
                      "bands_positive": f"{same_sign} of {len(g['by_band'])}"}
    print(f"  base log loss {res['base_logloss']:.4f} per position (24's features)")
    print(f"  + Maia clusters:    gain {g['gain'] * 1e4:+.1f} e-4 [{g['lo'] * 1e4:+.1f}, {g['hi'] * 1e4:+.1f}], "
          f"positive in {same_sign} of {len(g['by_band'])} bands")
    print(f"  + readable k-means: gain {res['readable_kmeans']['gain'] * 1e4:+.1f} e-4")
    print("  null (Maia clusters permuted within band x phase): "
          + ", ".join(f"{v * 1e4:+.1f}" for v in res["null"]) + " e-4")
    print(f"  verdict: beats null {res['verdict']['beats_null']}, beats readable k-means "
          f"{res['verdict']['beats_readable']}")

    # ---- final clusters on all peers
    pc = PositionClusters(K, dims).fit(Xp)
    lp = pc.km.labels_
    sec = [pd.Series(peers.ph[lp == k]).mode().iat[0] if (lp == k).any() else "middlegame" for k in range(K)]
    names = [f"cluster {k:02d}" for k in range(K)]
    from sklearn.linear_model import LogisticRegression
    rows, gids = [], peers.game_id.to_numpy()
    for k in range(K):
        m = lp == k
        g_ = peers[m]
        text, f1, prec, rec = rule(R, m, gids)
        try:
            lr = LogisticRegression(C=1e4, max_iter=500).fit(g_[["r"]].to_numpy() / 100, g_.err.to_numpy(bool))
            odds = float(np.exp(lr.coef_[0, 0]))
        except ValueError:                 # one class only
            odds = float("nan")
        rows.append({"cluster": names[k], "section": sec[k], "rule": text, "rule_f1": f1, "rule_precision": prec,
                     "rule_recall": rec, "peer_n": int(m.sum()), "peer_err": float(g_.err.mean()),
                     "peer_err_by_band": " · ".join(f"{label[b]} {v:.0%}" for b, v in g_.groupby("band").err.mean().items()),
                     "odds_per_100": odds})
    table_c = pd.DataFrame(rows)

    # ---- you
    me = load_set(cfg, "player", args.workers)
    if me is None:
        sys.exit("no player engine results")
    me = me[(me.r >= r_lo) & (me.r <= r_hi)].reset_index(drop=True)
    Xm, maia = embeddings(cfg, "player", me, elo, layer, maia=maia)
    lm = pc.assign(Xm)
    mk = lambda lab, df: [{"name": names[k], "section": sec[k], "kind": "cluster", "mask": lab == k, "event": None}
                          for k in range(K)]
    oof = crossfit(peers)
    cal = calibration(peers, oof, cs=mk(lp, peers))
    model = PeerModel().fit(peers)
    cs = mk(lm, me)
    you, info = excess(me, model.predict(me), cs, cal, B, MIN_EXP)
    out = table_c.merge(you.drop(columns=["section", "kind"]).rename(columns={"concept": "cluster"}), on="cluster")
    P = data_dir(cfg, "player")
    out.to_parquet(P / "clusters.parquet", index=False)
    res.update({k: v for k, v in info.items()})
    (P / "clusters_validation.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    print(f"\nyou: {info['positions']:,} positions, {info['games']} games; overall {info['theta']:.2f}x; "
          f"shrinkage alpha {info['alpha']:.0f}"
          + (f"; holdout correlation {info['holdout_corr_log_ratio']:+.2f} over {info['holdout_n']} clusters"
             if "holdout_corr_log_ratio" in info else ""))
    print(f"\n{'cluster':10} {'peers':>6} {'err':>5} {'odds/100':>8} {'you (shrunk) [90% CI]':>24} {'old/new':>11}  rule (F1)")
    for r in out.sort_values("ratio", ascending=False).itertuples():
        print(f"{r.cluster:10} {r.peer_n:6d} {r.peer_err:5.0%} {r.odds_per_100:8.2f} "
              f"{r.ratio:8.2f} [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}] ({r.observed:.0f}/{r.expected:.0f}) "
              f"{r.old_ratio:5.2f}/{r.new_ratio:4.2f}  {r.rule} ({r.rule_f1:.2f})")


# ------------------------------------------------------------------ puzzle sets
def settle(cfg, which, d):
    """Settled lines (03) for solutions and tags; the raw 02 lines if the set isn't (fully) settled yet."""
    try:
        s = load_analyzed(cfg, which, refined=False)[["pos_id", "best_line", "played_line"]]
    except (RuntimeError, FileNotFoundError) as e:
        print(f"  {which}: raw lines ({e})")
        return d
    return d.drop(columns=["best_line", "played_line"]).merge(s, on="pos_id", how="inner")


def pools(cfg, workers):
    """Your critical errors (whole history) and every peer set's critical positions (column band, gap)."""
    me = load_set(cfg, "player", workers)
    if me is None:
        sys.exit("no player engine results")
    me = settle(cfg, "player", me[me.err & me.critical.astype(bool)].reset_index(drop=True))
    me = me[me.best_line.map(len) > 0]                       # a puzzle needs its solution
    parts, label = [], {}
    for name, (lo, hi) in resolve_sets(cfg, settings(cfg)):
        d = load_set(cfg, name, workers)
        if d is None or not len(d):
            continue
        d = settle(cfg, name, d[d.critical.astype(bool)].reset_index(drop=True)).assign(band=name)
        d = d[d.best_line.map(len) > 0].reset_index(drop=True)
        d["gap"] = d.wp_best - d.cp_alt.map(win_prob)        # how much the only move matters
        parts.append(d)
        label[name] = f"{lo}-{hi}"
    peers = pd.concat(parts, ignore_index=True) if parts else None
    return me.reset_index(drop=True), peers, label


def embed_all(cfg, me, peers, elo, layer, pool, maia=None):
    Xm, maia = embeddings(cfg, "player", me, elo, layer, maia=maia, pool=pool)
    Xp = np.zeros((len(peers), Xm.shape[1]), np.float32)
    for b in peers.band.unique():
        m = (peers.band == b).to_numpy()
        Xp[m], maia = embeddings(cfg, b, peers[m], elo, layer, maia=maia, pool=pool)
    return Xm, Xp, maia


def san(fen, uci):
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except (ValueError, AssertionError):
        return uci


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"\(\d+%\)", "", name).lower()).strip("_")[:40] or "mixed"


def puzzles(cfg, args, cc):
    K = args.k or int(cc.get("k", 64))
    layer = args.layer if args.layer is not None else int(cc.get("layer", 3))
    pool = args.pool or cc.get("pool", "move")
    elo = args.elo or int(cc.get("elo", 1900))
    dims = int(cc.get("dims", 32))
    n_sets, per_set, more = int(cc.get("sets", 8)), int(cc.get("per_set", 12)), int(cc.get("more", 8))
    P = data_dir(cfg, "player")
    me, peers, label = pools(cfg, args.workers)
    print(f"pool: {len(me):,} critical errors of yours in {me.game_id.nunique()} games; "
          f"{0 if peers is None else len(peers):,} critical peer positions")
    if peers is None or len(me) < 3 * MIN_SET or len(peers) + len(me) < 20 * K:
        write(cfg, "clusters", [], expand=False)
        pd.DataFrame(columns=["pos_id", "source", "set_rank", "pick"]).to_parquet(P / "clusters_puzzles.parquet")
        print("too few positions for puzzle sets")
        return
    Xm, Xp, _ = embed_all(cfg, me, peers, elo, layer if layer >= 0 else None, pool)
    pc = PositionClusters(K, dims).fit(np.concatenate([Xp, Xm]))
    lm, lp = pc.assign(Xm), pc.assign(Xp)
    Zm, Zp = pc.project(Xm), pc.project(Xp)
    lab = puzzle_labels(me)
    T, summ = coherence(lab, lm, MIN_SET)
    coh = T.set_index("set")
    rows = []
    for k in np.unique(lm):
        i = np.flatnonzero(lm == k)
        g = me.iloc[i]
        if len(i) < MIN_SET or g.game_id.nunique() < 5:
            continue
        rows.append({"cluster": int(k), "name": name_set(lab.iloc[i], lab), "section": g.ph.mode().iat[0],
                     "n": len(i), "games": int(g.game_id.nunique()), "lost": float(g.wp_loss.sum()),
                     "mean_lost": float(g.wp_loss.mean()), "peer_n": int((lp == k).sum()),
                     **{f"{x}_{c.replace(' ', '_')}": float(coh.at[k, f"{c}{'' if x == 'agree' else ' ' + x}"])
                        for c in lab.columns for x in ("agree", "null", "p")}})
    S = pd.DataFrame(rows).sort_values("lost", ascending=False).reset_index(drop=True)
    S["rank"] = np.arange(1, len(S) + 1)

    for old in P.glob("clusters_drills_*.pgn"):
        old.unlink()
    items, picked = [], []
    me_boards = set(me.fen.str.split(" ").str[:2].str.join(" "))
    for r in S.head(n_sets).itertuples():
        i = np.flatnonzero(lm == r.cluster)
        centre = pc.km.cluster_centers_[r.cluster]
        mine = pick_puzzles(me.iloc[i], per_set, X=Zm[i], centre=centre)
        j = np.flatnonzero(lp == r.cluster)
        j = j[~peers.fen.iloc[j].str.split(" ").str[:2].str.join(" ").isin(me_boards).to_numpy()]
        theirs = pick_puzzles(peers.iloc[j], more, X=Zp[j], centre=centre, w_recent=0, cost_col="gap")
        picked += [mine.assign(source="you", set_rank=r.rank), theirs.assign(source=theirs.band, set_rank=r.rank)]
        sig = [f"{c} {getattr(r, 'agree_' + c.replace(' ', '_')):.0%} vs {getattr(r, 'null_' + c.replace(' ', '_')):.0%}"
               for c in lab.columns if getattr(r, "p_" + c.replace(" ", "_")) < 0.05]
        fname = f"clusters_drills_{r.rank:02d}_{slug(r.name)}"
        claim = (f"Exploratory. {r.n} of your critical errors ({r.games} games, {100 * r.lost:.0f} win-% points lost "
                 f"in total, {100 * r.mean_lost:.0f} per error) that Maia-3 sees as similar positions. Drill them as "
                 f"one set: {len(mine)} of your positions (typical, recent, costly, no duplicates) and {len(theirs)} "
                 f"like them from Lichess games. Two puzzles of this set share a tag more often than two random ones "
                 f"for: {', '.join(sig) if sig else 'no tag (a mixed set)'}. Which positions to practise, not a "
                 f"weakness claim: it doesn't say you err here more than peers.")
        ev = [[f"you #{z.pick}", str(z.date)[:10], z.ply // 2 + 1, san(z.fen, z.move), san(z.fen, z.best_move),
               f"{100 * z.wp_loss:.0f}"] for z in mine.itertuples()]
        ev += [[f"Lichess {label[z.band]}", "", z.ply // 2 + 1, san(z.fen, z.move) if z.err else "(found it)",
                san(z.fen, z.best_move), f"{100 * z.gap:.0f} (gap)"] for z in theirs.itertuples()]
        exs = [example(z.fen, z.best_line, z.played_line,
                       f"move {z.ply // 2 + 1} · {str(z.date)[:10]} · you lost {100 * z.wp_loss:.0f}", z.pos_id)
               for z in mine.head(3).itertuples()]
        pz = [{"fen": z.fen, "solution": list(z.best_line), "wrong_line": list(z.played_line) if z.err else None,
               "comment": f"Lichess game, players rated {label[z.band]}. "
                          + (f"The player to move went wrong here ({san(z.fen, z.move)}, -{100 * z.wp_loss:.0f})."
                             if z.err else "The player to move found it.")}
              for z in theirs.itertuples()]
        items.append(finding(f"clusters:set{r.rank:02d}", r.section, f"Puzzle set {r.rank}: {r.name}", claim,
                             "exploratory", examples=exs, detail=True, todo=f"Study file: {fname}.pgn (Files tab).",
                             evidence=table(["puzzle", "date", "move", "played", "best", "lost (win-%)"], ev),
                             drill={"theme": f"Puzzle set {r.rank}: {r.name}", "pos_ids": mine.pos_id.tolist(),
                                    "puzzles": pz}))
        ch = [puzzle_chapter(f"Set {r.rank} #{z.pick} {str(z.date)[:10]}", z.fen, list(z.best_line),
                             f"You played {san(z.fen, z.move)} (-{100 * z.wp_loss:.0f}).", wrong_line=list(z.played_line))
              for z in mine.itertuples()]
        ch += [puzzle_chapter(f"Set {r.rank} · Lichess {label[z.band]} #{k + 1}", d["fen"], d["solution"], d["comment"],
                              wrong_line=d["wrong_line"]) for k, (z, d) in enumerate(zip(theirs.itertuples(), pz))]
        write_studies(ch, P / fname)
    write(cfg, "clusters", items)

    keep = ["pos_id", "source", "set_rank", "pick", "typical"]
    pk = pd.concat(picked, ignore_index=True)[keep] if picked else pd.DataFrame(columns=keep)
    allp = me[["pos_id", "game_id", "date", "ph", "wp_loss"]].assign(cluster=lm)
    out = pd.concat([allp.merge(pk[pk.source == "you"].drop(columns="source"), on="pos_id", how="left")
                     .assign(source="you"),
                     pk[pk.source != "you"].merge(peers[["pos_id", "game_id", "ph", "gap"]].assign(cluster=lp),
                                                  on="pos_id")], ignore_index=True)
    out.to_parquet(P / "clusters_puzzles.parquet", index=False)
    S.to_parquet(P / "clusters_sets.parquet", index=False)
    (P / "clusters_coherence.json").write_text(json.dumps(
        {"k": K, "layer": layer, "pool": pool, "elo": elo, "n_puzzles": len(me), "n_peer": len(peers),
         "sets_shown": int(len(S)), "coherence": summ}, indent=1, default=float), encoding="utf-8")
    with open(P / "clusters_model.pkl", "wb") as fh:
        pickle.dump({"clusters": pc, "layer": layer, "elo": elo, "pool": pool, "k": K}, fh)

    print(f"\n{len(S)} puzzle sets (>= {MIN_SET} errors, >= 5 games) from k={K} clusters ({pool} pool, block {layer}); "
          "tag agreement within sets vs size-matched random sets (size-weighted):")
    for c in lab.columns:
        s = summ[c]
        print(f"  {c:10} {s['agreement']:.1%} vs {s['null']:.1%}  sets beating chance (p<0.05): "
              f"{s['sets_beating_null']:.0%}")
    print(f"\n{'#':>3} {'errors':>6} {'games':>5} {'lost':>5} {'peers':>5}  name")
    for r in S.itertuples():
        print(f"{r.rank:3d} {r.n:6d} {r.games:5d} {100 * r.lost:5.0f} {r.peer_n:5d}  {r.section}: {r.name}")


def variants(cfg, args, cc):
    """Coherence of embedding variants (+ a readable-feature k-means baseline) on the same puzzle pool."""
    elo, dims = args.elo or int(cc.get("elo", 1900)), int(cc.get("dims", 32))
    me, peers, _ = pools(cfg, args.workers)
    lab = puzzle_labels(me)
    V, maia = {}, None
    for pool, layer in (("mean", 3), ("mean", None), ("move", 3), ("move", 5), ("move", 7)):
        Xm, Xp, maia = embed_all(cfg, me, peers, elo, layer, pool, maia)
        V[f"{pool} block {'final' if layer is None else layer}"] = (Xm, Xp)
    R = readable(me)
    cols = [c for c in R.columns if not ("available" in c or "best move is" in c or "tactic" in c)]
    V["readable features (no tactic tags)"] = (R[cols].to_numpy(), readable(peers)[cols].to_numpy())
    res = []
    for name, (Xm, Xp) in V.items():
        for K in (32, 64):
            pc = PositionClusters(K, dims).fit(np.concatenate([Xp, Xm]))
            T, s = coherence(lab, pc.assign(Xm), MIN_SET)
            row = {"variant": name, "k": K, "sets": len(T)}
            for c in lab.columns:
                row[f"{c} agree"], row[f"{c} null"] = s[c]["agreement"], s[c]["null"]
                row[f"{c} sets>chance"] = s[c]["sets_beating_null"]
            res.append(row)
    out = pd.DataFrame(res)
    out.to_csv(data_dir(cfg, "player") / "clusters_variants.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="clusters (default clusters.k, 64; --validate 32)")
    ap.add_argument("--layer", type=int, default=None, help="Maia block (0-7 for 23m); -1 = final pooled vector")
    ap.add_argument("--pool", choices=["mean", "move"], default=None, help="embedding pooling (default move)")
    ap.add_argument("--elo", type=int, default=None)
    ap.add_argument("--variants", action="store_true", help="compare embedding variants by set coherence")
    ap.add_argument("--validate", action="store_true", help="the concept-source research check (no findings)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg.get("clusters", {}) or {}
    if args.validate:
        validate(cfg, args, cc)
    elif args.variants:
        variants(cfg, args, cc)
    else:
        puzzles(cfg, args, cc)


if __name__ == "__main__":
    mp.freeze_support()
    main()
