"""Weakness model: one peer model of human errors, the player's excess per concept (docs/weakness_model.md).

Stage 1 (peers, all Lichess bands): P(err | position, rating) and E[win-prob loss | position, rating],
gradient boosting on controls (rating, clock, eval, criticality, material, ...) + concept flags, so a
"fork position" is expected to be harder at every level. For outcome-defined events ("allowed a fork")
a second model gives P(event | err, x); expected events = P(err) x P(event | err).
Stage 2 (player, each game at its own Lichess-scale rating via the fitted Maia offset):
observed vs expected per concept, relative to the player's overall ratio (a chess.com -> Lichess
mismatch or a general level gap cancels), empirical-Bayes shrinkage across concepts, game bootstrap,
and a temporal holdout (older 70% of games vs newest 30%).
"""
from __future__ import annotations

import multiprocessing as mp

import chess
import numpy as np
import pandas as pd

from .categories import annotate
from .config import data_dir
from .difficulty import player_maia_elo
from .motifs import MOTIFS as _ALL, VAL, line_motifs, material

MOTIF_VERSION = 2              # bump when coach/motifs.py or the lines change (2: settled lines): cache rebuilt
MOTIFS = [m for m in _ALL if m != "other"]     # line_motifs never returns "other" (classify() does)
PHASES = ("opening", "middlegame", "endgame")
ETYPES = ("rook endings", "minor-piece endings", "pawn endings", "queen endings")
LOST = 0.15                     # positions already lost (wp_best below this) are left out: resigners vs not
# kind -> trainability weight used for ranking (diagnostic concepts are context, never the headline)
TRAIN = {"tactic missed": 1.0, "tactic allowed": 1.0, "endgame technique": 1.0, "positional": 0.5,
         "diagnostic": 0.0}


# ------------------------------------------------------------------ motif cache
def _tag(row):
    """missed = motifs in the mover's moves of the best line, allowed = in the opponent's replies of the
    played line (coach/motifs.line_motifs: fork / pin / skewer / discovered only when cashed in)."""
    pid, fen, best, played, is_best = row
    b = chess.Board(fen)
    npm = sum(VAL[p.piece_type] for p in b.piece_map().values() if p.piece_type != chess.KING)
    return (pid, line_motifs(fen, best, 0), [] if is_best else line_motifs(fen, played, 1),
            npm, material(b, b.turn), b.is_check())


def motif_table(cfg, which: str, df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """pos_id -> missed, allowed (motif lists), npm, balance, in_check; cached in <set>/motif_tags.parquet."""
    path = data_dir(cfg, which) / "motif_tags.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    if old is not None and (old.get("v") is None or (old.v != MOTIF_VERSION).any()):
        old = None
    have = set() if old is None else set(old.pos_id)
    todo = df[~df.pos_id.isin(have)]
    if len(todo):
        rows = list(zip(todo.pos_id, todo.fen, todo.best_line.map(list), todo.played_line.map(list),
                        todo.played_is_best.astype(bool)))
        print(f"  tagging motifs: {which} {len(rows):,} positions")
        with mp.Pool(workers or max(1, mp.cpu_count() - 2)) as pool:
            res = pool.map(_tag, rows, chunksize=256)
        new = pd.DataFrame(res, columns=["pos_id", "missed", "allowed", "npm", "balance", "in_check"])
        new["v"] = MOTIF_VERSION
        old = new if old is None else pd.concat([old, new], ignore_index=True)
        tmp = path.with_suffix(".tmp")
        old.to_parquet(tmp, index=False)
        tmp.replace(path)
    return df.merge(old.drop(columns="v"), on="pos_id", how="left")


# ------------------------------------------------------------------ data
def maia_table(cfg, which: str, df: pd.DataFrame, use: bool = True) -> pd.DataFrame:
    """pos_id -> maia_pbest (P(engine best move)), maia_entropy at the mover's own Lichess-scale Elo (df.r),
    cached in <set>/maia_difficulty.parquet. Without torch / maia3 (or use=False) missing rows stay NaN
    (gradient boosting handles NaN, so the control just drops out)."""
    path = data_dir(cfg, which) / "maia_difficulty.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["pos_id", "maia_pbest", "maia_entropy"])
    todo = df[~df.pos_id.isin(set(old.pos_id))]
    if len(todo) and use:
        try:
            from .maia import Maia, difficulty
            maia = Maia(cfg["maia"].get("model", "23m"))
        except Exception as e:                       # no torch / maia3 / weights: run without the control
            print(f"  Maia difficulty unavailable ({type(e).__name__}: {e}); control left out")
            maia = None
        if maia is not None:
            print(f"  Maia difficulty: {which} {len(todo):,} positions")
            pb, ent = difficulty(maia, todo.fen.tolist(), todo.history_fens.tolist(), todo.best_move.tolist(),
                                 todo.r.to_numpy(float))
            new = pd.DataFrame({"pos_id": todo.pos_id.to_numpy(), "maia_pbest": pb, "maia_entropy": ent})
            old = new if not len(old) else pd.concat([old, new], ignore_index=True)
            tmp = path.with_suffix(".tmp")
            old.to_parquet(tmp, index=False)
            tmp.replace(path)
    return df.merge(old, on="pos_id", how="left")


def prepare(cfg, df: pd.DataFrame, which: str, workers=None, maia: bool = True) -> pd.DataFrame:
    df = annotate(df)
    df = df[df.wp_best >= LOST].reset_index(drop=True)
    df = motif_table(cfg, which, df, workers)
    df["r"] = (player_maia_elo(df.mover_rating.to_numpy(float), cfg["maia"], df) if which == "player"
               else df.mover_rating.to_numpy(float))
    df = maia_table(cfg, which, df, maia)
    for m in MOTIFS:
        df[f"miss_{m}"] = df.missed.map(lambda x, m=m: m in x)
        df[f"allow_{m}"] = df.allowed.map(lambda x, m=m: m in x) & df.err
    df["any_shot"] = df.missed.map(len) > 0
    df["any_allowed"] = (df.allowed.map(len) > 0) & df.err
    return add_events(df)


def features(df: pd.DataFrame) -> pd.DataFrame:
    wp_alt = 1.0 / (1.0 + np.exp(-0.00368208 * df.cp_alt.astype(float)))
    X = pd.DataFrame({
        "r": df.r.astype(float),
        "clock_frac": df.clock_before.astype(float) / df.base_seconds.replace(0, np.nan).astype(float),
        "clock_s": df.clock_before.astype(float),
        "increment": df.increment.astype(float),
        "wp_best": df.wp_best.astype(float),
        "gap": (df.wp_best - wp_alt).astype(float),
        "critical": df.critical.astype(float),
        "ply": df.ply.astype(float),
        "n_legal": df.n_legal.astype(float),
        "phase": df.ph.map({p: i for i, p in enumerate(PHASES)}).astype(float),
        "etype": df.etype.map({e: i for i, e in enumerate(ETYPES)}).astype(float),
        "best_kind": df.best_kind.map({"quiet": 0, "capture": 1, "check": 2}).astype(float),
        "npm": df.npm.astype(float), "balance": df.balance.astype(float),
        "in_check": df.in_check.astype(float),
        "maia_pbest": df.maia_pbest.astype(float), "maia_entropy": df.maia_entropy.astype(float),
    })
    for m in MOTIFS:
        X[f"miss_{m}"] = df[f"miss_{m}"].astype(float)
    return X


# ------------------------------------------------------------------ concepts
def concepts(df: pd.DataFrame) -> list[dict]:
    """[{name, section, kind, mask (opportunity) or event (outcome), event_col}] in display order."""
    out = []
    quiet = ~df.any_shot.to_numpy()
    for sec in PHASES:
        ph = (df.ph == sec).to_numpy()
        for m in MOTIFS:
            out.append({"name": f"{sec}: missed {m}", "section": sec, "kind": "tactic missed",
                        "mask": ph & df[f"miss_{m}"].to_numpy(), "event": None})
            out.append({"name": f"{sec}: allowed {m}", "section": sec, "kind": "tactic allowed",
                        "mask": ph, "event": f"allow_{m}"})
        out.append({"name": f"{sec}: positional error (no tactic either way)", "section": sec,
                    "kind": "positional", "mask": ph, "event": "positional_err"})
        crit_quiet = ph & df.critical.to_numpy(bool) & quiet & (df.best_kind == "quiet").to_numpy()
        out.append({"name": f"{sec}: only move is quiet", "section": sec, "kind": "positional",
                    "mask": crit_quiet, "event": None})
        out.append({"name": f"{sec}: all positions", "section": sec, "kind": "diagnostic", "mask": ph,
                    "event": None})
        # the old slice leads (17b), re-tested with difficulty controlled: context only
        crit = ph & df.critical.to_numpy(bool)
        out.append({"name": f"{sec}: critical (one good move)", "section": sec, "kind": "diagnostic",
                    "mask": crit, "event": None})
        out.append({"name": f"{sec}: critical while pressing", "section": sec, "kind": "diagnostic",
                    "mask": crit & (df.wp_best >= 0.5).to_numpy(), "event": None})
    for e in ETYPES:
        out.append({"name": f"endgame: {e}", "section": "endgame", "kind": "endgame technique",
                    "mask": (df.etype == e).to_numpy(), "event": None})
    return out


def add_events(df: pd.DataFrame) -> pd.DataFrame:
    df["positional_err"] = df.err & ~df.any_shot & ~df.any_allowed
    return df


EVENTS = [f"allow_{m}" for m in MOTIFS] + ["positional_err"]


# ------------------------------------------------------------------ stage 1
STOP = dict(early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=0)


def _clf(small: bool = False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if small:                   # rare events: shallow trees, big leaves (they overfit otherwise)
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                              min_samples_leaf=100, l2_regularization=5.0, **STOP)
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=50, l2_regularization=1.0, **STOP)


def _reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=50, l2_regularization=1.0, **STOP)


class PeerModel:
    """P(err), E[loss], P(event | err) fitted on peer positions."""

    def fit(self, peers: pd.DataFrame, min_pos: int = 40):
        X = features(peers)
        y = peers.err.to_numpy(bool)
        self.err = _clf().fit(X, y)
        self.loss = _reg().fit(X, peers.wp_loss.to_numpy(float))
        e = peers[y]
        Xe = X[y]
        self.ev, self.ev_rate = {}, {}
        for c in EVENTS:
            t = e[c].to_numpy(bool)
            self.ev_rate[c] = float(t.mean()) if len(t) else 0.0
            self.ev[c] = _clf(small=True).fit(Xe, t) if t.sum() >= min_pos and (~t).sum() >= min_pos else None
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = features(df)
        out = pd.DataFrame({"p_err": self.err.predict_proba(X)[:, 1],
                            "l_hat": np.clip(self.loss.predict(X), 0, None)}, index=df.index)
        for c in EVENTS:
            q = self.ev[c].predict_proba(X)[:, 1] if self.ev[c] is not None else np.full(len(df), self.ev_rate[c])
            out[f"q_{c}"] = q
        return out


def crossfit(peers: pd.DataFrame, folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold predictions for every peer position (folds by game)."""
    games = peers.game_id.unique()
    fold = dict(zip(games, np.random.default_rng(seed).integers(0, folds, len(games))))
    f = peers.game_id.map(fold).to_numpy()
    parts = []
    for k in range(folds):
        te = f == k
        parts.append(PeerModel().fit(peers[~te]).predict(peers[te]))
    return pd.concat(parts).loc[peers.index]


def calibration(peers: pd.DataFrame, oof: pd.DataFrame, prior: float = 10.0, cs: list[dict] | None = None) -> dict:
    """concept -> (count factor, loss factor) so peers' out-of-fold observed/expected is 1 per concept
    (mildly shrunk to 1 with `prior` pseudo-counts). Applied to the player's expected values.
    cs: concepts on the peer rows (default: concepts(peers))."""
    out = {}
    for c in cs if cs is not None else concepts(peers):
        o, e, lo, le = expected(peers, oof, c)
        k = prior * le.sum() / max(e.sum(), 1e-9)           # `prior` events' worth of loss
        lf = (lo.sum() + k) / (le.sum() + k) if le.sum() > 0 else 1.0
        out[c["name"]] = (float((o.sum() + prior) / (e.sum() + prior)), float(lf))
    return out


def expected(df: pd.DataFrame, pred: pd.DataFrame, c: dict, cal: dict | None = None):
    """Per-row observed / expected count and loss for concept c (rows outside it are 0).
    cal: per-concept factors from calibration() scale the expected values."""
    m = c["mask"].astype(float)
    fc, fl = (cal or {}).get(c["name"], (1.0, 1.0))
    if c["event"] is None:
        o, e = df.err.to_numpy(float), pred.p_err.to_numpy()
        lo, le = df.wp_loss.to_numpy(float), pred.l_hat.to_numpy()
    else:
        q = pred[f"q_{c['event']}"].to_numpy()
        ev = df[c["event"]].to_numpy(float)
        o, e = ev, pred.p_err.to_numpy() * q
        lo, le = df.wp_loss.to_numpy(float) * ev, pred.l_hat.to_numpy() * q
    return m * o, m * e * fc, m * lo, m * le * fl


# ------------------------------------------------------------------ stage 2
def load_set(cfg, which: str, workers=None, maia: bool = True) -> pd.DataFrame | None:
    """Analyzed positions of one set with settled lines (03; line patterns need them), prepared for the
    model (None if it has no engine results). refined=False: same search depth for everyone."""
    from .results import load_analyzed
    try:
        df = load_analyzed(cfg, which, refined=False)
    except FileNotFoundError:
        return None
    return prepare(cfg, df, which, workers, maia)


def game_stats(df, pred, cs, cal):
    """games x (4 per concept + 4 overall) sums, and game ids."""
    cols = []
    for c in cs:
        cols += list(expected(df, pred, c, cal))
    cols += [df.err.to_numpy(float), pred.p_err.to_numpy(), df.wp_loss.to_numpy(float), pred.l_hat.to_numpy()]
    return per_game(df, cols)


def summarize(S, K, n_games, alpha=None, fit=None):
    """S: (..., 4K+4) sums -> dict of arrays (relative ratio raw/shrunk, elo, E').
    alpha None: fitted on the concepts where `fit` is True (non-diagnostic; they overlap less)."""
    from .findings import ELO_PER_SCORE
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


def excess(df, pred, cs, cal, B: int = 500, min_exp: float = 3.0, seed: int = 0):
    """Observed / expected per concept relative to the overall ratio: shrunk (alpha fitted on the
    non-diagnostic concepts), 90% game-bootstrap CI, temporal holdout (older 70% of games vs newest 30%).
    Returns (one row per concept, info dict)."""
    rng = np.random.default_rng(seed)
    K = len(cs)
    gids, M = game_stats(df, pred, cs, cal)
    G = len(gids)
    pt = summarize(M.sum(0), K, G, fit=np.array([c["kind"] != "diagnostic" for c in cs]))
    alpha = pt["alpha"]
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)])
    bs = summarize(W @ M, K, G, alpha)
    gdate = df.groupby("game_id").date.min().reindex(gids).to_numpy()
    order = np.argsort(gdate, kind="stable")
    cut = int(0.7 * G)
    old, new = np.zeros(G, bool), np.zeros(G, bool)
    old[order[:cut]], new[order[cut:]] = True, True
    tr = summarize(M[old].sum(0), K, old.sum(), alpha)
    ts = summarize(M[new].sum(0), K, new.sum(), alpha)
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
    info = {"alpha": alpha, "theta": float(pt["theta"][0]), "theta_lo": float(np.percentile(bs["theta"], 5)),
            "theta_hi": float(np.percentile(bs["theta"], 95)), "games": int(G), "positions": int(len(df))}
    judged = res[(res.old_exp >= min_exp) & (res.new_exp >= min_exp) & (res.kind != "diagnostic")]
    if len(judged) >= 5:
        info["holdout_corr_log_ratio"] = float(np.corrcoef(np.log(judged.old_ratio.clip(0.05)),
                                                          np.log(judged.new_ratio.clip(0.05)))[0, 1])
        info["holdout_n"] = int(len(judged))
    return res, info


def per_game(df, cols: list[np.ndarray]):
    """games x len(cols) sums."""
    gids, g = np.unique(df.game_id.to_numpy(), return_inverse=True)
    M = np.zeros((len(gids), len(cols)))
    for j, v in enumerate(cols):
        np.add.at(M[:, j], g, v)
    return gids, M


def eb_alpha(O, E) -> float:
    """Gamma(alpha, alpha) prior on ratios, fitted by the negative-binomial marginal likelihood."""
    from scipy.special import gammaln
    ok = E >= 3
    if ok.sum() < 5:
        return 20.0
    O, E = O[ok], E[ok]
    grid = np.exp(np.linspace(np.log(0.5), np.log(1e5), 200))     # top of grid = no spread between concepts
    ll = [np.sum(gammaln(O + a) - gammaln(a) - gammaln(O + 1) + a * np.log(a / (a + E)) + O * np.log(E / (a + E)))
          for a in grid]
    return float(grid[int(np.argmax(ll))])


def shrink(O, E, a):
    return (O + a) / (E + a)


# ------------------------------------------------------------------ line-pattern uptake (coach/linepatterns.py)
# Opportunity rows (best line has pattern X, 2nd-best doesn't, gap >= GAP): did the mover take it?
# Peer model P(took | controls, rating, pattern) -> the player's expected declines; ratio = observed /
# expected declines relative to the player's overall decline ratio theta; family -> pattern empirical-Bayes
# shrinkage (claims at family level; one player has ~10-60 opportunities per narrow pattern).
UP_DROP = [f"miss_{m}" for m in MOTIFS]         # concept flags are not controls


def opportunities(df: pd.DataFrame, lp: pd.DataFrame) -> pd.DataFrame:
    """Prepared positions x their line-pattern opportunity rows (one row per position x pattern)."""
    o = lp[lp.opportunity.astype(bool)].drop(columns=["opportunity"])
    return o.merge(df, on="pos_id", how="inner").reset_index(drop=True)


def up_features(o: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    X = features(o).drop(columns=UP_DROP)
    X["pattern"] = o.pattern.map({p: i for i, p in enumerate(patterns)}).astype(float)
    return X


class UptakeModel:
    """P(took | controls (rating, clock, eval, gap, Maia difficulty, ...), pattern) on peer opportunities."""

    def fit(self, o: pd.DataFrame, patterns: list[str]):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.patterns = patterns
        X = up_features(o, patterns)
        cat = [c == "pattern" for c in X.columns]
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                min_samples_leaf=40, l2_regularization=2.0,
                                                categorical_features=cat, **STOP)
        self.m.fit(X, o.took.to_numpy(bool))
        return self

    def predict(self, o: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(up_features(o, self.patterns))[:, 1]


def up_crossfit(o: pd.DataFrame, patterns, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(took) for every peer opportunity (folds by game)."""
    games = o.game_id.unique()
    fold = dict(zip(games, np.random.default_rng(seed).integers(0, folds, len(games))))
    f = o.game_id.map(fold).to_numpy()
    p = np.zeros(len(o))
    for k in range(folds):
        te = f == k
        p[te] = UptakeModel().fit(o[~te], patterns).predict(o[te])
    return p


def up_calibration(o: pd.DataFrame, p_oof: np.ndarray, patterns, B: int = 300, prior: float = 5.0, seed: int = 1):
    """Per-pattern decline factors c_k = (observed + prior) / (expected + prior) on the peers' out-of-fold
    predictions, plus B peer-game bootstrap draws of them. Their noise is shared by every player estimate,
    so it goes into the player's CIs and into the holdout null."""
    gids, g = np.unique(o.game_id.to_numpy(), return_inverse=True)
    k = o.pattern.map({p: i for i, p in enumerate(patterns)}).to_numpy()
    D = np.zeros((len(gids), len(patterns)))
    E = np.zeros_like(D)
    np.add.at(D, (g, k), (~o.took.to_numpy(bool)).astype(float))
    np.add.at(E, (g, k), 1 - p_oof)
    point = (D.sum(0) + prior) / (E.sum(0) + prior)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(gids), len(gids)), minlength=len(gids)) for _ in range(B)])
    boot = (W @ D + prior) / (W @ E + prior)
    return point, boot


def rating_slopes(o: pd.DataFrame, groups: dict, B: int = 200, seed: int = 2) -> dict:
    """Peer uptake vs rating per group (name -> pattern list): logistic slope of took on rating per 100
    Elo, with gap / eval / Maia P(best) / pattern dummies as controls; 90% game-bootstrap CI.
    > 0 = higher-rated peers take it more = skill; ~0 = style (no Elo claim)."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    out = {}
    for name, pats in groups.items():
        s = o[o.pattern.isin(pats)]
        if len(s) < 60 or s.took.nunique() < 2:
            out[name] = (np.nan, np.nan, np.nan, int(len(s)))
            continue
        wp_alt = 1.0 / (1.0 + np.exp(-0.00368208 * s.cp_alt.astype(float)))
        pb = s.maia_pbest.astype(float)
        X = pd.DataFrame({"r": (s.r.to_numpy(float) - 2000) / 100, "gap": (s.wp_best - wp_alt).to_numpy(float),
                          "wp": s.wp_best.to_numpy(float), "pb": pb.fillna(pb.mean() if pb.notna().any() else 0.3)})
        for q in pats[1:]:
            X[q] = (s.pattern == q).to_numpy(float)
        X = X.fillna(0.0).to_numpy()
        y = s.took.to_numpy(bool)

        def fit(i):
            if len(np.unique(y[i])) < 2:
                return np.nan
            return LogisticRegression(C=100.0, max_iter=500).fit(X[i], y[i]).coef_[0][0]
        gids, g = np.unique(s.game_id.to_numpy(), return_inverse=True)
        order = np.argsort(g, kind="stable")
        starts = np.searchsorted(g[order], np.arange(len(gids) + 1))
        bs = []
        for _ in range(B):
            pick = rng.integers(0, len(gids), len(gids))
            bs.append(fit(np.concatenate([order[starts[j]:starts[j + 1]] for j in pick])))
        lo, hi = np.nanpercentile(bs, [5, 95])
        out[name] = (float(fit(np.arange(len(s)))), float(lo), float(hi), int(len(s)))
    return out


def _up_cols(o: pd.DataFrame, p: np.ndarray, patterns):
    """games x (patterns x 4): opportunities, declines, raw expected declines, cost on declines."""
    k = o.pattern.map({q: i for i, q in enumerate(patterns)}).to_numpy()
    dec = (~o.took.to_numpy(bool)).astype(float)
    cost = o.cost.to_numpy(float)
    cols = []
    for i in range(len(patterns)):
        m = (k == i).astype(float)
        cols += [m, m * dec, m * (1 - p), m * dec * cost]
    return per_game(o, cols)


def up_summarize(S, c, fam_idx, inc, alphas=None):
    """S (B, 4P) sums, c (B, P) or (P,) decline factors, fam_idx: family -> pattern indices, inc: pattern
    indices that define theta. Raw / shrunk ratios (relative to theta) for patterns and families."""
    S = np.atleast_2d(S)
    c = np.atleast_2d(c)
    n, D, E0, C = (S[:, j::4] for j in range(4))
    E = E0 * c
    theta = D[:, inc].sum(1, keepdims=True) / np.maximum(E[:, inc].sum(1, keepdims=True), 1e-9)
    Er = theta * E
    F = list(fam_idx)
    Df = np.stack([D[:, fam_idx[f]].sum(1) for f in F], 1)
    Ef = np.stack([Er[:, fam_idx[f]].sum(1) for f in F], 1)
    if alphas is None:
        aF = eb_alpha(Df[0], Ef[0])
        rf0 = (Df[0] + aF) / (Ef[0] + aF)
        m = np.ones(D.shape[1])
        for j, f in enumerate(F):
            m[fam_idx[f]] = rf0[j]
        aP = eb_alpha(D[0][inc], (Er[0] * m)[inc])
    else:
        aF, aP = alphas
    rf = (Df + aF) / (Ef + aF)
    prior = np.ones_like(D)
    for j, f in enumerate(F):
        prior[:, fam_idx[f]] = rf[:, [j]]
    rp = (D + aP * prior) / (Er + aP)
    return {"n": n, "D": D, "Er": Er, "C": C, "raw": D / np.maximum(Er, 1e-9), "ratio": rp,
            "Df": Df, "Ef": Ef, "raw_f": Df / np.maximum(Ef, 1e-9), "ratio_f": rf, "theta": theta[:, 0],
            "alphas": (aF, aP)}


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    o = np.argsort(p[ok])
    r = p[ok][o] * ok.sum() / np.arange(1, ok.sum() + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(ok.sum())
    qq[o] = np.minimum(r, 1)
    q[ok] = qq
    return q


def _pval(boot_ratio: np.ndarray) -> np.ndarray:
    """Two-sided bootstrap p-value of ratio = 1 (floored at 1/B); used for BH only, never for ranking."""
    B = boot_ratio.shape[0]
    lo, hi = (boot_ratio <= 1).mean(0), (boot_ratio >= 1).mean(0)
    return np.clip(2 * np.minimum(lo, hi), 1.0 / B, 1.0)


def uptake_gate(o: pd.DataFrame, p: np.ndarray, patterns, families: dict, c_point, c_boot, inc,
                B: int = 500, seed: int = 0, peer_cost: dict | None = None):
    """The player's uptake per family and pattern vs the peer model -> (table, info).
    families: family -> pattern names (patterns allowed in family claims); inc: pattern indices for theta.
    CIs: player games resampled jointly with peer-calibration draws. Elo ~ 695 x f x delta x c_bar (upper
    bound): f opportunities per game, delta shrunk uptake deficit, c_bar mean win-prob cost when declined
    (the player's, else the peers')."""
    from .findings import ELO_PER_SCORE
    rng = np.random.default_rng(seed)
    pidx = {q: i for i, q in enumerate(patterns)}
    fam_idx = {f: [pidx[q] for q in qs] for f, qs in families.items() if qs}
    gids, M = _up_cols(o, p, patterns)
    G = len(gids)
    pt = up_summarize(M.sum(0), c_point, fam_idx, inc)
    al = pt["alphas"]
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)])
    cb = c_boot[rng.integers(0, len(c_boot), B)]
    bs = up_summarize(W @ M, cb, fam_idx, inc, al)
    gdate = o.groupby("game_id").date.min().reindex(gids).to_numpy()
    old = np.zeros(G, bool)
    old[np.argsort(gdate, kind="stable")[:int(0.7 * G)]] = True
    tr = up_summarize(M[old].sum(0), c_point, fam_idx, inc, al)
    ts = up_summarize(M[~old].sum(0), c_point, fam_idx, inc, al)

    def cbar(csum, dsum, name):
        return csum / dsum if dsum >= 5 else (peer_cost or {}).get(name, np.nan)

    def pct(v, q):
        v = np.asarray(v, float)
        return float(np.nanpercentile(v, q)) if np.isfinite(v).any() else float("nan")

    def row(level, fam, pat, n, D, Er, r, rb, rawb, d_raw, cb_, elo_b, pv, o_r, o_e, n_r, n_e):
        return {"level": level, "family": fam, "pattern": pat, "n_opp": int(n), "per_game": float(n / G),
                "declined": float(D), "expected": float(Er), "ratio_raw": float(D / max(Er, 1e-9)),
                "ratio": float(r), "ratio_lo": pct(rb, 5), "ratio_hi": pct(rb, 95),
                "raw_lo": pct(rawb, 5), "raw_hi": pct(rawb, 95),
                "delta": float((r - 1) * Er / max(n, 1)), "mde": float(2.49 * np.std(d_raw)),
                "cost_declined": float(cb_), "elo": float(ELO_PER_SCORE * (r - 1) * Er * cb_ / G),
                "elo_lo": pct(elo_b, 5), "elo_hi": pct(elo_b, 95),
                "p": float(pv), "old_ratio": float(o_r), "old_exp": float(o_e), "new_ratio": float(n_r),
                "new_exp": float(n_e)}
    rows = []
    pv_p, pv_f = _pval(bs["raw"]), _pval(bs["raw_f"])
    for j, (f, ii) in enumerate(fam_idx.items()):
        n = pt["n"][0, ii].sum()
        cb_ = cbar(pt["C"][0, ii].sum(), pt["D"][0, ii].sum(), f)
        d_raw = (bs["Df"][:, j] - bs["Ef"][:, j]) / np.maximum(bs["n"][:, ii].sum(1), 1)
        elo_b = ELO_PER_SCORE * (bs["ratio_f"][:, j] - 1) * bs["Ef"][:, j] * cb_ / G
        rows.append(row("family", f, "", n, pt["Df"][0, j], pt["Ef"][0, j], pt["ratio_f"][0, j], bs["ratio_f"][:, j],
                        bs["raw_f"][:, j], d_raw, cb_, elo_b, pv_f[j], tr["raw_f"][0, j], tr["Ef"][0, j], ts["raw_f"][0, j],
                        ts["Ef"][0, j]))
    for q, i in pidx.items():
        cb_ = cbar(pt["C"][0, i], pt["D"][0, i], q)
        d_raw = (bs["D"][:, i] - bs["Er"][:, i]) / np.maximum(bs["n"][:, i], 1)
        elo_b = ELO_PER_SCORE * (bs["ratio"][:, i] - 1) * bs["Er"][:, i] * cb_ / G
        rows.append(row("pattern", next((f for f, qs in families.items() if q in qs), ""), q, pt["n"][0, i],
                        pt["D"][0, i], pt["Er"][0, i], pt["ratio"][0, i], bs["ratio"][:, i], bs["raw"][:, i], d_raw, cb_, elo_b,
                        pv_p[i], tr["raw"][0, i], tr["Er"][0, i], ts["raw"][0, i], ts["Er"][0, i]))
    res = pd.DataFrame(rows)
    res["q"] = np.nan
    for lev in ("family", "pattern"):
        m = (res.level == lev).to_numpy()
        res.loc[m, "q"] = bh(res.loc[m, "p"].to_numpy())
    info = {"theta": float(pt["theta"][0]), "theta_lo": float(np.percentile(bs["theta"], 5)),
            "theta_hi": float(np.percentile(bs["theta"], 95)), "alpha_family": float(al[0]),
            "alpha_pattern": float(al[1]), "games": int(G), "opportunities": int(len(o)),
            "theta_old": float(tr["theta"][0]), "theta_new": float(ts["theta"][0])}
    hc, hn = holdout_corr(M[old].sum(0), M[~old].sum(0), c_point, fam_idx, inc, al)
    info.update({"holdout_corr": hc, "holdout_n": hn})
    return res, info


def holdout_corr(Mo, Mn, c, fam_idx, inc, al, min_exp: float = 3.0):
    """Correlation of log raw decline ratios, older vs newest games, over patterns in `inc` with enough
    expected declines in both halves."""
    a = up_summarize(Mo, c, fam_idx, inc, al)
    b = up_summarize(Mn, c, fam_idx, inc, al)
    ok = (a["Er"][0] >= min_exp) & (b["Er"][0] >= min_exp)
    ok &= np.isin(np.arange(len(ok)), inc)
    if ok.sum() < 4:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(np.log(np.clip(a["raw"][0][ok], 0.05, None)),
                             np.log(np.clip(b["raw"][0][ok], 0.05, None)))[0, 1]), int(ok.sum())


def simulate_declines(o, p, patterns, c_true, theta, rng, boost: dict | None = None):
    """Null player: declines ~ Bernoulli(theta x (1 - p) x c_true[pattern]) (x boost[pattern] if given)."""
    k = o.pattern.map({q: i for i, q in enumerate(patterns)}).to_numpy()
    q = theta * (1 - p) * c_true[k]
    if boost:
        q = q * o.pattern.map(boost).fillna(1.0).to_numpy()
    return rng.random(len(o)) < np.clip(q, 0, 1)


def null_checks(o, p, patterns, families, c_point, c_boot, inc, theta, S: int = 200, inject: float = 1.5,
                S_inj: int = 30, B: int = 200, seed: int = 5, q_max: float = 0.10):
    """(1) Holdout correlation under the null, with peer-calibration noise shared by both halves (true
    factors drawn from the peer bootstrap, estimates use the point factors). (2) False-positive rate of the
    family gate (CI excludes 1 and BH q < q_max) on null players. (3) Injected effect: decline rate x
    `inject` in one family at a time -> how often the gate finds it (power at that effect size)."""
    rng = np.random.default_rng(seed)
    pidx = {q: i for i, q in enumerate(patterns)}
    fam_idx = {f: [pidx[q] for q in qs] for f, qs in families.items() if qs}
    gids = np.unique(o.game_id.to_numpy())
    gdate = o.groupby("game_id").date.min().reindex(gids).to_numpy()
    old = np.zeros(len(gids), bool)
    old[np.argsort(gdate, kind="stable")[:int(0.7 * len(gids))]] = True
    corr = []
    for _ in range(S):
        dec = simulate_declines(o, p, patterns, c_boot[rng.integers(0, len(c_boot))], theta, rng)
        _, M = _up_cols(o.assign(took=~dec), p, patterns)
        corr.append(holdout_corr(M[old].sum(0), M[~old].sum(0), c_point, fam_idx, inc, None)[0])

    def flagged(dec):
        res, _ = uptake_gate(o.assign(took=~dec), p, patterns, families, c_point, c_boot, inc, B=B,
                             seed=int(rng.integers(1 << 30)))
        f = res[res.level == "family"]
        return set(f.family[((f.raw_lo > 1) | (f.raw_hi < 1)) & (f.q < q_max)])
    fp = [len(flagged(simulate_declines(o, p, patterns, c_point, theta, rng))) / max(len(fam_idx), 1)
          for _ in range(max(S_inj // 2, 10))]
    det = {f: float(np.mean([f in flagged(simulate_declines(o, p, patterns, c_point, theta, rng,
                                                              {q: inject for q in families[f]}))
                             for _ in range(S_inj)])) for f in fam_idx}
    return {"holdout_null": [float(x) for x in corr if x == x], "false_positive_rate": float(np.mean(fp)),
            "inject": inject, "detection": det}
