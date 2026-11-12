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

MOTIF_VERSION = 1              # bump when coach/motifs.py changes: the cache is rebuilt
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
def prepare(cfg, df: pd.DataFrame, which: str, workers=None) -> pd.DataFrame:
    df = annotate(df)
    df = df[df.wp_best >= LOST].reset_index(drop=True)
    df = motif_table(cfg, which, df, workers)
    df["r"] = (player_maia_elo(df.mover_rating.to_numpy(float), cfg["maia"], df) if which == "player"
               else df.mover_rating.to_numpy(float))
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


def calibration(peers: pd.DataFrame, oof: pd.DataFrame, prior: float = 10.0) -> dict:
    """concept -> (count factor, loss factor) so peers' out-of-fold observed/expected is 1 per concept
    (mildly shrunk to 1 with `prior` pseudo-counts). Applied to the player's expected values."""
    out = {}
    for c in concepts(peers):
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
