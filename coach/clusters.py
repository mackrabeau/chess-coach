"""Position clusters for puzzle selection (exploratory): k-means on Maia-3 hidden states groups the player's
critical errors into recurring kinds of position, and pick_puzzles() chooses representative, non-duplicate
puzzles from a group (or from any candidate set: line-pattern drills call it too).

Clusters are NOT a concept source (docs/future_work.md "Concept groups": they reduce to material / phase /
difficulty slices and add no peer predictive information beyond 24's features), so nothing here makes a
weakness or Elo claim. They decide WHICH positions to drill, not WHAT to train. Sets are judged by coherence:
within-set agreement of motif / swing type / phase vs size-matched random sets (coherence()), plus eyeballing.

Embeddings (embeddings(), cached per set in <set>/maia_emb_*.npz), one fixed Elo for both sides, current board:
  pool "mean"  block `layer` averaged over the 64 squares (layer None: the final pooled vector)
  pool "move"  [square mean, state at the best move's from-square, state at its to-square] of block `layer`:
               describes the position AND what the solution does there (puzzle themes are about the solution).
readable() + rule() name clusters by short rules on readable features (used by 25's --validate research path).
"""
from __future__ import annotations

import chess
import numpy as np
import pandas as pd

from .config import data_dir
from .weakness import ETYPES, MOTIFS, PHASES

EMB_VERSION = 1     # bump when the embedding recipe changes: caches are rebuilt


# ------------------------------------------------------------------ embeddings
def _square_states(maia, tokens, elo: int, layer: int):
    """(B, 64, d) output of transformer block `layer` at one Elo for both sides (squares in the side-to-move
    frame: python-chess index of the mirrored board when Black is to move)."""
    torch = maia.torch
    got = {}
    hook = maia.model.transformer.layers[layer].register_forward_hook(lambda m, i, o: got.__setitem__("x", o))
    try:
        e = torch.full((tokens.shape[0],), int(elo), dtype=torch.long, device=maia.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=maia.amp):
            maia.model(tokens.to(maia.device), e, e)
    finally:
        hook.remove()
    return got["x"].float().cpu().numpy()


def _move_squares(fen: str, uci: str) -> tuple[int, int]:
    """Token indices of a move's from / to squares (Maia mirrors the board when Black is to move)."""
    m = chess.Move.from_uci(uci)
    if chess.Board(fen).turn == chess.WHITE:
        return m.from_square, m.to_square
    return chess.square_mirror(m.from_square), chess.square_mirror(m.to_square)


def embeddings(cfg, which: str, df: pd.DataFrame, elo: int, layer: int | None, batch: int = 256, maia=None,
               pool: str = "mean"):
    """Maia-3 hidden state per row of df (float32, aligned with df), cached per set in
    <set>/maia_emb_[mv]L<layer>_e<elo>.npz. Current board only (no move history), as Maia's UCI default.
    pool "move" needs df.best_move and a block layer; its cache key is pos_id|best_move.
    Returns (X, maia) so one loaded model serves several sets."""
    import torch
    assert pool in ("mean", "move") and not (pool == "move" and layer is None), (pool, layer)
    tag = ("mv" if pool == "move" else "") + f"L{'last' if layer is None else layer}"
    path = data_dir(cfg, which) / f"maia_emb_{tag}_e{elo}.npz"
    key = df.pos_id.astype(str) + ("|" + df.best_move.astype(str) if pool == "move" else "")
    ids0, X0 = np.array([], dtype=str), None
    if path.exists():
        with np.load(path, allow_pickle=False) as z:     # closed before the rewrite below (Windows locks it)
            if int(z["v"]) == EMB_VERSION:
                ids0, X0 = z["pos_id"], z["X"]
    ids = key.to_numpy()
    idx = pd.Index(ids0).get_indexer(ids)
    todo = df.assign(key=key)[idx < 0].drop_duplicates("key")
    if len(todo):
        if maia is None:
            from .maia import Maia
            maia = Maia(cfg["maia"].get("model", "23m"))
        print(f"  Maia embeddings ({tag}): {which} {len(todo):,} positions")
        fens, out = todo.fen.tolist(), []
        moves = todo.best_move.tolist() if pool == "move" else None
        for b0 in range(0, len(fens), batch):
            T = torch.stack([maia.tokens(f) for f in fens[b0:b0 + batch]])
            if pool == "mean":
                out.append(maia.embed(T, elo, layer).astype(np.float16))
            else:
                S = _square_states(maia, T, elo, layer)
                sq = np.array([_move_squares(f, u) for f, u in zip(fens[b0:b0 + batch], moves[b0:b0 + batch])])
                r = np.arange(len(S))
                out.append(np.concatenate([S.mean(1), S[r, sq[:, 0]], S[r, sq[:, 1]]], 1).astype(np.float16))
            if (b0 // batch) % 40 == 39:
                print(f"    {b0 + batch:,} / {len(fens):,}")
        new = np.concatenate(out)
        ids0 = np.concatenate([ids0, todo.key.astype(str).to_numpy()]).astype(str)   # not object: no pickle
        X0 = new if X0 is None else np.concatenate([X0, new])
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, pos_id=ids0, X=X0, v=EMB_VERSION)
        tmp.replace(path)
        idx = pd.Index(ids0).get_indexer(ids)
    return X0[idx].astype(np.float32), maia


# ------------------------------------------------------------------ clusters
class PositionClusters:
    """Standardize -> PCA -> k-means (25 fits it on peer + your puzzle positions; --validate on peers only)."""

    def __init__(self, k: int = 32, dims: int = 32, seed: int = 0):
        self.k, self.dims, self.seed = k, dims, seed

    def fit(self, X):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.pca = PCA(min(self.dims, X.shape[1]), random_state=self.seed).fit(Z)
        self.km = KMeans(self.k, n_init=4, random_state=self.seed).fit(self.pca.transform(Z))
        return self

    def project(self, X):
        return self.pca.transform(self.sc.transform(X))

    def assign(self, X):
        return self.km.predict(self.project(X))


# ------------------------------------------------------------------ naming
def readable(df: pd.DataFrame) -> pd.DataFrame:
    """Position features a player can see or be told, for naming rules (no rating / clock)."""
    board = df.fen.str.split(" ").str[0]
    F = {p: df.ph == p for p in PHASES}
    F.update({e: df.etype == e for e in ETYPES})
    F.update({
        "queens on": board.str.contains("Q") | board.str.contains("q"),
        "critical (one good move)": df.critical.astype(bool),
        "best move is a capture": df.best_kind == "capture",
        "best move is a check": df.best_kind == "check",
        "in check": df.in_check.astype(bool),
        "a tactic is available": df.any_shot.astype(bool),
    })
    F.update({f"{m.replace('_', ' ')} available": df[f"miss_{m}"].astype(bool) for m in MOTIFS})
    F.update({"material balance (pawns, mover)": df.balance, "non-pawn material": df.npm,
              "win chance with best move": df.wp_best, "legal moves": df.n_legal, "move number": df.ply // 2 + 1})
    return pd.DataFrame(F, index=df.index).astype(float)


def rule(F: pd.DataFrame, y: np.ndarray, groups: np.ndarray, depth: int = 3, seed: int = 0):
    """Best single conjunction (one leaf of a depth-`depth` tree) recovering membership y.
    Chosen on half the games, scored on the other half. Returns (text, held-out F1, held-out precision, recall)."""
    from sklearn.tree import DecisionTreeClassifier
    u = np.unique(groups)
    half = set(np.random.default_rng(seed).permutation(u)[: len(u) // 2])
    tr = np.array([g in half for g in groups])
    X = F.to_numpy()
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=30, class_weight="balanced",
                               random_state=seed).fit(X[tr], y[tr])
    T, leaf = t.tree_, t.apply(X)

    def leaves(node, conds):
        if T.children_left[node] == -1:
            yield node, conds
            return
        f, th = T.feature[node], T.threshold[node]
        yield from leaves(T.children_left[node], conds + [(f, False, th)])
        yield from leaves(T.children_right[node], conds + [(f, True, th)])

    def f1(pred, truth):
        tp = (pred & truth).sum()
        p, r = tp / max(pred.sum(), 1), tp / max(truth.sum(), 1)
        return (2 * p * r / (p + r) if p + r else 0.0), p, r

    best = max(leaves(0, []), key=lambda lc: f1(leaf[tr] == lc[0], y[tr].astype(bool))[0])
    score = f1(leaf[~tr] == best[0], y[~tr].astype(bool))
    binary = {c for c in F.columns if set(np.unique(F[c])) <= {0.0, 1.0}}
    parts = []
    for f, above, th in best[1]:
        c = F.columns[f]
        parts.append((c if above else f"not {c}") if c in binary else f"{c} {'>' if above else '<='} {th:.2f}")
    return " and ".join(dict.fromkeys(parts)) or "(all positions)", *map(float, score)


# ------------------------------------------------------------------ puzzles
def puzzle_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Readable tags of each puzzle (for coherence and set names): phase, piece the best move moves, swing type (coach/motifs.classify:
    missed_win / allowed / both / no_material), primary motif, kind of best move, queens on / off.
    df needs fen, ph, best_kind and (settled) best_line / played_line."""
    from .motifs import classify
    rows = []
    for fen, b, p in zip(df.fen, df.best_line, df.played_line):
        try:
            c = classify(fen, list(b), list(p))
            rows.append((c["direction"], c["motif"]))
        except ValueError:                        # an illegal move in a stored line
            rows.append(("no_material", "other"))
    board = df.fen.str.split(" ").str[0]
    piece = [chess.piece_name(chess.Board(f).piece_type_at(chess.Move.from_uci(u).from_square) or chess.PAWN)
             for f, u in zip(df.fen, df.best_move)]
    return pd.DataFrame({"phase": df.ph.to_numpy(), "piece": piece, "swing": [r[0] for r in rows], "motif": [r[1] for r in rows],
                         "best move": df.best_kind.to_numpy(),
                         "queens": np.where(board.str.contains("Q") | board.str.contains("q"), "on", "off")},
                        index=df.index)


def _agree(v: np.ndarray) -> float:
    m = np.bincount(v)
    return float((m * (m - 1)).sum() / max(len(v) * (len(v) - 1), 1))


def coherence(lab: pd.DataFrame, sets: np.ndarray, min_n: int = 8, reps: int = 300, seed: int = 0):
    """Within-set agreement vs size-matched random sets from the same pool.

    Per label column and set: agreement = chance that two different puzzles of the set share the tag
    (sum of n_v (n_v - 1) / n (n - 1); sensitive to a minority tag concentrating, unlike the modal share); the
    null is the agreement of random subsets of the same size drawn from all rows of lab. Returns (per-set
    DataFrame with agreement, null mean and one-sided p per label; summary dict: per label the size-weighted
    mean agreement, null and excess, plus the share of sets beating the null at p < 0.05, ~0.05 by chance)."""
    rng = np.random.default_rng(seed)
    codes = {c: pd.factorize(lab[c])[0] for c in lab.columns}
    out, null_cache = [], {}
    for s in pd.unique(sets):
        i = np.flatnonzero(sets == s)
        n = len(i)
        if n < min_n:
            continue
        row = {"set": s, "n": n}
        for c, v in codes.items():
            pur = _agree(v[i])
            if (c, n) not in null_cache:
                null_cache[(c, n)] = np.array([_agree(v[rng.choice(len(v), n, replace=False)]) for _ in range(reps)])
            nl = null_cache[(c, n)]
            row.update({c: pur, f"{c} null": nl.mean(), f"{c} p": (1 + (nl >= pur).sum()) / (reps + 1)})
        out.append(row)
    T = pd.DataFrame(out)
    summ = {}
    if len(T):
        w = T.n / T.n.sum()
        for c in codes:
            summ[c] = {"agreement": float((w * T[c]).sum()), "null": float((w * T[f"{c} null"]).sum()),
                       "excess": float((w * (T[c] - T[f"{c} null"])).sum()),
                       "sets_beating_null": float((T[f"{c} p"] < 0.05).mean())}
        summ["mean_excess"] = float(np.mean([summ[c]["excess"] for c in codes]))
    return T, summ


TAG_TEXT = {"phase": "{v}", "piece": "{v} move", "swing": {"missed_win": "a win you missed", "allowed": "a tactic you allowed",
                                      "both": "missed + allowed tactic", "no_material": "no material swing"},
            "motif": "{v}", "best move": {"quiet": "quiet best move", "capture": "best move captures",
                                           "check": "best move checks"}, "queens": "queens {v}"}


def name_set(lab_set: pd.DataFrame, lab_all: pd.DataFrame, share: float = 0.5, lift: float = 1.25,
             most: int = 3) -> str:
    """Short name from the tags that dominate the set (share >= `share`) and are over-represented vs the
    whole pool (lift >= `lift`), most over-represented first, e.g. "endgame (90%) · best move is a check
    (71%)". Falls back to the most common phase."""
    parts = []
    for c in lab_set.columns:
        vc = lab_set[c].value_counts(normalize=True)
        v, sh = vc.index[0], float(vc.iat[0])
        base = float((lab_all[c] == v).mean())
        if sh >= share and sh / max(base, 1e-9) >= lift and not (c == "motif" and v == "other"):
            t = TAG_TEXT.get(c, "{v}")
            text = t.get(v, v) if isinstance(t, dict) else t.format(v=str(v).replace("_", " "))
            parts.append((sh / base, f"{text} ({sh:.0%})"))
    parts = [p for _, p in sorted(parts, reverse=True)[:most]]
    return " · ".join(parts) or f"{lab_set.phase.mode().iat[0]} (mixed)"


def _board_key(fen: str) -> str:
    return " ".join(fen.split(" ")[:2])


def pick_puzzles(df: pd.DataFrame, k: int = 10, X: np.ndarray | None = None, centre: np.ndarray | None = None,
                 per_game: int = 1, min_dist: float = 0.3, w_typical: float = 1.0, w_recent: float = 1.0,
                 w_cost: float = 1.0, diversity: float = 1.0, cost_col: str = "wp_loss") -> pd.DataFrame:
    """Pick up to k puzzles from candidate positions: representative, recent, costly and not near-duplicates.

    df      one row per candidate; needs fen (and pos_id for the caller). Optional columns, each used when
            present: game_id (per-game cap), date (recency), `cost_col` (default wp_loss: win prob lost),
            history_fens (same opening line), ph (the opening-line rule applies to opening positions only).
    X       optional embedding rows aligned with df (e.g. coach.clusters.embeddings(..., pool="move"), or a
            PositionClusters.project() of it). Without X there is no typicality / distance rule.
    centre  the point "typical" is measured from (default: mean of X), e.g. the k-means cluster centre.
    Hard rules (a candidate is skipped): more than `per_game` picks from one game; the same board + side to
      move as a pick; in the opening, one position on the other's line (in its history_fens); embedding
      distance to a pick below `min_dist` x the median pairwise distance of the candidates.
    Soft score (each term a percentile rank in [0, 1], weights w_*): typical (close to centre) + recent
      (later date) + cost; picked greedily, minus `diversity` x closeness to the nearest pick
      (1 - distance / median distance, floored at 0), so later picks spread out over the set.
    Returns the picked rows in pick order, with columns pick (1..), typical (percentile, 1 = most central,
    NaN without X) and score. Deterministic (ties broken by df order). O(n^2) memory in X: keep n <~ 5,000."""
    n = len(df)
    if n == 0 or k <= 0:
        return df.iloc[:0].assign(pick=pd.Series(dtype=int), typical=pd.Series(dtype=float),
                                  score=pd.Series(dtype=float))

    def rank(v):
        return pd.Series(v).rank(pct=True, method="average").to_numpy() if n > 1 else np.ones(n)

    score = np.zeros(n)
    typical = np.full(n, np.nan)
    D = None
    if X is not None:
        X = np.asarray(X, dtype=np.float64)
        c = X.mean(0) if centre is None else np.asarray(centre, dtype=np.float64)
        typical = rank(-np.linalg.norm(X - c, axis=1))
        score += w_typical * typical
        sq = (X ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        med = float(np.median(D[np.triu_indices(n, 1)])) if n > 1 else 1.0
        D = D / max(med, 1e-12)
    if "date" in df and w_recent:
        score += w_recent * rank(pd.to_datetime(df.date, errors="coerce").astype("int64").to_numpy())
    if cost_col in df and w_cost:
        score += w_cost * rank(df[cost_col].to_numpy(float))
    games = df.game_id.to_numpy() if "game_id" in df else np.arange(n)
    boards = df.fen.map(_board_key).to_numpy()
    opening = (df.ph == "opening").to_numpy() if "ph" in df else np.zeros(n, bool)
    hist = ([{_board_key(h) for h in (hs if hs is not None else [])} for hs in df.history_fens]
            if "history_fens" in df else [set()] * n)
    picked, per, pboards = [], {}, set()
    alive = np.ones(n, bool)
    near = np.zeros(n)                           # closeness to the nearest pick, 0..1
    while len(picked) < k and alive.any():
        s = np.where(alive, score - diversity * near, -np.inf)
        j = int(np.argmax(s))
        picked.append((j, float(s[j])))
        alive[j] = False
        per[games[j]] = per.get(games[j], 0) + 1
        pboards.add(boards[j])
        for i in np.flatnonzero(alive):
            if per.get(games[i], 0) >= per_game or boards[i] in pboards or \
                    (opening[i] and opening[j] and (boards[j] in hist[i] or boards[i] in hist[j])):
                alive[i] = False
        if D is not None:
            alive &= D[j] >= min_dist
            near = np.maximum(near, np.clip(1 - D[j], 0, None))
    idx = [j for j, _ in picked]
    return df.iloc[idx].assign(pick=np.arange(1, len(idx) + 1), typical=typical[idx],
                               score=[s for _, s in picked])
