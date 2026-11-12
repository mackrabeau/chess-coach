"""What goes wrong in balanced endgames: you vs peers.

Balanced endgame = endgame phase and the best move's win probability in
practical.contested-style range [0.2, 0.8] (endgames.balanced). Error = mistake or blunder.

  1. By endgame type (pawn / rook / minor / queen / mixed): error rate and loss per 100 moves.
  2. By the piece you moved: error rate; and how often the best move was a king move /
     pawn move that wasn't played.
  3. Endgame features at the end of the best vs the played line (king activity, passed
     pawns, rook placement, trades...): in errors, how often the best line ended with
     MORE / LESS of each, you vs peers.
  4. Error anatomy (09c): how deep the punishment was, what kind of reply, on the radar.
  5. Maia difficulty (09b): error rate by how often a player at your level finds the move.
  6. Think time (you only).
Main comparison: your games >= player.min_player_rating vs peers; a second column for your
whole history (you vs yourself, more data).

    python scripts/11b_balanced_endgames.py --config configs/v0.yaml

Output: stdout + data/player/balanced_endgames.html
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.difficulty import p_best_own
from coach.endgame_features import READABLE, end_of, features, material_class
from coach.handcrafted import phase
from coach.report import page, position_card
from coach.results import load_analyzed
from coach.stats import ci, game_bootstrap

PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
         chess.QUEEN: "queen", chess.KING: "king"}


def balanced(cfg, which):
    df = load_analyzed(cfg, which)
    lo, hi = cfg.get("endgames", {}).get("balanced", [0.2, 0.8])
    boards = [chess.Board(f) for f in df.fen]
    ph = np.array([phase(b) for b in boards])
    keep = (ph == "endgame") & (df.wp_best >= lo).to_numpy() & (df.wp_best <= hi).to_numpy()
    df = df[keep].reset_index(drop=True)
    boards = [b for b, k in zip(boards, keep) if k]
    df["phase"] = "endgame"
    df["etype"] = [material_class(b) for b in boards]
    df["moved"] = [PIECE[b.piece_type_at(chess.Move.from_uci(m).from_square)] for b, m in zip(boards, df.move)]
    df["best_piece"] = [PIECE[b.piece_type_at(chess.Move.from_uci(m).from_square)] for b, m in zip(boards, df.best_move)]
    df["err"] = (df.label.isin(["mistake", "blunder"]) & ~df.played_is_best).to_numpy()
    return df


def rate_rows(groups, me, pop, B, min_n=20):
    """Error rate per group: you (CI over games) vs peers. groups: list of (label, mask_me, mask_pop)."""
    rows, lines = [], []
    for label, mm, mp in groups:
        if mm.sum() < min_n or mp.sum() < min_n:
            continue
        bt = game_bootstrap(me, np.arange(len(me)), np.stack([mm, mm & me.err.to_numpy()], 1).astype(float), B)
        lo, hi = ci(bt[:, 1] / np.maximum(bt[:, 0], 1e-9))
        r_me = (mm & me.err.to_numpy()).sum() / mm.sum()
        r_pop = (mp & pop.err.to_numpy()).sum() / mp.sum()
        clear = lo > r_pop or hi < r_pop
        tag = ("clear" if clear else "within noise")
        cls = "muted" if not clear else ("down" if r_me > r_pop else "up")
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{mm.sum()}</td><td>{r_me:.1%} <span class='muted'>"
                    f"[{lo:.1%}, {hi:.1%}]</span></td><td>{mp.sum()}</td><td>{r_pop:.1%}</td>"
                    f"<td><span class='{cls}'>{tag}</span></td></tr>")
        lines.append(f"  {label:34s} you {r_me:5.1%} [{lo:.1%}, {hi:.1%}] (n {mm.sum()})  peers {r_pop:5.1%} "
                     f"(n {mp.sum()})  {tag}")
    table = ("<table><tr><th></th><th>your moves</th><th>your error rate (90% CI)</th><th>peer moves</th>"
             "<th>peers</th><th>vs peers</th></tr>" + "".join(rows) + "</table>")
    return table, lines


def feature_directions(df):
    """Per error: sign of (best-line end - played-line end) for every endgame feature."""
    out = []
    e = df[df.err]
    for fen, best, played in zip(e.fen, e.best_line, e.played_line):
        us = chess.Board(fen).turn
        try:
            fb = features(end_of(fen, list(best)), us)
            fp = features(end_of(fen, list(played)), us)
        except Exception:
            continue
        out.append({k: np.sign(fb[k] - fp[k]) for k in fb})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    B = cfg["report"]["bootstrap"]
    cmp_min = cfg["player"]["min_player_rating"]
    me_all, pop = balanced(cfg, "player"), balanced(cfg, "population")
    me = me_all[me_all.mover_rating >= cmp_min].reset_index(drop=True)
    lines = [f"balanced endgame positions: you {len(me):,} moves in {me.game_id.nunique()} games (>= {cmp_min}), "
             f"whole history {len(me_all):,} moves; peers {len(pop):,} moves",
             f"error rate overall: you {me.err.mean():.1%}, whole history {me_all.err.mean():.1%}, peers {pop.err.mean():.1%}"]
    body = ["<h1>Balanced endgames: what goes wrong</h1>",
            f"<p class='muted'>Endgame positions where neither side is clearly better (best-move win probability "
            f"0.2–0.8). Error = mistake or blunder. You: games at chess.com ≥ {cmp_min} ({len(me):,} moves, "
            f"{me.game_id.nunique()} games); peers: {len(pop):,} moves. Overall error rate: you {me.err.mean():.1%}, "
            f"peers {pop.err.mean():.1%}.</p>"]

    # 1. by endgame type
    types = sorted(set(me.etype) | set(pop.etype))
    t, l = rate_rows([(t_, (me.etype == t_).to_numpy(), (pop.etype == t_).to_numpy()) for t_ in types], me, pop, B)
    body += ["<h2>1. By endgame type</h2>", t]
    lines += ["1. error rate by endgame type:"] + l

    # 2. by moved piece, and best-move piece not played
    pieces = ["king", "pawn", "rook", "bishop", "knight", "queen"]
    t, l = rate_rows([(f"you moved a {p}", (me.moved == p).to_numpy(), (pop.moved == p).to_numpy()) for p in pieces],
                     me, pop, B)
    body += ["<h2>2. By the piece moved</h2>", t]
    lines += ["2. error rate by the piece moved:"] + l
    miss_rows = []
    lines.append("   when the best move was a ... and you played something else (share of all such positions):")
    for p in ["king", "pawn", "rook"]:
        bm, bp = (me.best_piece == p), (pop.best_piece == p)
        if bm.sum() < 20 or bp.sum() < 20:
            continue
        r_me = (bm & (me.moved != p) & me.err).sum() / bm.sum()
        r_pop = (bp & (pop.moved != p) & pop.err).sum() / bp.sum()
        lines.append(f"     best = {p} move: you erred with another piece {r_me:.1%} (n {bm.sum()}) vs peers {r_pop:.1%}")
        miss_rows.append(f"<tr><td>best move was a {p} move</td><td>{bm.sum()}</td><td>{r_me:.1%}</td>"
                         f"<td>{bp.sum()}</td><td>{r_pop:.1%}</td></tr>")
    body.append("<table><tr><th></th><th>your positions</th><th>you erred with a different piece</th>"
                "<th>peer positions</th><th>peers</th></tr>" + "".join(miss_rows) + "</table>")

    # 3. feature directions in errors
    fd_me, fd_pop = feature_directions(me), feature_directions(pop)
    rows3 = []
    lines.append(f"3. in errors, the best line ended with MORE of ... than the played line (you n {len(fd_me)}, peers n {len(fd_pop)}):")
    if len(fd_me) >= 20 and len(fd_pop) >= 20:
        scored = []
        for col in fd_me.columns:
            for sign, word in ((1, "more"), (-1, "less")):
                a, b = (fd_me[col] == sign).mean(), (fd_pop[col] == sign).mean()
                n = len(fd_me)
                se = np.sqrt(max(a * (1 - a), 1e-9) / n + max(b * (1 - b), 1e-9) / len(fd_pop))
                scored.append((col, word, a, b, (a - b) / se))
        scored.sort(key=lambda x: -abs(x[4]))
        for col, word, a, b, z in scored[:16]:
            who, f = col.split(" ", 1)
            text = f"{word} {who} {READABLE.get(f, f)}"
            # ~60 comparisons: Bonferroni-ish |z| >= 3; errors treated as independent (not game-clustered)
            flag = "clear" if abs(z) >= 3 and (a > 0.05 or b > 0.05) else "within noise"
            lines.append(f"  {text:52s} you {a:5.1%}  peers {b:5.1%}  ({flag})")
            rows3.append(f"<tr><td>best line had {html.escape(text)}</td><td>{a:.0%}</td><td>{b:.0%}</td>"
                         f"<td>{flag}</td></tr>")
    body += ["<h2>3. What the best line had that yours didn't</h2>",
             "<p class='muted'>Endgame features compared at the end of the best line and the played line, in "
             "errors only. Sorted by how different you are from peers; 'clear' needs |z| ≥ 3 because ~60 "
             "comparisons are made (errors treated as independent).</p>",
             "<table><tr><th>in your errors, the best line ended with</th><th>you</th><th>peers</th><th>vs peers</th></tr>"
             + "".join(rows3) + "</table>"]

    # 4. anatomy (09c)
    paths = {n: data_dir(cfg, n) / "anatomy.parquet" for n in ("player", "population")}
    if all(p.exists() for p in paths.values()):
        def depth(df, name):
            an = pd.read_parquet(paths[name]).drop_duplicates("pos_id")
            a = df[df.err][["pos_id"]].merge(an, on="pos_id", how="left")
            rp, pr = a.refute_ply.to_numpy(float), a.p_reply.to_numpy(float)
            return {"punished next move": np.mean(rp <= 2), "2-move combination": np.mean((rp > 2) & (rp <= 4)),
                    "deeper": np.mean(rp > 4), "no material swing": np.mean(np.isnan(rp)),
                    "refuting reply off the radar (< 5%)": np.nanmean(np.where(np.isnan(rp), np.nan, pr < 0.05)),
                    "missed a material win": np.mean(np.isfinite(a.gain_ply.to_numpy(float)))}
        dm, dp = depth(me, "player"), depth(pop, "population")
        lines.append("4. error anatomy (share of errors):")
        rows4 = []
        for k in dm:
            lines.append(f"  {k:38s} you {dm[k]:5.1%}  peers {dp[k]:5.1%}")
            rows4.append(f"<tr><td>{k}</td><td>{dm[k]:.0%}</td><td>{dp[k]:.0%}</td></tr>")
        body += ["<h2>4. How the errors were punished</h2><table><tr><th>share of errors</th><th>you</th>"
                 "<th>peers</th></tr>" + "".join(rows4) + "</table>"]

    # 5. Maia difficulty (09b)
    dme, dpo = p_best_own(cfg, me, "player"), p_best_own(cfg, pop, "population")
    if dme is not None and dpo is not None:
        bands = [(0, 0.2, "hard: Maia at your level finds best < 20%"), (0.2, 0.5, "medium: 20-50%"),
                 (0.5, 1.01, "easy: >= 50%")]
        t, l = rate_rows([(n, (dme >= a) & (dme < b), (dpo >= a) & (dpo < b)) for a, b, n in bands], me, pop, B)
        body += ["<h2>5. By difficulty for your level (Maia)</h2>", t]
        lines += ["5. error rate by Maia difficulty:"] + l

    # 6. think time (you, whole history)
    an = data_dir(cfg, "player") / "anatomy.parquet"
    if an.exists():
        T = me_all[["pos_id"]].merge(pd.read_parquet(an)[["pos_id", "think_s"]].drop_duplicates("pos_id"),
                                     on="pos_id", how="left").think_s.to_numpy(float)
        lines.append("6. think time in balanced endgames (you, whole history):")
        rows6 = []
        for a, b, n in [(0, 10, "< 10 s"), (10, 30, "10-30 s"), (30, 1e9, "30 s +")]:
            m = np.isfinite(T) & (T >= a) & (T < b)
            if m.sum() >= 20:
                r = me_all.err.to_numpy()[m].mean()
                lines.append(f"  {n:8s} {m.mean():4.0%} of moves, error rate {r:.1%} (n {m.sum()})")
                rows6.append(f"<tr><td>{n}</td><td>{m.mean():.0%}</td><td>{r:.1%}</td><td>{m.sum()}</td></tr>")
        body += ["<h2>6. Think time (you, whole history)</h2><table><tr><th>time spent</th><th>share of moves</th>"
                 "<th>error rate</th><th>n</th></tr>" + "".join(rows6) + "</table>"]

    # your worst balanced-endgame errors
    worst = me_all[me_all.err].sort_values("wp_loss", ascending=False).head(12)
    cards = []
    for i, r in worst.iterrows():
        b = chess.Board(r.fen)
        cards.append(position_card(r, note=f"{r.etype}; you moved a {r.moved}, best was a {r.best_piece} move "
                                           f"({b.san(chess.Move.from_uci(r.best_move))})"))
    body += ["<h2>Your worst balanced-endgame errors (whole history)</h2>", f"<div class='grid'>{''.join(cards)}</div>"]

    path = data_dir(cfg, "player") / "balanced_endgames.html"
    path.write_text(page("Balanced endgames", "".join(body)), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
