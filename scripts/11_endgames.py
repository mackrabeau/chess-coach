"""Endgames, judged by outcomes and by perfect play rather than average loss per move.

Average loss per move flatters anyone who plays lost endgames out (moves in lost positions
cost almost nothing in win probability) and is skewed by sampling (peers contribute at most
~12 positions per game, you contribute every move). So:

A. Practical endgames (outcome-based). One random endgame position per game, for you and
   for peers, grouped by the engine's win probability for the mover at that point; then the
   final game result. Conversion = win rate from winning positions; holding = draw-or-better
   rate from worse / equal ones. Needs results (00 re-run, 01b for peers).
B. Counterplay (you only; peers have ~1 analyzed position per game). After your first
   winning endgame position in a game: how often does the eval later fall back toward equal?
C. Theoretical endgames (tablebase). Positions with few enough pieces for your local Syzygy
   tables: only moves made from a won or drawn position count (so playing on when lost
   doesn't matter), and an error is throwing the result away (win -> draw/loss, draw -> loss).
   Grouped by material (pawn, rook, queen, minor-piece endings), you vs peers, with examples.

    python scripts/11_endgames.py --config configs/v0.yaml

Output: stdout + findings/endgames.json (the report's Endgame tab, "In depth")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess.syzygy
import numpy as np
import pandas as pd

from coach.config import data_dir, load_config, resolve
from coach.endgame_features import material_class
from coach.handcrafted import phase
from coach.outcomes import mover_results
from coach.findings import detail, example, table, write
from coach.results import load_analyzed

EVAL_BINS = [(0.0, 0.2, "losing"), (0.2, 0.4, "worse"), (0.4, 0.6, "equal"),
             (0.6, 0.8, "better"), (0.8, 1.01, "winning")]


def prep(cfg, which):
    df = load_analyzed(cfg, which)
    df["phase"] = [phase(chess.Board(f)) for f in df.fen]
    df["score"] = mover_results(cfg, df, which)
    return df


def one_per_game(df, seed=0):
    """One random endgame position per game (matches the peers' sampling)."""
    e = df[df.phase == "endgame"]
    return e.groupby("game_id", group_keys=False).sample(1, random_state=seed)


def outcome_table(me, pop, B, rng):
    """Per eval bin: n, mean score, win / draw / loss rates; you vs peers, bootstrap CI on score."""
    rows, lines = [], []
    for lo, hi, name in EVAL_BINS:
        m = me[(me.wp_best >= lo) & (me.wp_best < hi) & me.score.notna()]
        p = pop[(pop.wp_best >= lo) & (pop.wp_best < hi) & pop.score.notna()]
        if len(m) < 10 or len(p) < 10:
            lines.append(f"  {name:8s} too few (you {len(m)}, peers {len(p)})")
            continue
        s = m.score.to_numpy()
        boots = [s[rng.integers(0, len(s), len(s))].mean() for _ in range(B)]
        lo_s, hi_s = np.percentile(boots, [5, 95])
        sp = p.score.mean()
        flag = "clear" if (lo_s > sp or hi_s < sp) else "within noise"
        wdl = lambda x: ((x == 1).mean(), (x == 0.5).mean(), (x == 0).mean())
        wm, dm, lm = wdl(s)
        wp, dp, lp = wdl(p.score.to_numpy())
        rows.append([name, len(m), f"{s.mean():.2f} [{lo_s:.2f}, {hi_s:.2f}]", f"{sp:.2f}", flag,
                     f"{wm:.0%} / {dm:.0%} / {lm:.0%}", f"{wp:.0%} / {dp:.0%} / {lp:.0%}", len(p)])
        lines.append(f"  {name:8s} you {s.mean():.2f} [{lo_s:.2f}, {hi_s:.2f}] (W/D/L {wm:.0%}/{dm:.0%}/{lm:.0%}, "
                     f"n {len(m)})  peers {sp:.2f} (W/D/L {wp:.0%}/{dp:.0%}/{lp:.0%}, n {len(p)})  {flag}")
    header = ["engine says (side to move)", "your games", "your final score (90% CI)", "peers", "vs peers",
              "you W / D / L", "peers W / D / L", "peer games"]
    return table(header, rows), lines


def counterplay(df):
    """Your games: after the first winning endgame position, did the eval come back down?"""
    out = []
    for g, sub in df.sort_values("ply").groupby("game_id"):
        e = sub[(sub.phase == "endgame")]
        win = e[e.wp_best >= 0.8]
        if win.empty:
            continue
        after = sub[sub.ply >= win.ply.iloc[0]]
        out.append({"game_id": g, "date": sub.date.iloc[0], "min_after": after.wp_best.min(),
                    "score": sub.score.iloc[0]})
    o = pd.DataFrame(out)
    return o[o.score.notna()] if len(o) else o     # only games with a known result


def wdl3(v: int) -> int:
    """Syzygy WDL -> practical result: 2 win, 0 draw (incl. 50-move-rule cursed / blessed), -2 loss."""
    return 2 if v == 2 else (-2 if v == -2 else 0)


def theory(df, tb, with_moves: bool = True):
    """Tablebase verdict before and after each played move, for positions the tables cover."""
    rows = []
    for i in range(len(df)):
        b = chess.Board(df.fen.iat[i])
        if chess.popcount(b.occupied) > 7:
            continue
        try:
            before = wdl3(tb.probe_wdl(b))
            b.push_uci(df.move.iat[i])
            after = -wdl3(tb.probe_wdl(b)) if not b.is_checkmate() else 2
            b.pop()
        except (chess.syzygy.MissingTableError, KeyError, IndexError):
            continue
        if before == -2:
            continue           # already lost: nothing to throw away
        good = []
        for m in (b.legal_moves if with_moves else []):
            b.push(m)
            try:
                v = 2 if b.is_checkmate() else -wdl3(tb.probe_wdl(b))
            except (chess.syzygy.MissingTableError, KeyError):
                v = None
            b.pop()
            if v is not None and v >= before:
                good.append(b.san(m))
        rows.append({"row": i, "class": material_class(b), "before": before, "after": after,
                     "error": after < before, "good_moves": good})
    return pd.DataFrame(rows)


def peer_small(cfg, max_pieces: int = 5) -> pd.DataFrame:
    """All stored peer positions (positions_sae, ~1.5M) with at most max_pieces pieces.
    The tablebase needs no engine, so the peer baseline can use the whole pool."""
    sae = pd.read_parquet(data_dir(cfg, "population") / "positions_sae.parquet",
                          columns=["pos_id", "fen", "move", "mover_rating"])
    board = sae.fen.str.split(" ", n=1).str[0]
    n = board.str.count(r"[pnbrqkPNBRQK]")
    return sae[n <= max_pieces].reset_index(drop=True)


def wilson(k: int, n: int, z: float = 1.645):
    """90% Wilson interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    return mid - c, mid + c


def theory_table(tm, tp):
    rows, lines = [], []
    classes = sorted(set(tm["class"]) | set(tp["class"])) if len(tm) and len(tp) else []
    for c in ["all"] + classes:
        m = tm if c == "all" else tm[tm["class"] == c]
        p = tp if c == "all" else tp[tp["class"] == c]
        if len(m) == 0:
            continue
        rm, rp = m.error.mean(), (p.error.mean() if len(p) else float("nan"))
        lo_r, hi_r = wilson(int(m.error.sum()), len(m))
        clear = len(p) >= 30 and (lo_r > rp or hi_r < rp)
        wm = m[m.before == 2]
        verdict = ("worse" if rm > rp else "better") if clear else "same"
        rows.append([c, len(m), f"{rm:.1%} [{lo_r:.1%}, {hi_r:.1%}]", len(p), f"{rp:.1%}", verdict, len(wm),
                     f"{(wm.after < 2).mean() if len(wm) else float('nan'):.1%}"])
        lines.append(f"  {c:20s} you: {m.error.sum()}/{len(m)} moves threw the result ({rm:.1%} [{lo_r:.1%}, {hi_r:.1%}])"
                     f"{'  CLEAR' if clear else ''}   "
                     f"peers: {int(p.error.sum()) if len(p) else 0}/{len(p)} ({rp:.1%})   "
                     f"from won positions you let it slip {(wm.after < 2).mean() if len(wm) else float('nan'):.1%} (n {len(wm)})")
    header = ["endgame type", "your moves (won/drawn)", "threw the result", "peer moves", "peers threw it",
              "vs peers", "your moves from won", "win let slip"]
    return table(header, rows), lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    B = cfg["report"]["bootstrap"]
    rng = np.random.default_rng(0)
    cmp_min = cfg["player"]["min_player_rating"]
    me_all, pop = prep(cfg, "player"), prep(cfg, "population")
    me = me_all[me_all.mover_rating >= cmp_min]
    items, out_lines = [], []

    # A. practical endgames
    if me.score.notna().any() and pop.score.notna().any():
        t, l = outcome_table(one_per_game(me), one_per_game(pop), B, rng)
        outcome_ev = t
        out_lines += ["A. practical endgames (final score from one endgame position per game):"] + l
        # contested-only loss per move, per game averaged (resign-proof version of the old metric)
        def contested(df):
            e = df[(df.phase == "endgame") & (df.wp_best >= 0.2) & (df.wp_best <= 0.8)]
            return 100 * e.groupby("game_id").wp_loss.mean().to_numpy()   # per game, so long games don't dominate
        gm, gp = contested(me), contested(pop)
        boots = [gm[rng.integers(0, len(gm), len(gm))].mean() for _ in range(B)]
        lo_c, hi_c = np.percentile(boots, [5, 95])
        verdict = "clear" if (lo_c > gp.mean() or hi_c < gp.mean()) else "within noise"
        out_lines.append(f"  contested endgame positions only (win prob 0.2-0.8), loss per 100 moves averaged per game: "
                         f"you {gm.mean():.2f} [{lo_c:.2f}, {hi_c:.2f}] ({len(gm)} games) vs peers {gp.mean():.2f} "
                         f"({len(gp)} games)  {verdict}")
        items.append(detail("endgame:outcomes", "endgame", "What happens from an endgame position",
                            "One random endgame position per game, grouped by how good Stockfish says it is, then "
                            "the final result: the winning row is converting, the worse/equal rows are holding. "
                            "Immune to resigning habits and game length. In balanced endgame positions only, you lose "
                            f"{gm.mean():.2f} win-probability points per 100 moves [{lo_c:.2f}, {hi_c:.2f}] vs peers "
                            f"{gp.mean():.2f} ({verdict}).", outcome_ev))
    else:
        out_lines.append("A. skipped: no game results yet (re-run 00; run 01b for peers)")

    # B. counterplay (you, whole history)
    if me_all.score.notna().any():
        cpl = counterplay(me_all)
        if len(cpl):
            cpl["year"] = cpl.date.astype(str).str[:4]
            lines = ["B. counterplay after reaching a winning endgame (you, whole history):"]
            rows = []
            for yr, sub in [("all", cpl)] + list(cpl.groupby("year")):
                if len(sub) < 10:
                    continue
                back = (sub.min_after < 0.6).mean()
                lines.append(f"  {yr}: {len(sub)} games; eval fell back below 60% in {back:.0%}; "
                             f"final score {sub.score.mean():.2f} (won {(sub.score == 1).mean():.0%})")
                rows.append([yr, len(sub), f"{back:.0%}", f"{sub.score.mean():.2f}", f"{(sub.score == 1).mean():.0%}"])
            out_lines += lines
            items.append(detail("endgame:counterplay", "endgame", "After you reach a winning endgame",
                                "Your games, by year: how often the position slipped back below 60% after you first "
                                "had a winning endgame, and how those games ended.",
                                table(["year", "games", "fell back below 60%", "final score", "won"], rows)))

    # C. theoretical endgames
    tb_path = cfg.get("endgames", {}).get("syzygy_path")
    if tb_path and resolve(cfg, tb_path).exists():
        with chess.syzygy.open_tablebase(str(resolve(cfg, tb_path))) as tb:
            tm_all = theory(me_all, tb)
            tp = theory(peer_small(cfg), tb, with_moves=False)   # every stored peer position the tables cover, not just the engine sample
        tm = tm_all[me_all.mover_rating.to_numpy()[tm_all.row.to_numpy()] >= cmp_min] if len(tm_all) else tm_all
        t, l = theory_table(tm, tp)
        t2, l2 = theory_table(tm_all, tp)
        out_lines += [f"C. theoretical endgames, your games >= {cmp_min} vs peers:"] + l
        out_lines += ["   whole history:"] + l2
        exs = []
        if len(tm_all):
            errs = tm_all[tm_all.error].copy()
            errs["severity"] = errs.before - errs.after
            errs = errs.sort_values("severity", ascending=False).head(9)
            names = {2: "win", 0: "draw", -2: "loss"}
            for r in errs.itertuples():
                x = me_all.iloc[r.row]
                exs.append(example(x.fen, x.best_line, x.played_line,
                                   f"{x.date} · tablebase: {names[r.before]} → {names[r.after]}; moves that keep it: "
                                   f"{', '.join(r.good_moves[:6]) or '—'}", x.pos_id))
        items.append(detail("endgame:tablebase", "endgame", "Theoretical endgames (tablebase)",
                            "Positions with few enough pieces for the tablebase. Only moves from won or drawn positions "
                            f"count; an error throws the result away. Table: your games at ≥ {cmp_min}. Boards: your "
                            "worst result-throwing moves (whole history).", t, exs))
        items.append(detail("endgame:tablebase-all", "endgame", "Theoretical endgames, your whole history",
                            "Same table over all your games (more data, includes your lower-rated years).", t2))
    else:
        out_lines.append("C. skipped: set endgames.syzygy_path to your Syzygy folder (3-4-5 WDL + DTZ files)")

    write(cfg, "endgames", items)
    print("\n".join(out_lines))
    print(f"{len(items)} in-depth items -> findings/endgames.json")


if __name__ == "__main__":
    main()
