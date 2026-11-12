"""Opening traps (exploitation layer, exploratory): lines your opponents at their level fall for.

Not part of the main opening analysis (14 is engine-only: play solid). This asks a different
question: where can you give the opponent chances to go wrong without giving anything up?

  A. Positions you already reach where opponents go wrong (no new engine work):
     opponent-to-move positions in your tree (visits >= traps.min_count in recent games), where
     Lichess players in your opponents' band (book from 14) play a move that loses
     >= traps.bad (8 points) with total probability >= traps.min_err_mass. "Steer here."
  B. Trap moves you could play: for your-move positions in your tree, Stockfish candidates
     within practical.delta of the best; for each, the opponent's reply distribution
       - from the book (their band) when >= traps.min_book games reached that position: real
         error rates, so traps everyone at that level knows (e.g. Fried Liver lines at 1900)
         score low automatically;
       - otherwise from Maia at your opponents' typical level (lines too rare for the book).
     EV(m) = sum_r P(r) V(m, r) + (1 - sum P) V*(m)   (unlisted = they find the best reply)
     trap gain = EV(trap move) - EV(Stockfish's move), objective cost = Stockfish wp difference.
     Ranked by visits x gain. Same machinery as 12 (validated there: Maia's reply
     probabilities matched realized trap value), applied to your opening tree.

Assumes you then punish the mistake (no follow-up model here; see 13 for that discount).

    python scripts/18_opening_traps.py --config configs/v0.yaml       (needs 14 to have run)

Output: stdout + data/player/opening_traps.html + data/openings/traps.parquet
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.config import data_dir, load_config, resolve
from coach.difficulty import player_maia_elo
from coach.engine import Analyzer
from coach.maia import Maia, maia_lists
from coach.openings import band_labels, opening_engine, player_tree
from coach.report import page, position_card

TRAPS = {"min_count": 2, "bad": 0.08, "min_err_mass": 0.15, "min_book": 30, "multipv": 5,
         "nodes_root": 1000000, "nodes_reply": 300000, "reply_mass": 0.9, "max_replies": 6, "min_gain": 0.02}
OPEN_DEFAULTS = {"max_ply": 30, "out_ply": 20, "bands": [1600, 1800, 2000, 2200, 2400, 4000]}

_an = None
_tc = {}


def _init(ecfg, sf, tc):
    global _an, _tc
    _an, _tc = Analyzer(ecfg, sf), tc


def _cands(item):
    epd, fen = item
    try:
        c, best = _an.candidates(fen, None, _tc["multipv"], _tc["nodes_root"], _tc["delta"])
        return epd, c, best
    except Exception as e:
        print(f"  candidates failed: {e}", flush=True)
        return epd, None, None


def _replies(item):
    key, fen, move, replies = item
    try:
        return key, _an.reply_values(fen, move, replies, _tc["nodes_reply"])
    except Exception as e:
        print(f"  replies failed: {e}", flush=True)
        return key, None


def pool_map(fn, items, cfg, tc, workers, label):
    out, t0 = [], time.time()
    sf = str(resolve(cfg, cfg["paths"]["stockfish"]))
    with mp.Pool(workers, initializer=_init, initargs=(dict(cfg["engine"]), sf, tc)) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, items, chunksize=2), 1):
            out.append(r)
            if i % 100 == 0 or i == len(items):
                el = time.time() - t0
                print(f"  {label}: {i:,}/{len(items):,}  ETA {el / i * (len(items) - i) / 60:.0f} min", flush=True)
    return out


def san(fen, uci):
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def history(path_san: str) -> list[str]:
    """FENs before each move of a SAN path (oldest first), for Maia's history input."""
    b, out = chess.Board(), []
    for t in path_san.split():
        out.append(b.fen())
        b.push_san(t)
    return out


def card(fen, move, best, loss, note):
    r = pd.Series({"fen": fen, "move": move, "best_move": best, "best_line": [best], "played_line": [move],
                   "wp_loss": loss, "phase": "opening"})
    return position_card(r, note=note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = {**TRAPS, **cfg.get("traps", {}), "delta": cfg["practical"]["delta"]}
    oc = {**OPEN_DEFAULTS, **cfg.get("openings", {})}
    workers = args.workers or cfg["engine"]["workers"]
    cmp_min = cfg["player"]["min_player_rating"]
    od = data_dir(cfg, "openings")
    if not (od / "book.parquet").exists():
        sys.exit("run 14_openings.py first (it builds the book from the Lichess dump)")

    tree, _ = player_tree(cfg, oc)
    recent = tree[tree.rating >= cmp_min]
    # your opponents' level, on the Lichess/Maia scale
    pos = pd.read_parquet(data_dir(cfg, "player") / "positions.parquet", columns=["game_id", "opp_rating", "mover_rating"])
    pos = pos[pos.mover_rating >= cmp_min]
    opp_elo = float(np.median(player_maia_elo(pos.opp_rating.to_numpy(float), cfg["maia"], pos)))
    labels = band_labels(oc["bands"])
    opp_band = next((l for l, a, b in zip(labels, oc["bands"][:-1], oc["bands"][1:]) if a <= opp_elo < b), labels[-1])
    print(f"your opponents (games >= {cmp_min}): ~{opp_elo:.0f} on the Lichess/Maia scale -> book band {opp_band}")
    book = pd.read_parquet(od / "book.parquet")
    book = {e: g[["move", "n", "score"]] for e, g in book[book.band == opp_band].groupby("epd")}
    E = {p[5:]: v for p, (_, v) in opening_engine(cfg).items() if p.startswith("tree:")}
    L, body = [], ["<h1>Opening traps (exploratory)</h1>",
                   f"<p class='muted'>Exploitation layer, not the main analysis. Opponents modelled at ~{opp_elo:.0f} "
                   f"(book band {opp_band}; Maia where the book is thin). Gains assume you punish the mistake.</p>"]

    # A. where opponents already go wrong
    vc = recent.groupby("epd").agg(n=("game_id", "size"), mine=("mine", "first"), fen=("fen", "first"),
                                   path=("path", "first"))
    rows = []
    for e, r in vc[(vc.n >= tc["min_count"]) & ~vc.mine].iterrows():
        v, vals = book.get(e), E.get(e)
        if v is None or vals is None or v.n.sum() < tc["min_book"]:
            continue
        best = max(vals.values())
        share = v.assign(p=v.n / v.n.sum(), loss=v.move.map(lambda m: best - vals[m] if m in vals else np.nan))
        bad = share[share.loss >= tc["bad"]]
        if bad.p.sum() >= tc["min_err_mass"]:
            top = bad.sort_values("p", ascending=False).iloc[0]
            rows.append({"epd": e, "fen": r.fen, "path": r.path, "visits": int(r.n), "err_mass": bad.p.sum(),
                         "exp_loss": float((bad.p * bad.loss).sum()), "trap": top.move, "trap_p": top.p,
                         "trap_loss": top.loss, "book_n": int(v.n.sum()),
                         "best": max(vals, key=vals.get)})
    A = pd.DataFrame(rows)
    A.to_parquet(od / "traps_A.parquet", index=False)
    cards = []
    L.append(f"A. positions you reach where {opp_band} players often go wrong (>= {100 * tc['bad']:.0f} points):")
    if len(A):
        A["value"] = A.visits * A.exp_loss
        for r in A.sort_values("value", ascending=False).head(12).itertuples():
            L.append(f"   {r.path}: {r.err_mass:.0%} go wrong (book n {r.book_n}); most common: "
                     f"{san(r.fen, r.trap)} {r.trap_p:.0%} (loses {100 * r.trap_loss:.0f}); you reached it {r.visits}x")
            cards.append(card(r.fen, r.trap, r.best, r.trap_loss,
                              f"{r.path} · {r.err_mass:.0%} of {opp_band} go wrong · common error "
                              f"{san(r.fen, r.trap)} ({r.trap_p:.0%}) · you reached it {r.visits}x"))
    else:
        L.append("   none (needs 14's engine values for the opponents' moves and >= min_book book games)")
    body += ["<h2>A. Positions you already reach where opponents often go wrong</h2>",
             "<p class='muted'>Red = the opponents' common mistake, green = the right move. Know the punishment.</p>",
             f"<div class='grid'>{''.join(cards)}</div>"]

    # B. trap moves you could play
    mine = vc[(vc.n >= tc["min_count"]) & vc.mine]
    print(f"B. {len(mine)} of your positions: Stockfish candidates")
    res = pool_map(_cands, list(zip(mine.index, mine.fen)), cfg, tc, workers, "candidates")
    cand = {e: (c, b) for e, c, b in res if c}
    maia = Maia(cfg["maia"]["model"])
    items, meta = [], []
    for e, (c, best) in cand.items():
        r = mine.loc[e]
        hist = history(r.path)
        for m, wp in c:
            b = chess.Board(r.fen)
            b.push_uci(m)
            v = book.get(b.epd())
            if v is not None and v.n.sum() >= tc["min_book"]:
                dist = list(zip(v.move, v.n / v.n.sum()))
                dist = sorted(dist, key=lambda x: -x[1])[: tc["max_replies"]]
                src = f"book n {int(v.n.sum())}"
            else:
                dist, src = None, "Maia"
            meta.append({"epd": e, "fen": r.fen, "path": r.path, "visits": int(r.n), "move": m, "wp": wp,
                         "is_best": m == best, "dist": dist, "src": src, "fen_after": b.fen(),
                         "hist": (hist + [r.fen])[-7:]})
    need = [k for k, d in enumerate(meta) if d["dist"] is None]
    if need:
        lists = maia_lists(maia, [(meta[k]["fen_after"], meta[k]["hist"], opp_elo, opp_elo, []) for k in need],
                           tc["reply_mass"], tc["max_replies"])
        for k, (lst, _) in zip(need, lists):
            meta[k]["dist"] = lst
    items = [(k, d["fen"], d["move"], [m for m, _ in d["dist"]]) for k, d in enumerate(meta)]
    rv = dict(pool_map(_replies, items, cfg, tc, workers, "reply values"))
    for k, d in enumerate(meta):
        r = rv.get(k)
        if r is None:
            d["ev"] = np.nan
            continue
        rstar, vstar, vals = r
        listed = [(m, p) for m, p in d["dist"] if m in vals or m == rstar]
        d["ev"] = sum(p * vals.get(m, vstar) for m, p in listed) + max(0.0, 1 - sum(p for _, p in listed)) * vstar
        d["vstar"] = vstar
        # the bait: the likely reply (>= 5%) that is best for you (values are your win prob)
        likely = [(vals.get(m, vstar), m, p) for m, p in listed
                  if p >= 0.05 and vals.get(m, vstar) - vstar >= 0.03]     # a real mistake, not a normal move
        top = max(likely, default=(vstar, None, 0.0))
        d["bait"], d["bait_p"], d["bait_v"] = top[1], top[2], top[0]
    T = pd.DataFrame([{k: v for k, v in d.items() if k not in ("dist", "hist")} for d in meta])
    T.to_parquet(od / "traps.parquet", index=False)
    out = []
    for e, g in T.dropna(subset=["ev"]).groupby("epd"):
        sf = g[g.is_best]
        if sf.empty:
            continue
        sf = sf.iloc[0]
        tr = g.loc[g.ev.idxmax()]
        if tr.move == sf.move or tr.ev - sf.ev < tc["min_gain"]:
            continue
        out.append({**tr.to_dict(), "sf_move": sf.move, "gain": tr.ev - sf.ev, "cost": sf.wp - tr.wp})
    Bt = pd.DataFrame(out)
    cards = []
    L.append(f"B. trap moves within {100 * tc['delta']:.0f} points of the best, gaining >= {100 * tc['min_gain']:.0f} "
             f"in expectation: {len(Bt)} positions")
    if len(Bt):
        Bt["value"] = Bt.visits * Bt.gain
        for r in Bt.sort_values("value", ascending=False).head(12).itertuples():
            L.append(f"   {r.path} [{san(r.fen, r.move)} instead of {san(r.fen, r.sf_move)}]: expected +{100 * r.gain:.1f}, "
                     f"cost {100 * r.cost:.1f}; bait {san(r.fen_after, r.bait) if isinstance(r.bait, str) else '-'} "
                     f"({r.bait_p:.0%}, {r.src}); you reached it {r.visits}x")
            cards.append(card(r.fen, r.move, r.sf_move, r.cost,
                              f"{r.path} · trap {san(r.fen, r.move)} (red) vs Stockfish {san(r.fen, r.sf_move)} (green) · "
                              f"expected +{100 * r.gain:.1f}, cost {100 * r.cost:.1f} · they play "
                              f"{san(r.fen_after, r.bait) if isinstance(r.bait, str) else '-'} {r.bait_p:.0%} ({r.src})"))
    body += ["<h2>B. Trap moves you could play</h2>",
             "<p class='muted'>Red = trap move, green = Stockfish's. 'they play' = the opponents' most damaging likely "
             "reply and how often; book = real Lichess games at their level, Maia = model for rarer lines.</p>",
             f"<div class='grid'>{''.join(cards)}</div>"]
    print("\n".join(L))
    path = data_dir(cfg, "player") / "opening_traps.html"
    path.write_text(page("Opening traps", "".join(body)), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
