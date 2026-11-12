"""Openings: your repertoire vs the population, by engine and by what players at each level play.

Three cached stages (rerun skips finished ones; --rebuild redoes stage 2):
  1. Your opening tree from the chess.com archives: every position up to openings.max_ply
     (default 24 plies = move 12), with your move or the opponent's, result, opening name.
  2. Population "opening book" from the local Lichess dump (paths.lichess_dump): for every
     position in YOUR tree, how often each move is played and how it scores, per rating
     band of the mover (openings.bands, default 1600/1800/2000/2200/2400+), rapid, rated.
     Games are followed only while they stay in your tree (stop after openings.leave_k plies
     outside it). Also stores the ply-20/21 positions of openings.peer_sample peer-band games
     (population.rating_center +- halfwidth) as the baseline for "out of the opening".
     (The Lichess opening explorer API now needs a login token, so the dump is used instead.)
  3. Stockfish (resumable): every tree position reached >= openings.min_count times in your
     recent games (rating >= player.min_player_rating), valuing the top openings.multipv moves,
     every move you or your opponents played there, and moves with >= openings.pop_share at the
     peer band or 2400+; plus the out-of-the-opening positions (yours and peers').

Report (stdout + data/player/openings_report.html)
  A. Out of the opening: your win prob at move 10 (ply 20 as White / 21 as Black) vs peers;
     by opening; and how that eval converts into results (you and peers).
  B. Moves you play repeatedly (>= 2x) that lose >= openings.flag (5) points by the engine, ranked
     by count x loss, with the engine's top moves. Engine-only: what opponents play doesn't matter
     here (exploitative lines belong to the exploitation layer, which uses the book from stage 2).
  C. Must-know positions: only one good move (best beats 2nd by >= 10 points), reached >= 2x:
     how often you find it. Plus opponents' opening mistakes: do you punish them?
Covers the first 15 moves (max_ply 30): score accounting (15) shows moves ~10-15 are where you
net-lose points vs your opponents.

    python scripts/14_openings.py --config configs/v0.yaml
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.config import data_dir, load_config, resolve
from coach.mine import is_rapid, mover_score
from coach.openings import band_labels, evaluate, player_tree, san_tokens
from coach.pgn_stream import iter_raw_games
from coach.report import page, position_card

OPEN = {"max_ply": 30, "min_count": 3, "min_visits": 2, "flag": 0.05, "dump_games": 2_000_000, "bands": [1600, 1800, 2000, 2200, 2400, 4000],
        "nodes": 300000, "multipv": 3, "peer_sample": 3000, "out_ply": 20, "leave_k": 4, "pop_share": 0.05,
        "good": 0.03, "bad": 0.08, "critical_gap": 0.10}
# ---------------------------------------------------------------- stage 2
def population_book(cfg, oc, epds: set):
    fcfg, pop = cfg["filters"], cfg["population"]
    bands = oc["bands"]
    lo_p, hi_p = pop["rating_center"] - pop["rating_halfwidth"], pop["rating_center"] + pop["rating_halfwidth"]
    counts = defaultdict(lambda: [0.0] * (2 * (len(bands) - 1)))
    peers, n_seen, n_used, t0 = [], 0, 0, time.time()
    dump = resolve(cfg, cfg["paths"]["lichess_dump"])
    for headers, text in iter_raw_games(dump):
        n_seen += 1
        if n_seen % 1_000_000 == 0:
            print(f"  scanned {n_seen:,} games ({n_seen / (time.time() - t0):,.0f}/s), used {n_used:,}", flush=True)
        if "Rated" not in headers.get("Event", "") or headers.get("Variant", "Standard") != "Standard":
            continue
        if not is_rapid(headers.get("TimeControl"), fcfg):
            continue
        if headers.get("Termination", "Normal") not in ("Normal", "Time forfeit"):
            continue
        try:
            we, be = int(headers.get("WhiteElo", 0)), int(headers.get("BlackElo", 0))
        except ValueError:
            continue
        if min(we, be) < bands[0]:
            continue
        n_used += 1
        want_peer = len(peers) < oc["peer_sample"] and lo_p <= we <= hi_p and lo_p <= be <= hi_p
        ws = mover_score(headers.get("Result"), True)
        if np.isnan(ws):
            continue
        b, miss = chess.Board(), 0
        try:
            for ply, t in enumerate(san_tokens(text)):
                if want_peer and ply in (oc["out_ply"], oc["out_ply"] + 1):
                    peers.append({"color": "white" if b.turn else "black", "fen": b.fen(),
                                  "result": ws if b.turn else 1 - ws, "elo": we if b.turn else be})
                if ply >= oc["max_ply"]:
                    if not want_peer or ply > oc["out_ply"] + 1:
                        break
                    b.push_san(t)
                    continue
                epd = b.epd()
                elo = we if b.turn else be
                mv = b.push_san(t)
                if epd in epds:
                    miss = 0
                    k = int(np.searchsorted(bands, elo, side="right")) - 1
                    if 0 <= k < len(bands) - 1:
                        c = counts[(epd, mv.uci())]
                        c[2 * k] += 1
                        c[2 * k + 1] += ws if not b.turn else 1 - ws   # b.turn flipped: mover = not b.turn
                else:
                    miss += 1
                    if miss >= oc["leave_k"] and not want_peer:
                        break
        except ValueError:
            pass
        if n_used >= oc["dump_games"]:
            break
    labels = band_labels(bands)
    recs = [{"epd": e, "move": m, "band": labels[k], "n": c[2 * k], "score": c[2 * k + 1] / c[2 * k]}
            for (e, m), c in counts.items() for k in range(len(labels)) if c[2 * k] > 0]
    print(f"  dump: scanned {n_seen:,}, used {n_used:,} rapid games; {len(recs):,} (position, move, band) rows; "
          f"{len(peers):,} peer out-of-opening positions")
    return pd.DataFrame(recs), pd.DataFrame(peers)


# ---------------------------------------------------------------- stage 3
def run_engine(cfg, oc, items, workers):
    return evaluate(cfg, items, oc["nodes"], workers)   # deepest result per position, all node counts


# ---------------------------------------------------------------- analysis helpers
def boot_mean(x, B=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, 5), np.percentile(m, 95)


def band_view(book, epd, band):
    s = book.get((epd, band))
    return s if s is not None else pd.DataFrame(columns=["move", "n", "score"])


def good_rate(view, vals, best_wp, good):
    """Share of band games at this position whose move is within `good` of the best (unvalued moves = not good)."""
    if view.empty or view.n.sum() == 0:
        return np.nan
    ok = view.move.map(lambda m: m in vals and best_wp - vals[m] < good)
    return float(view.n[ok].sum() / view.n.sum())


def san(fen, uci):
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def top_move(view, fen):
    if view.empty:
        return "-"
    r = view.sort_values("n", ascending=False).iloc[0]
    return f"{san(fen, r.move)} {r.n / view.n.sum():.0%} (scores {r.score:.2f}, n {int(view.n.sum())})"


def card(fen, move, best, loss, note):
    r = pd.Series({"fen": fen, "move": move, "best_move": best, "best_line": [best], "played_line": [move],
                   "wp_loss": loss, "phase": "opening"})
    return position_card(r, note=note)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--rebuild", action="store_true", help="redo the dump pass (stage 2)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--nodes", type=int, default=None, help="search size (refine passes use more; deepest wins)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    oc = {**OPEN, **cfg.get("openings", {})}
    if args.nodes:
        oc["nodes"] = args.nodes
    cmp_min = cfg["player"]["min_player_rating"]
    out = data_dir(cfg, "openings")
    labels = band_labels(oc["bands"])
    pop = cfg["population"]
    peer_band = next((l for l, a in zip(labels, oc["bands"]) if a <= pop["rating_center"] < a + 200), labels[-2])
    strong = labels[-1]

    print("1. your opening tree")
    tree, outs = player_tree(cfg, oc)
    if tree.empty:
        sys.exit("no games found in data/player/archives; run 00_fetch_player.py first")
    recent = tree[tree.rating >= cmp_min]
    print(f"   {tree.game_id.nunique():,} games ({recent.game_id.nunique():,} at >= {cmp_min}), "
          f"{tree.epd.nunique():,} positions")

    print("2. population book from the Lichess dump")
    bp, pp = out / "book.parquet", out / "peer_out.parquet"
    if args.rebuild or not bp.exists():
        book_df, peers = population_book(cfg, oc, set(tree.epd))
        book_df.to_parquet(bp, index=False)
        peers.to_parquet(pp, index=False)
    book_df, peers = pd.read_parquet(bp), pd.read_parquet(pp)
    book = {k: g[["move", "n", "score"]] for k, g in book_df.groupby(["epd", "band"])}

    print("3. Stockfish")
    vc = recent.groupby("epd").size()
    hot = set(vc[vc >= oc["min_count"]].index)
    fen_of = dict(zip(tree.epd, tree.fen))
    moves_at = tree.groupby("epd").move.agg(lambda s: sorted(set(s))).to_dict()
    items = []
    for e in sorted(hot):
        mv = set(moves_at.get(e, []))
        for band in (peer_band, strong):
            v = band_view(book, e, band)
            if not v.empty:
                mv |= set(v.move[v.n / v.n.sum() >= oc["pop_share"]])
        items.append(("tree:" + e, fen_of[e], sorted(mv), oc["multipv"]))
    items += [("out:" + g, f, [], 1) for g, f in zip(outs.game_id, outs.fen)]
    items += [(f"peer:{i}", f, [], 1) for i, f in enumerate(peers.fen)]
    E = run_engine(cfg, oc, items, args.workers or cfg["engine"]["workers"])

    def wp_best(pid):
        r = E.get(pid)
        return max(r[1].values()) if r and r[1] else np.nan

    L, body = [], ["<h1>Openings</h1>",
                   f"<p class='muted'>Your chess.com rapid games, first {oc['max_ply'] // 2} moves, judged by the engine "
                   f"only (peers = Lichess rapid {peer_band}, used only as the out-of-opening baseline). Stockfish "
                   f"{oc['nodes']:,} nodes. Win-probability points.</p>"]

    # A. out of the opening
    outs["wp"] = [wp_best("out:" + g) for g in outs.game_id]
    peers["wp"] = [wp_best(f"peer:{i}") for i in range(len(peers))]
    ro = outs[outs.rating >= cmp_min]
    L.append(f"A. out of the opening (move {oc['out_ply'] // 2 + 1}, side to move = you), mean win prob:")
    rows = []
    items = {"out": {}, "costly": [], "must_know": [], "missed_punish": []}   # for the report (20)
    for col in ("white", "black"):
        a, p = ro[ro.color == col], peers[peers.color == col]
        lo, hi = boot_mean(a.wp)
        items["out"][col] = {"you": float(a.wp.mean()), "lo": float(lo), "hi": float(hi),
                             "peers": float(p.wp.mean()), "n": int(a.wp.notna().sum())}
        tag = "clear" if np.isfinite(lo) and (p.wp.mean() < lo or p.wp.mean() > hi) else "within noise"
        L.append(f"   as {col}: you {a.wp.mean():.3f} [{lo:.3f}, {hi:.3f}] (n {a.wp.notna().sum()}) vs peers "
                 f"{p.wp.mean():.3f} (n {p.wp.notna().sum()})  {tag}")
        rows.append(f"<tr><td>{col}</td><td>{a.wp.mean():.3f} [{lo:.3f}, {hi:.3f}]</td><td>{p.wp.mean():.3f}</td>"
                    f"<td>{tag}</td></tr>")
    body += ["<h2>A. Out of the opening</h2><table><tr><th>as</th><th>your win prob at move "
             f"{oc['out_ply'] // 2 + 1} (90% CI)</th><th>peers</th><th></th></tr>" + "".join(rows) + "</table>"]
    L.append("   how that converts into results (score by win prob out of the opening):")
    conv = []
    for a, b in [(0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
        m1 = (ro.wp >= a) & (ro.wp < b) & ro.result.notna()
        m2 = (peers.wp >= a) & (peers.wp < b) & peers.result.notna()
        if m1.sum() >= 10:
            L.append(f"     {a:.2f}-{min(b, 1):.2f}: you {ro.result[m1].mean():.2f} (n {m1.sum()}), "
                     f"peers {peers.result[m2].mean():.2f} (n {m2.sum()})")
            conv.append(f"<tr><td>{a:.2f}–{min(b, 1):.2f}</td><td>{ro.result[m1].mean():.2f} ({m1.sum()})</td>"
                        f"<td>{peers.result[m2].mean():.2f} ({m2.sum()})</td></tr>")
    body.append("<h3>Score by win prob out of the opening</h3><table><tr><th>win prob</th><th>your score (n)</th>"
                "<th>peers (n)</th></tr>" + "".join(conv) + "</table>")
    fam = ro.groupby(["color", "family"]).agg(n=("wp", "size"), wp=("wp", "mean"), score=("result", "mean"))
    fam = fam[fam.n >= 8].sort_values("n", ascending=False)
    L.append("   by opening (>= 8 games): games | win prob out of the opening | score")
    frows = []
    for (col, f), r in fam.head(20).iterrows():
        pw = peers[peers.color == col].wp.mean()
        L.append(f"     {col:5s} {f[:40]:40s} {int(r.n):4d} | {r.wp:.3f} ({r.wp - pw:+.3f} vs peers) | {r.score:.2f}")
        frows.append(f"<tr><td>{col}</td><td>{f}</td><td>{int(r.n)}</td><td>{r.wp:.3f}</td><td>{r.wp - pw:+.3f}</td>"
                     f"<td>{r.score:.2f}</td></tr>")
    body.append("<h3>By opening</h3><table><tr><th>as</th><th>opening</th><th>games</th><th>win prob out of opening"
                "</th><th>vs peers (all openings)</th><th>score</th></tr>" + "".join(frows) + "</table>")

    # B. frequent inaccurate moves
    mine = recent[recent.mine]
    g = mine.groupby(["epd", "move"]).agg(c=("game_id", "size"), score=("result", "mean"),
                                          fen=("fen", "first"), path=("path", "first")).reset_index()
    recs = []
    for r in g.itertuples():
        ev = E.get("tree:" + r.epd)
        if not ev or r.move not in ev[1] or r.c < 2:
            continue
        vals = ev[1]
        best = max(vals, key=vals.get)          # cached results from before the engine fix may differ
        loss = vals[best] - vals[r.move]
        if loss >= oc["flag"]:
            recs.append((r.c * loss, r, best, vals, loss))
    recs.sort(key=lambda x: -x[0])
    L.append(f"B. moves you play repeatedly that lose >= {100 * oc['flag']:.0f} points by the engine (ranked by count x loss):")
    cards = []
    for tot, r, best, vals, loss in recs[:15]:
        line = f"{r.path} [{san(r.fen, r.move)}]".strip()
        alts = ", ".join(f"{san(r.fen, m)} {100 * (vals[best] - v):.1f}" for m, v in list(vals.items())[:3])
        L.append(f"   {line}")
        L.append(f"      you {r.c}x, loses {100 * loss:.1f} (your score {r.score:.2f}); engine: {san(r.fen, best)} "
                 f"(top moves, points below best: {alts})")
        cards.append(card(r.fen, r.move, best, loss,
                          f"{line} · you played this {r.c}x (score {r.score:.2f}) · engine top: {alts}"))
        items["costly"].append({"line": line, "fen": r.fen, "move": r.move, "best": best, "loss": float(loss),
                                "count": int(r.c), "score": float(r.score), "alts": alts})
    body += ["<h2>B. Moves you play a lot that cost win probability</h2>",
             "<p class='muted'>Red = your move, green = Stockfish's. Ranked by how often × how much.</p>",
             f"<div class='grid'>{''.join(cards)}</div>"]

    # C. must-know positions + punishing opponents' mistakes
    crit = []
    for e in hot:
        ev = E.get("tree:" + e)
        if not ev or len(ev[1]) < 2:
            continue
        vs = sorted(ev[1].values(), reverse=True)
        if vs[0] - vs[1] < oc["critical_gap"]:
            continue
        visits = mine[mine.epd == e]
        if len(visits) < oc["min_visits"]:
            continue
        hit = visits.move.map(lambda m: m in ev[1] and vs[0] - ev[1][m] < oc["good"]).mean()
        gr = good_rate(band_view(book, e, peer_band), ev[1], vs[0], oc["good"])
        crit.append({"epd": e, "fen": fen_of[e], "visits": len(visits), "hit": hit, "peer": gr, "best": ev[0],
                     "path": visits.path.iloc[0], "usual": visits.move.mode().iat[0], "vals": ev[1]})
    C = pd.DataFrame(crit)
    cards = []
    if len(C):
        w = C.visits
        L.append(f"C. must-know positions (one good move, gap >= {100 * oc['critical_gap']:.0f}): {len(C)} positions, "
                 f"{int(w.sum())} visits; you find it {np.average(C.hit, weights=w):.0%} (visit-weighted)")
        C["missed"] = C.visits * (1 - C.hit)
        for r in C.sort_values("missed", ascending=False).head(10).itertuples():
            L.append(f"   {r.path} -> {san(r.fen, r.best)}: you {r.hit:.0%} of {r.visits}")
            loss = max(r.vals.values()) - r.vals.get(r.usual, min(r.vals.values()))
            cards.append(card(r.fen, r.usual, r.best, loss,
                              f"{r.path} · only good move {san(r.fen, r.best)} · you found it {r.hit:.0%} of "
                              f"{r.visits}"))
            items["must_know"].append({"line": r.path, "fen": r.fen, "move": r.usual, "best": r.best,
                                       "loss": float(loss), "hit": float(r.hit), "visits": int(r.visits)})
    else:
        L.append("C. must-know positions: none reached often enough")
    # opponents' frequent mistakes and your reply
    opp = recent[~recent.mine]
    nxt = recent[recent.mine].set_index(["game_id", "ply"])
    pun = []
    for r in opp.itertuples():
        ev = E.get("tree:" + r.epd)
        if not ev or r.move not in ev[1]:
            continue
        if max(ev[1].values()) - ev[1][r.move] < oc["bad"]:
            continue
        try:
            me = nxt.loc[(r.game_id, r.ply + 1)]
        except KeyError:
            continue
        ev2 = E.get("tree:" + me.epd)
        if not ev2 or me.move not in ev2[1]:
            continue
        b2 = max(ev2[1].values())
        pun.append({"epd2": me.epd, "fen": me.fen, "found": b2 - ev2[1][me.move] < oc["good"], "best": ev2[0],
                    "move": me.move, "loss": b2 - ev2[1][me.move], "path": me.path,
                    "peer": good_rate(band_view(book, me.epd, peer_band), ev2[1], b2, oc["good"])})
    Pn = pd.DataFrame(pun)
    if len(Pn):
        L.append(f"   opponents' opening mistakes (>= {100 * oc['bad']:.0f} points) you faced: {len(Pn)}; you punished "
                 f"{Pn.found.mean():.0%}")
        miss = Pn[~Pn.found].groupby("epd2").agg(n=("found", "size"), fen=("fen", "first"), best=("best", "first"),
                                                 move=("move", "first"), loss=("loss", "mean"), path=("path", "first"),
                                                 peer=("peer", "first"))
        for e, r in miss.sort_values("n", ascending=False).head(5).iterrows():
            L.append(f"     missed {int(r.n)}x: {r.path} -> {san(r.fen, r.best)} (you played {san(r.fen, r.move)})")
            items["missed_punish"].append({"line": r.path, "fen": r.fen, "move": r.move, "best": r.best,
                                           "loss": float(r.loss), "count": int(r.n)})
            cards.append(card(r.fen, r.move, r.best, r.loss, f"{r.path} · opponent erred; punish with "
                              f"{san(r.fen, r.best)} · you missed it {int(r.n)}x"))
    body += ["<h2>C. Must-know positions and punishing opening mistakes</h2>",
             f"<div class='grid'>{''.join(cards)}</div>"]

    print("\n".join(L))
    if len(Pn):
        items["punish_rate"] = float(Pn.found.mean())
        items["punish_n"] = int(len(Pn))
    (out / "report_items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    path = data_dir(cfg, "player") / "openings_report.html"
    path.write_text(page("Openings", "".join(body)), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
