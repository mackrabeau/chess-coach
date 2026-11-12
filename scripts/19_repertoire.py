"""Your lines, extended: the engine-best repertoire grown from the openings you actually play.

Engine-only (main analysis, not exploitation): even where your moves are fine, show where they can
be better and what to play next against each common reply.

Per colour, a tree from the start position:
  your move   Stockfish at repertoire.nodes (default 1M; deeper cached results are reused). The
              recommendation keeps YOUR usual move when it is within repertoire.also_fine (2 points)
              of the best (less to relearn); otherwise the engine's best. Other moves within
              also_fine are listed as "also fine"; your usual move's cost is shown when it's worse.
  their move  replies covering repertoire.reply_mass (90%) of what players in your opponents' band
              play (Lichess book from 14, >= min_book games), else Maia at your opponents' level;
              replies under min_reply_p (5%) are dropped. Replies that lose >= repertoire.bad (8
              points) compared with their best listed reply are flagged, with the punishment.
Grown most-likely-first to move ~15 (max_ply 30), stopping when a line's probability < min_prob (0.2%;
probabilities multiply along a line, so 1% stopped main lines around move 8),
within an engine budget per colour (resumable; the cache is shared with 14).

Output:
  data/player/repertoire_white.pgn, repertoire_black.pgn (Lichess studies; one chapter per first
  opponent decision, e.g. 1.e4 c5 / 1.e4 e5 / ...; split at 64 chapters)
  data/player/repertoire.html (decision points: where your usual move differs, ranked by how often x cost)

    python scripts/19_repertoire.py --config configs/v0.yaml
    python scripts/19_repertoire.py --config configs/v0.yaml --band 2200-2399   # replies at a target level
"""
from __future__ import annotations

import argparse
import html
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess.pgn
import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.difficulty import player_maia_elo
from coach.findings import elo_gain, example, finding, write as write_findings
from coach.maia import Maia, maia_lists
from coach.openings import band_labels, evaluate, opening_engine, player_tree
from coach.report import page, position_card
from coach.study import chapter, write_studies

REP = {"max_ply": 30, "reply_mass": 0.9, "min_reply_p": 0.05, "max_replies": 4, "min_prob": 0.002,
       "also_fine": 0.02, "bad": 0.08, "nodes": 1000000, "multipv": 3, "budget": 1500, "min_book": 30,
       "chapter_min_prob": 0.02, "root_mass": 0.98, "root_max": 10}
OPEN_DEFAULTS = {"max_ply": 30, "out_ply": 20, "bands": [1600, 1800, 2000, 2200, 2400, 4000]}


def board_of(path):
    b = chess.Board()
    for u in path:
        b.push_uci(u)
    return b


def history(path):
    b, out = chess.Board(), []
    for u in path:
        out.append(b.fen())
        b.push_uci(u)
    return out[-7:]


def san(fen, uci):
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def line_san(path):
    b, out = chess.Board(), []
    for u in path:
        mv = chess.Move.from_uci(u)
        out.append((f"{b.fullmove_number}." if b.turn else "") + b.san(mv))
        b.push(mv)
    return " ".join(out)


def build(color, cfg, rc, usual, book, opp_elo, maia, workers):
    """Grow the tree level by level. Returns (mine, theirs) dicts keyed by epd."""
    mine, theirs = {}, {}
    frontier = {chess.Board().epd(): ([], 1.0)}
    spent = 0
    for ply in range(rc["max_ply"]):
        if not frontier:
            break
        my_turn = (ply % 2 == 0) == (color == chess.WHITE)
        if my_turn:
            E = opening_engine(cfg, with_nodes=True)
            need = []
            for e, (path, prob) in sorted(frontier.items(), key=lambda kv: -kv[1][1]):
                c = E.get("tree:" + e)
                if c is None or c[2] < rc["nodes"]:
                    if spent + len(need) >= rc["budget"]:
                        continue
                    need.append(("tree:" + e, board_of(path).fen(), sorted(usual.get(e, {}).keys()), rc["multipv"]))
            if need:
                E = evaluate(cfg, need, rc["nodes"], workers, with_nodes=True)
                spent += len(need)
            nxt = {}
            for e, (path, prob) in frontier.items():
                c = E.get("tree:" + e)
                if c is None:
                    continue                                    # out of budget: leaf
                vals = c[1]
                best = max(vals, key=vals.get)
                u = usual.get(e, {})
                top = max(u, key=lambda m: u[m][0]) if u else None
                rec = top if top in vals and vals[best] - vals[top] <= rc["also_fine"] else best
                node = {"path": path, "prob": prob, "fen": board_of(path).fen(), "vals": vals, "best": best,
                        "rec": rec, "usual": u, "top": top,
                        "alts": [m for m, v in vals.items() if m != rec and vals[best] - v <= rc["also_fine"]]}
                mine[e] = node
                b = board_of(path + [rec])
                if not b.is_game_over():
                    k = b.epd()
                    old = nxt.get(k)
                    nxt[k] = (path + [rec], prob + (old[1] if old else 0.0))
            frontier = nxt
        else:
            nodes, need = {}, []
            for e, (path, prob) in frontier.items():
                v = book.get(e)
                if v is not None and v.n.sum() >= rc["min_book"]:
                    p = (v.n / v.n.sum()).to_numpy()
                    nodes[e] = (path, prob, sorted(zip(v.move, p), key=lambda x: -x[1]), f"book n {int(v.n.sum())}")
                else:
                    nodes[e] = (path, prob, None, "Maia")
                    need.append(e)
            if need:
                lists = maia_lists(maia, [(board_of(frontier[e][0]).fen(), history(frontier[e][0]), opp_elo, opp_elo, [])
                                          for e in need], rc["reply_mass"], rc["max_replies"])
                for e, (lst, _) in zip(need, lists):
                    path, prob, _, src = nodes[e]
                    nodes[e] = (path, prob, sorted(lst, key=lambda x: -x[1]), src)
            nxt = {}
            # the first opponent decisions (1st/2nd move) define the chapters: cover them more widely
            mass_cap, n_cap = ((rc["root_mass"], rc["root_max"]) if ply <= 2 else (rc["reply_mass"], rc["max_replies"]))
            min_p = rc["chapter_min_prob"] if ply <= 2 else rc["min_reply_p"]
            for e, (path, prob, dist, src) in nodes.items():
                chosen, mass = [], 0.0
                for m, p in dist or []:
                    if mass >= mass_cap or len(chosen) >= n_cap or p < min_p:
                        break
                    chosen.append((m, float(p)))
                    mass += p
                theirs[e] = {"path": path, "prob": prob, "fen": board_of(path).fen(), "replies": chosen, "src": src}
                for m, p in chosen:
                    if prob * p < rc["min_prob"]:
                        continue
                    b = board_of(path + [m])
                    if not b.is_game_over():
                        k = b.epd()
                        old = nxt.get(k)
                        nxt[k] = (path + [m], prob * p + (old[1] if old else 0.0))
            frontier = nxt
    # flag replies that lose a lot compared with the best listed reply (values from your side)
    for e, t in theirs.items():
        vs = {}
        for m, _ in t["replies"]:
            child = mine.get(board_of(t["path"] + [m]).epd())
            if child:
                vs[m] = max(child["vals"].values())
        if vs:
            lo = min(vs.values())
            t["mistakes"] = {m: v - lo for m, v in vs.items() if v - lo >= rc["bad"]}
        else:
            t["mistakes"] = {}
    return mine, theirs, spent


def to_pgn(color, mine, theirs, rc, title_prefix):
    """One chapter per reply at the first opponent decision."""
    root = chess.Board()
    path = []
    if color == chess.WHITE:
        n = mine.get(root.epd())
        if not n:
            return []
        path = [n["rec"]]
    first = theirs.get(board_of(path).epd())
    if not first:
        return []
    chapters = []
    orient = "white" if color == chess.WHITE else "black"
    for m, p in first["replies"]:
        if p < rc["chapter_min_prob"]:
            continue
        line = path + [m]
        g = chapter(f"{title_prefix}: {line_san(line)}", orient)
        node = g
        for u in line:
            node = node.add_variation(chess.Move.from_uci(u))
        node.comment = f"{p:.0%} of opponents ({first['src']})"
        grow(node, line, mine, theirs, rc, seen=set())
        chapters.append(g)
    return chapters


def grow(gnode, path, mine, theirs, rc, seen):
    b = board_of(path)
    e = b.epd()
    if e in seen:
        gnode.comment = (gnode.comment + " " if gnode.comment else "") + "(transposes to a line above)"
        return
    seen.add(e)
    if e in mine:
        n = mine[e]
        vals, best, rec = n["vals"], n["best"], n["rec"]
        child = gnode.add_variation(chess.Move.from_uci(rec))
        notes = [f"{100 * vals[rec]:.0f}%"]
        if rec != best:
            notes.append(f"your move, kept: the engine's {san(n['fen'], best)} is only "
                         f"{100 * (vals[best] - vals[rec]):.1f} better")
        if n["alts"]:
            notes.append("also fine: " + ", ".join(f"{san(n['fen'], a)} ({100 * (vals[a] - vals[rec]):+.1f})"
                                                   for a in n["alts"][:3]))
        if not n["usual"]:
            notes.append("new for you")
        child.comment = "; ".join(notes)
        top = n["top"]
        if top and top != rec and top in vals:
            cnt, score = n["usual"][top]
            var = gnode.add_variation(chess.Move.from_uci(top))
            var.comment = (f"your usual move ({cnt}x, score {score:.2f}): {100 * (vals[best] - vals[top]):.1f} "
                           f"points worse than {san(n['fen'], rec)}")
        grow(child, path + [rec], mine, theirs, rc, seen)
    elif e in theirs:
        t = theirs[e]
        for m, p in t["replies"]:
            child = gnode.add_variation(chess.Move.from_uci(m))
            note = f"{p:.0%} ({t['src']})"
            if m in t.get("mistakes", {}):
                note += f"; mistake (-{100 * t['mistakes'][m]:.0f}): punish it"
            child.comment = note
            grow(child, path + [m], mine, theirs, rc, seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--nodes", type=int, default=None, help="search size (refine uses more)")
    ap.add_argument("--band", default=None, help="opponent band for replies, e.g. 2200-2399 (default: your opponents')")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rc = {**REP, **cfg.get("repertoire", {})}
    if args.nodes:
        rc["nodes"] = args.nodes
    oc = {**OPEN_DEFAULTS, **cfg.get("openings", {}), "max_ply": rc["max_ply"]}
    workers = args.workers or cfg["engine"]["workers"]
    cmp_min = cfg["player"]["min_player_rating"]
    od = data_dir(cfg, "openings")
    if not (od / "book.parquet").exists():
        sys.exit("run 14_openings.py first (it builds the book from the Lichess dump)")

    tree, _ = player_tree(cfg, oc)
    recent = tree[(tree.rating >= cmp_min) & tree.mine]
    usual = {}
    for (e, m), g in recent.groupby(["epd", "move"]):
        usual.setdefault(e, {})[m] = (len(g), float(g.result.mean()) if g.result.notna().any() else float("nan"))
    pos = pd.read_parquet(data_dir(cfg, "player") / "positions.parquet", columns=["opp_rating", "mover_rating"])
    pos = pos[pos.mover_rating >= cmp_min]
    opp_elo = float(np.median(player_maia_elo(pos.opp_rating.to_numpy(float), cfg["maia"], pos)))
    labels = band_labels(oc["bands"])
    band = args.band or next((l for l, a, b in zip(labels, oc["bands"][:-1], oc["bands"][1:]) if a <= opp_elo < b),
                             labels[-1])
    if args.band:
        lo = int(args.band.split("-")[0].rstrip("+"))
        opp_elo = lo + 100
    bk = pd.read_parquet(od / "book.parquet")
    book = {e: g[["move", "n", "score"]] for e, g in bk[bk.band == band].groupby("epd")}
    maia = Maia(cfg["maia"]["model"])
    print(f"replies from book band {band} (else Maia at {opp_elo:.0f}); engine {rc['nodes']:,} nodes, "
          f"budget {rc['budget']:,} new positions per colour")

    all_cards, summary, files = [], [], []
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        print(f"\n[{name}]")
        mine, theirs, spent = build(color, cfg, rc, usual, book, opp_elo, maia, workers)
        chapters = to_pgn(color, mine, theirs, rc, f"Repertoire ({name})")
        files += write_studies(chapters, data_dir(cfg, "player") / f"repertoire_{name}")
        # decision points: your usual move is worse than the recommendation
        dp = []
        for e, n in mine.items():
            top = n["top"]
            if top and top != n["rec"] and top in n["vals"] and n["usual"][top][0] >= 2:
                cost = n["vals"][n["best"]] - n["vals"][top]
                cnt, score = n["usual"][top]
                dp.append((n["prob"] * cnt * cost, n, top, cost, cnt, score))
        dp.sort(key=lambda x: -x[0])
        pd.DataFrame([{"color": name, "line": line_san(n["path"]), "fen": n["fen"], "move": top, "best": n["rec"],
                       "cost": cost, "count": cnt, "score": score, "prob": n["prob"], "value": v}
                      for v, n, top, cost, cnt, score in dp]).to_parquet(
            data_dir(cfg, "player") / f"repertoire_points_{name}.parquet", index=False)
        new = sum(1 for n in mine.values() if not n["usual"])
        summary.append(f"{name}: {len(mine)} of your positions in the tree ({new} new for you), {len(theirs)} opponent "
                       f"positions, {spent} new engine searches, {len(chapters)} chapters; "
                       f"{len(dp)} places where a move you've played 2+ times can be improved")
        print("   " + summary[-1])
        for _, n, top, cost, cnt, score in dp[:10]:
            print(f"   {line_san(n['path'])} ... you play {san(n['fen'], top)} ({cnt}x, score {score:.2f}), "
                  f"better {san(n['fen'], n['rec'])} (+{100 * cost:.1f})")
        for _, n, top, cost, cnt, score in dp[:15]:
            r = pd.Series({"fen": n["fen"], "move": top, "best_move": n["rec"], "best_line": [n["rec"]],
                           "played_line": [top], "wp_loss": cost, "phase": "opening"})
            all_cards.append(position_card(r, note=f"{line_san(n['path'])} · you: {san(n['fen'], top)} "
                                                   f"({cnt}x, score {score:.2f}) · better: {san(n['fen'], n['rec'])}"))
    # findings: decision points you still play the worse move in (latest game), as Elo gain
    last = recent.sort_values("date").groupby("epd").move.last()
    n_games = max(recent.game_id.nunique(), 1)
    fi = []
    for name in ("white", "black"):
        p = data_dir(cfg, "player") / f"repertoire_points_{name}.parquet"
        if not p.exists():
            continue
        for r in pd.read_parquet(p).itertuples():
            e = chess.Board(r.fen).epd()
            if last.get(e) != r.move:
                continue                                   # your latest game already played something else
            fi.append(finding(f"rep:{e}", "opening", f"{r.line} … {san(r.fen, r.move)}",
                              f"You play <b>{san(r.fen, r.move)}</b> ({r.count}x, score {r.score:.2f}); "
                              f"<b>{san(r.fen, r.best)}</b> is {100 * r.cost:.1f} points better (as {name}).",
                              "lead" if r.cost < 0.05 else "solid", elo_gain(r.cost * r.count / n_games),
                              "cost × times per game", None, [example(r.fen, [r.best], [r.move], r.line)],
                              {"theme": "opening lines", "puzzles": [{"fen": r.fen, "solution": [r.best],
                                                                     "comment": f"Better than {san(r.fen, r.move)}."}]},
                              "The repertoire study has the follow-ups.", "repertoire.html"))
    write_findings(cfg, "repertoire", fi)
    body = ("<h1>Your lines, extended</h1>"
            f"<p class='muted'>Engine-best repertoire grown from the openings you play. Your usual move is kept when "
            f"it's within {100 * rc['also_fine']:.0f} points of the best. Replies: Lichess players {html.escape(band)} "
            f"(book) or Maia where the book is thin. Study files: {', '.join(html.escape(p.name) for p in files)} "
            f"(lichess.org/study → import PGN).</p>"
            + "".join(f"<p>{html.escape(s)}</p>" for s in summary)
            + "<h2>Where your usual move can be better</h2><p class='muted'>Red = your usual move, green = "
              "recommended. Ranked by how often you reach it × how much it costs.</p>"
            + f"<div class='grid'>{''.join(all_cards)}</div>")
    out = data_dir(cfg, "player") / "repertoire.html"
    out.write_text(page("Repertoire", body), encoding="utf-8")
    print("\nwrote " + ", ".join(str(p) for p in files + [out]))


if __name__ == "__main__":
    mp.freeze_support()
    main()
