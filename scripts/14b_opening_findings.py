"""Opening findings: where your openings cost rating, ranked by expected Elo gain (needs 14).

Focus on weaknesses, not on lines you already handle:
  1. per opening you play (colour × opening family, >= opening_findings.min_games recent games):
       out of the opening: your win prob at move 11 vs peers (same colour)
       moves ~10-15: your loss per move in this opening vs your average there (the part where the
       game leaves theory and plans start; the analysed positions start at ply 20)
     elo = ELO_PER_SCORE x [max(0, peers - you at move 11) + max(0, extra loss/move x moves/game)] x share
     of your games in this opening. Examples and puzzles: your worst moves 10-15 in it.
  2. moves you repeat that cost >= 5 points, unless your LATEST game in that position already had
     a good move (= you've learned it; not shown). elo = loss x times per game.
  3. regressions: positions you've reached >= 6 times where your recent choices are worse than your
     earlier ones (by >= 3 points).
  4. must-know positions (one good move) you still miss, and opponents' opening mistakes you didn't
     punish (from 14).
  5. opening evolution over time (per opening per year: games, score, eval out of the opening) ->
     the Opening tab's "In depth" (with the out-of-the-opening tables 14 computed).

    python scripts/14b_opening_findings.py --config configs/v0.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.categories import annotate
from coach.config import data_dir, load_config
from coach.findings import detail, elo_gain, example, finding, table, write
from coach.openings import opening_engine, player_tree
from coach.results import load_analyzed

OF = {"min_games": 6, "flag": 0.05, "good": 0.03, "regress": 0.03, "min_visits_regress": 6, "critical_gap": 0.10}
OPEN_DEFAULTS = {"max_ply": 30, "out_ply": 20, "bands": [1600, 1800, 2000, 2200, 2400, 4000]}


def san(fen, uci):
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def boot_ci(x, B=1000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    m = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, 5), np.percentile(m, 95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    oc = {**OPEN_DEFAULTS, **cfg.get("openings", {})}
    fc = {**OF, **cfg.get("opening_findings", {})}
    cmp_min = cfg["player"]["min_player_rating"]
    O = data_dir(cfg, "openings")
    if not (O / "report_items.json").exists():
        sys.exit("run 14_openings.py first")
    tree, outs = player_tree(cfg, oc)
    E = opening_engine(cfg)
    vals_of = lambda e: (E.get("tree:" + e) or (None, None))[1]
    peers = (pd.read_parquet(O / "peer_out.parquet") if (O / "peer_out.parquet").exists()
             else pd.DataFrame(columns=["color", "fen", "result", "elo"]))      # quick mode: no peer baseline
    peers["wp"] = [max(E[f"peer:{i}"][1].values()) if f"peer:{i}" in E else np.nan for i in range(len(peers))]
    outs["wp"] = [max(E["out:" + g][1].values()) if "out:" + g in E else np.nan for g in outs.game_id]
    ro = outs[outs.rating >= cmp_min]
    n_games = ro.game_id.nunique()
    items = []

    # ---------------- 1. per opening
    me = annotate(load_analyzed(cfg, "player"))
    me = me[me.mover_rating >= cmp_min]
    fam = dict(zip(outs.game_id, zip(outs.color, outs.family)))
    me["opening"] = [fam.get(g, (None, None)) for g in me.game_id]
    m1015 = me[(me.ply <= 31)]
    base_loss = m1015.wp_loss.mean()
    for (col, f), g in ro.groupby(["color", "family"]):
        if len(g) < fc["min_games"] or f == "unknown":
            continue
        share = len(g) / max(n_games, 1)
        pw = peers[peers.color == col].wp.mean()
        lo, hi = boot_ci(g.wp)
        d_out = max(0.0, pw - g.wp.mean())
        sub = m1015[m1015.opening == (col, f)]
        per_game = len(sub) / max(sub.game_id.nunique(), 1) if len(sub) else 0
        extra = max(0.0, sub.wp_loss.mean() - base_loss) if len(sub) >= 20 else 0.0
        gain = elo_gain((d_out + extra * per_game) * share)
        if gain < 1:
            continue
        tag = "solid" if (np.isfinite(hi) and hi < pw) or (len(sub) >= 40 and sub.wp_loss.mean() > 1.3 * base_loss) else "lead"
        errs = sub[sub.err.to_numpy()].sort_values("wp_loss", ascending=False)
        exs = [example(x.fen, x.best_line, x.played_line, f"{x.date} · move {x.ply // 2 + 1} · lost {100 * x.wp_loss:.0f}",
                       x.pos_id)
               for x in errs.head(3).itertuples()]
        claim = (f"{len(g)} games as {col} (score {g.result.mean():.2f}). Out of the opening you stand at "
                 f"<b>{g.wp.mean():.0%}</b> [{lo:.0%}, {hi:.0%}] vs {pw:.0%} for peers; in moves ~10–15 you lose "
                 f"{100 * sub.wp_loss.mean():.1f} points per move vs your average {100 * base_loss:.1f}.")
        items.append(finding(f"open:{col}:{f}", "opening", f"{f} (as {col})", claim, tag, gain,
                             "(deficit out of the opening + extra loss in moves 10–15) × share of your games",
                             {"header": ["", "you", "reference"],
                              "rows": [["win prob at move 11", f"{g.wp.mean():.1%}", f"{pw:.1%} (peers, all openings)"],
                                       ["loss/move, moves 10–15", f"{100 * sub.wp_loss.mean():.2f}", f"{100 * base_loss:.2f} (you, all openings)"],
                                       ["score", f"{g.result.mean():.2f}", ""]]},
                             exs, {"theme": f"{f} as {col}: moves 10-15",
                                   "pos_ids": errs[errs.critical.astype(bool)].pos_id.head(64).tolist()},
                             "Study the typical middlegame plans of this opening; drill the positions."))

    # ---------------- 2. costly repeated moves, unless already fixed
    mine = tree[tree.mine & (tree.rating >= cmp_min)].sort_values("date")
    for (e, mv), g in mine.groupby(["epd", "move"]):
        vals = vals_of(e)
        if not vals or mv not in vals or len(g) < 2:
            continue
        best = max(vals, key=vals.get)
        loss = vals[best] - vals[mv]
        if loss < fc["flag"]:
            continue
        last = mine[mine.epd == e].iloc[-1]
        if last.move != mv and vals.get(last.move, -1) >= vals[best] - fc["good"]:
            continue                                    # your latest game played a good move: learned
        gain = elo_gain(loss * len(g) / max(n_games, 1))
        r = g.iloc[0]
        items.append(finding(f"line:{e}:{mv}", "opening", f"{r.path} {san(r.fen, mv)}?",
                             f"You played <b>{san(r.fen, mv)}</b> {len(g)}x (score {g.result.mean():.2f}); it costs "
                             f"{100 * loss:.0f} points. Play <b>{san(r.fen, best)}</b>.", "solid", gain,
                             "loss × times per game", None, [example(r.fen, [best], [mv], r.path)],
                             {"theme": "opening lines", "puzzles": [{"fen": r.fen, "solution": [best],
                                                                    "comment": f"Not {san(r.fen, mv)} (-{100 * loss:.0f}).",
                                                                    "wrong": mv}]},
                             "Learn the move; the repertoire study (Drills tab) has the follow-ups."))

    # ---------------- 3. regressions over time
    allmine = tree[tree.mine].sort_values("date")
    for e, g in allmine.groupby("epd"):
        vals = vals_of(e)
        if not vals or len(g) < fc["min_visits_regress"]:
            continue
        best = max(vals.values())
        loss = g.move.map(lambda m: best - vals[m] if m in vals else np.nan).to_numpy(float)
        if np.isnan(loss).mean() > 0.5:
            continue
        h = len(g) // 2
        early, late = np.nanmean(loss[:h]), np.nanmean(loss[h:])
        em, lm = g.move.iloc[:h].mode().iat[0], g.move.iloc[h:].mode().iat[0]
        if late - early >= fc["regress"] and em != lm:       # a real change of move, not noise in one move's value
            r = g.iloc[-1]
            gain = elo_gain((late - early) * (len(g) - h) / max(n_games, 1))
            items.append(finding(f"regress:{e}", "opening", f"Regressed: {r.path}",
                                 f"You used to play <b>{san(r.fen, em)}</b> here (avg loss {100 * early:.1f}); lately "
                                 f"<b>{san(r.fen, lm)}</b> (avg loss {100 * late:.1f}).", "lead", gain,
                                 "extra loss × recent times per game", None,
                                 [example(r.fen, [max(vals, key=vals.get)], [lm], f"{r.path} · {len(g)} visits")],
                                 todo="Go back to the earlier move (or the engine's)."))

    # ---------------- 4. must-know positions still missed, missed punishments (from 14)
    for e, g in mine.groupby("epd"):
        vals = vals_of(e)
        if not vals or len(vals) < 2 or len(g) < 2:
            continue
        vs = sorted(vals.values(), reverse=True)
        if vs[0] - vs[1] < fc["critical_gap"]:
            continue
        hit = g.move.map(lambda m: m in vals and vs[0] - vals[m] < fc["good"]).mean()
        last_ok = g.move.iloc[-1] in vals and vs[0] - vals[g.move.iloc[-1]] < fc["good"]
        if hit >= 0.8 or last_ok:
            continue                                      # known
        best = max(vals, key=vals.get)
        r = g.iloc[0]
        items.append(finding(f"mustknow:{e}", "opening", f"Only move: {r.path}",
                             f"Only <b>{san(r.fen, best)}</b> holds here; you found it {hit:.0%} of {len(g)} times.",
                             "solid", elo_gain((vs[0] - vs[1]) * (1 - hit) * len(g) / max(n_games, 1)),
                             "gap to 2nd best × misses per game", None,
                             [example(r.fen, [best], [g.move.mode().iat[0]], r.path)],
                             {"theme": "opening lines", "puzzles": [{"fen": r.fen, "solution": [best], "comment": "Only move.",
                                                                    "wrong": g.move.mode().iat[0]}]}))
    ri = O / "report_items.json"
    if ri.exists():
        for c in json.loads(ri.read_text()).get("missed_punish", []):
            items.append(finding(f"punish:{c['fen']}", "opening", f"Punish: {c['line']}",
                                 f"Your opponent erred; <b>{san(c['fen'], c['best'])}</b> punishes it. You missed it "
                                 f"{c['count']}x.", "lead", elo_gain(c["loss"] * c["count"] / max(n_games, 1)),
                                 "loss × times per game", None, [example(c["fen"], [c["best"]], [c["move"]], c["line"])],
                                 {"theme": "opening lines", "puzzles": [{"fen": c["fen"], "solution": [c["best"]],
                                                                        "comment": "Punish the mistake.",
                                                                        "wrong": c["move"]}]}))

    # ---------------- 5. in depth: out of the opening (from 14), evolution over time
    ri_d = json.loads(ri.read_text()) if ri.exists() else {}
    mv = ri_d.get("out_ply", oc.get("out_ply", 20)) // 2 + 1
    if ri_d.get("out"):
        o = ri_d["out"]
        items.append(detail("open:out", "opening", f"Where you stand after the opening (move {mv})",
                            f"Stockfish's win probability for you at move {mv}, your recent games vs Lichess peers "
                            f"({ri_d.get('peer_band', 'your band')}) at the same point. Engine only: what your opponents "
                            "usually play doesn't count.",
                            table(["as", "you (90% CI)", "peers", "your games"],
                                  [[c, f"{v['you']:.1%} [{v['lo']:.1%}, {v['hi']:.1%}]", f"{v['peers']:.1%}", v["n"]]
                                   for c, v in o.items()])))
    if ri_d.get("out_conv"):
        items.append(detail("open:conv", "opening", "How your position after the opening turns into results",
                            f"Your final score by how good your position was at move {mv}, vs peers.",
                            table(["win prob at move " + str(mv), "your score (games)", "peers (games)"],
                                  ri_d["out_conv"])))
    if ri_d.get("by_opening"):
        items.append(detail("open:by-opening", "opening", "Your openings: position after the opening and score",
                            f"Openings you've played at least 8 times (recent games): win probability at move {mv}, "
                            "compared with peers over all openings, and your score.",
                            table(["as", "opening", "games", f"win prob at move {mv}", "vs peers (points)", "score"],
                                  ri_d["by_opening"])))
    outs["year"] = outs.date.astype(str).str[:4]
    rows = []
    for (col, f), g in outs.groupby(["color", "family"]):
        if len(g) < fc["min_games"]:
            continue
        for y, gy in g.groupby("year"):
            rows.append([col, f, y, len(gy), f"{gy.result.mean():.2f}", f"{gy.wp.mean():.1%}"])
    if rows:
        items.append(detail("open:evolution", "opening", "Your openings over the years",
                            f"Per opening and year (all your games): games, score, win probability at move {mv}. "
                            "Positions where you got worse over time are listed above as 'Regressed'.",
                            table(["as", "opening", "year", "games", "score", f"win prob at move {mv}"], rows)))

    write(cfg, "openings", items)
    items.sort(key=lambda f: -(f["elo"] or 0))
    print(f"{len(items)} opening findings")
    for f in items[:12]:
        print(f"   {f['tag']:6s} {f['elo'] or 0:6.1f} Elo  {f['title']}")


if __name__ == "__main__":
    main()
