"""Head-to-head: you vs one other player — relative strengths, and opening prep against them.

    python scripts/23_compare_players.py --config configs/v0.yaml --other configs/fawzx.yaml

Needs both players run through the pipeline (main.py with each config). Output:
data/compare/<you>_vs_<them>/report.html + prep.pgn (Lichess study).

  Overview   your level vs theirs per kind of position, both on the same Lichess band scale (17), so
             the comparison holds even though your peer groups differ; where each of you is stronger.
  Opening prep (recency first: a line only counts as exploitable if their LATEST game in that position
             still went wrong and most of their recent visits did; lines they used to get wrong but
             have fixed are listed separately as not exploitable any more)
             - positions they reach (as the player to move) where they keep making the same mistake,
               marked when you reach them too in your own repertoire
             - their recent repertoire against your first moves (what they answer, how they score)
  Their weaknesses / yours  the top findings of each report side by side.
One person is far more predictable than a population, but also a small sample: counts are shown.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.config import data_root, load_config, scope_path
from coach.findings import ELO_PER_SCORE, load_all
from coach.openings import opening_engine, player_tree
from coach.study import chapter, write_studies
from coach.webreport import Report, esc

CMP = {"bad": 0.05, "good": 0.03, "recent_months": 12, "recent_min_games": 100, "min_visits": 2, "recent_k": 3}
OPEN_DEFAULTS = {"max_ply": 30, "out_ply": 20, "bands": [1600, 1800, 2000, 2200, 2400, 4000]}


def san(fen, uci):
    try:
        return chess.Board(fen).san(chess.Move.from_uci(uci))
    except Exception:
        return uci


def levels(cfg):
    p = scope_path(cfg, "player") / "multiband.parquet"
    if not p.exists():
        return None
    mb = pd.read_parquet(p)
    return mb[mb.metric == "loss per 100 moves"].set_index("category").level


def recent_mask(tree, cc):
    """Games in the last recent_months (by the latest game date), or the last recent_min_games games."""
    d = pd.to_datetime(tree.date.astype(str).str.replace(".", "-", regex=False), errors="coerce")
    games = tree.assign(d=d).groupby("game_id").d.first().sort_values()
    if games.notna().any():
        cut = games.max() - pd.DateOffset(months=cc["recent_months"])
        keep = set(games[games >= cut].index)
        if len(keep) < cc["recent_min_games"]:
            keep = set(games.index[-cc["recent_min_games"]:])
    else:
        keep = set(games.index)
    return tree.game_id.isin(keep).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--other", required=True, help="the other player's config")
    args = ap.parse_args()
    A, B = load_config(args.config), load_config(args.other)
    cc = {**CMP, **A.get("compare", {})}
    an, bn = A["player"]["username"], B["player"]["username"]
    out = data_root(A) / "compare" / f"{an.lower()}_vs_{bn.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    rep = Report(f"{an} vs {bn}")

    # ---------------- overview: levels on the shared band scale
    O = rep.tab("overview", "Overview")
    la, lb = levels(A), levels(B)
    if la is not None and lb is not None:
        cats = [c for c in la.index if c in lb.index and pd.notna(la[c]) and pd.notna(lb[c])]
        rows = sorted(((c, la[c], lb[c], la[c] - lb[c]) for c in cats), key=lambda r: -r[3])
        O.append(Report.finding("Your level vs theirs, by kind of position",
                                f"Lichess rapid level whose loss per move matches each of you (17, same bands for both). "
                                f"Overall: you ~{la.get('all moves', np.nan):.0f}, {esc(bn)} ~{lb.get('all moves', np.nan):.0f}. "
                                "Rough (±100); compare rows, not exact numbers.", "info",
                                Report.table(["kind of position", "you", esc(bn), "you − them"],
                                             [[esc(c), f"{a:.0f}", f"{b:.0f}", Report.level_chip(d, f"{d:+.0f}")]
                                              for c, a, b, d in rows])))
        top_a = [r for r in rows if r[3] >= 100][:5]
        top_b = [r for r in reversed(rows) if r[3] <= -100][:5]
        O.append(Report.finding("Where you're stronger", ", ".join(f"{esc(c)} ({d:+.0f})" for c, _, _, d in top_a)
                                or "No category 100+ above them.", "strength"))
        O.append(Report.finding(f"Where {esc(bn)} is stronger", ", ".join(f"{esc(c)} ({-d:+.0f})" for c, _, _, d in top_b)
                                or "No category 100+ above you.", "lead"))
    else:
        O.append("<p>Run the multi-band stage (17) for both players to compare levels.</p>")

    # ---------------- opening prep (recency-aware)
    P = rep.tab("prep", "Opening prep")
    oa = {**OPEN_DEFAULTS, **A.get("openings", {})}
    ta, _ = player_tree(A, oa)
    tb, ob = player_tree(B, {**OPEN_DEFAULTS, **B.get("openings", {})})
    E = {**opening_engine(B), **opening_engine(A)}
    tb = tb.sort_values("date")
    rec = recent_mask(tb, cc)
    a_reach = ta[~ta.mine].groupby("epd").size()
    exploit, fixed, chapters = [], [], []
    for e, g in tb[tb.mine].groupby("epd"):
        v = (E.get("tree:" + e) or (None, None))[1]
        if not v or len(g) < cc["min_visits"]:
            continue
        best = max(v, key=v.get)
        loss = g.move.map(lambda m: v[best] - v[m] if m in v else np.nan).to_numpy(float)
        if np.isnan(loss[-1]):
            continue
        bad = loss >= cc["bad"]
        recent = bad[-cc["recent_k"]:]               # their last few visits (g is in date order)
        r = g.iloc[-1]
        item = {"epd": e, "fen": r.fen, "path": r.path, "move": r.move, "best": best, "loss": float(loss[-1]),
                "visits": len(g), "recent_bad": int(recent.sum()), "recent_n": len(recent),
                "you_reach": int(a_reach.get(e, 0)), "last": r.date}
        if bad[-1] and recent.mean() >= 0.5:
            exploit.append(item)
        elif bad[:-1].mean() >= 0.5 and not bad[-1]:
            fixed.append(item)
    exploit.sort(key=lambda x: -(x["loss"] * (1 + x["you_reach"]) * x["recent_bad"]))
    ex = "".join(rep.example(x["fen"], [x["best"]], [x["move"]],
                             f"{esc(x['path'])} · {esc(bn)} played {esc(san(x['fen'], x['move']))} in "
                             f"{x['recent_bad']} of their last {x['recent_n']} visits (last {esc(x['last'])}) · "
                             f"−{100 * x['loss']:.0f} · " + ("<b>you reach this</b>" if x["you_reach"] else "not in your games yet"),
                             labels=("best move", "their move")) for x in exploit[:12])
    P.append(Report.finding("Mistakes they keep making", f"{len(exploit)} positions where {esc(bn)}'s latest game still "
                            "went wrong and most recent visits did. Steer there; know the punishment.", "exploratory",
                            examples=ex, elo=ELO_PER_SCORE * sum(x["loss"] for x in exploit[:5]) / 5 if exploit else None,
                            elo_note="average loss of the top 5 per game they reach them (rough)"))
    for x in exploit[:40]:
        g = chapter(f"vs {bn}: {x['path']}", "white" if chess.Board(x["fen"]).turn else "black", x["fen"])
        node = g.add_variation(chess.Move.from_uci(x["move"]))
        node.comment = (f"{bn} played this in {x['recent_bad']} of the last {x['recent_n']} visits "
                        f"(-{100 * x['loss']:.0f}); best was {san(x['fen'], x['best'])}.")
        chapters.append(g)
    if fixed:
        P.append(Report.finding("Lines they've fixed (not exploitable any more)",
                                ", ".join(f"{esc(x['path'])} {esc(san(x['fen'], x['move']))}" for x in fixed[:10]),
                                "info", open_=False))
    # their recent repertoire against your first moves
    rows = []
    for col in ("white", "black"):
        mine_first = ta[(ta.mine) & (ta.color == col) & (ta.ply == (0 if col == "white" else 1))]
        their_col = "black" if col == "white" else "white"
        tbr = tb[rec & (tb.color == their_col) & tb.mine.astype(bool) & (tb.ply == (1 if col == "white" else 2))]
        for m, k in mine_first.move.value_counts().head(2).items():
            b0 = chess.Board()
            if col == "black":
                continue
            b0.push_uci(m)
            resp = tbr[tbr.epd == b0.epd()].move.value_counts()
            if len(resp):
                rows.append([f"you 1.{esc(san(chess.STARTING_FEN, m))}", ", ".join(
                    f"{esc(san(b0.fen(), r))} {n}" for r, n in resp.head(4).items())])
    fam = ob.assign(rec=ob.game_id.isin(set(tb.game_id[rec]))).query("rec")
    if len(fam):
        t = fam.groupby(["color", "family"]).agg(games=("game_id", "size"), score=("result", "mean")).reset_index()
        t = t[t.games >= 3].sort_values("games", ascending=False).head(12)
        P.append(Report.finding(f"{esc(bn)}'s recent openings", "Last ~12 months: games and score per opening.", "info",
                                Report.table(["as", "opening", "games", "score"],
                                             [[r.color, esc(r.family), r.games, f"{r.score:.2f}"] for r in t.itertuples()])))
    if rows:
        P.append(Report.finding("Their answers to your first move", "Recent games.", "info",
                                Report.table(["your move", "their replies (count)"], rows)))

    # ---------------- weaknesses side by side
    for tid, label, cfg in (("theirs", f"{bn}'s weaknesses", B), ("yours", "Your weaknesses", A)):
        T = rep.tab(tid, label)
        F = sorted([f for f in load_all(cfg) if f["tag"] in ("solid", "lead") and f["section"] != "games"],
                   key=lambda f: -(f.get("elo") or 0))[:12]
        T += [Report.finding(f"{f['section']}: {f['title']}", f["claim"], f["tag"], elo=f.get("elo"), open_=False)
              for f in F] or ["<p>No findings yet: run the pipeline for this player.</p>"]
    M = rep.tab("method", "Method")
    M.append(f"<p>Recency: a mistake counts as exploitable only if {esc(bn)}'s latest visit to the position still "
             f"went wrong and at least half of their last {cc['recent_k']} visits did. Their recent repertoire uses the "
             f"last {cc['recent_months']} months (at least {cc['recent_min_games']} games). Levels: 17's band fits.</p>")
    files = write_studies(chapters, out / "prep") if chapters else []
    rep.write(out / "report.html")
    print(f"{len(exploit)} exploitable positions ({sum(x['you_reach'] > 0 for x in exploit)} you reach), "
          f"{len(fixed)} fixed lines; wrote {out / 'report.html'}" + "".join(f", {p.name}" for p in files))


if __name__ == "__main__":
    main()
