"""Head-to-head report: you vs one other player. Separate from main.py (which compares you with the
population); run it once both players have been through the pipeline:

    python scripts/23_compare_players.py --config configs/v0.yaml --other configs/fawzx.yaml

Output: data/compare/<you>_vs_<them>/report.html (+ prep_<a>_to_<b>.pgn Lichess studies)

Tabs
  Overview      both players on the same Lichess band scale (17) per kind of position: opening /
                middlegame / endgame, critical, contested, clock, ...; who's stronger where; your games
                against each other (from the chess.com archives).
  You → them    how you can exploit them; Them → you: how they can exploit you. Each:
                  Opening (the focus): their recurring opening mistakes, RECENT only (their latest
                    visit still went wrong and most of their last visits did; lines they've fixed are
                    listed as not exploitable any more), marked when your repertoire reaches them; their
                    recent replies to your first moves and recent results per opening; openings they're
                    weak in that you can steer to (you play the other side).
                  Middlegame / endgame: kinds of positions where they're clearly weaker than you
                    (>= 100 on the band scale) -> where to steer the game; their weaknesses (findings).
  Method
One person is far more predictable than a population, but a small sample: counts are always shown.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.config import data_dir, data_root, load_config, scope_path
from coach.fetch_chesscom import iter_player_games
from coach.findings import load_all
from coach.openings import opening_engine, player_tree
from coach.study import chapter, write_studies
from coach.webreport import Report, esc

CMP = {"bad": 0.05, "good": 0.03, "recent_months": 12, "recent_min_games": 100, "min_visits": 2, "recent_k": 3,
       "steer_gap": 100}
OPEN_DEFAULTS = {"max_ply": 30, "out_ply": 20, "bands": [1600, 1800, 2000, 2200, 2400, 4000]}
PHASES = ("phase: opening (moves 10-15)", "phase: middlegame", "phase: endgame")


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


def recent_games(tree, cc):
    d = pd.to_datetime(tree.date.astype(str).str.replace(".", "-", regex=False), errors="coerce")
    games = tree.assign(d=d).groupby("game_id").d.first().sort_values()
    if not games.notna().any():
        return set(games.index)
    keep = set(games[games >= games.max() - pd.DateOffset(months=cc["recent_months"])].index)
    return keep if len(keep) >= cc["recent_min_games"] else set(games.index[-cc["recent_min_games"]:])


class Player:
    def __init__(self, cfg, cc):
        self.cfg, self.name = cfg, cfg["player"]["username"]
        oc = {**OPEN_DEFAULTS, **cfg.get("openings", {})}
        self.tree, self.outs = player_tree(cfg, oc)
        self.tree = self.tree.sort_values("date")
        self.recent = recent_games(self.tree, cc)
        self.levels = levels(cfg)
        self.findings = load_all(cfg)


def head_to_head(a: Player, b: Player):
    paths = sorted(data_dir(a.cfg, "player", "archives").glob("*.json"))
    res = []
    for g in iter_player_games(paths, a.cfg["player"]):
        w, bl = g["white"]["username"].lower(), g["black"]["username"].lower()
        if {w, bl} == {a.name.lower(), b.name.lower()}:
            me = g["white"] if w == a.name.lower() else g["black"]
            r = me.get("result")
            if not r:
                continue                                    # unknown result
            res.append(1.0 if r == "win" else 0.5 if r in ("agreed", "repetition", "stalemate", "insufficient",
                                                          "50move", "timevsinsufficient") else 0.0)
    return res


def exploit_section(rep, X: Player, Y: Player, E, cc, out_dir):
    """How X can exploit Y. Returns html parts and the number of exploitable positions."""
    parts = []
    tx, ty = X.tree, Y.tree
    x_reach = tx[~tx.mine].groupby("epd").size()
    yrec = ty.game_id.isin(Y.recent).to_numpy()
    exploit, fixed = [], []
    for e, g in ty[ty.mine].groupby("epd"):
        v = (E.get("tree:" + e) or (None, None))[1]
        if not v or len(g) < cc["min_visits"]:
            continue
        best = max(v, key=v.get)
        loss = g.move.map(lambda m: v[best] - v[m] if m in v else np.nan).to_numpy(float)
        if np.isnan(loss[-1]):
            continue
        bad = loss >= cc["bad"]
        recent = bad[-cc["recent_k"]:]                  # their last visits (date order)
        r = g.iloc[-1]
        item = {"fen": r.fen, "path": r.path, "move": r.move, "best": best, "loss": float(loss[-1]),
                "visits": len(g), "recent_bad": int(recent.sum()), "recent_n": len(recent),
                "reach": int(x_reach.get(e, 0)), "last": r.date}
        if bad[-1] and recent.mean() >= 0.5:
            exploit.append(item)
        elif len(bad) > 1 and bad[:-1].mean() >= 0.5 and not bad[-1]:
            fixed.append(item)
    exploit.sort(key=lambda x: -(x["loss"] * (1 + x["reach"]) * x["recent_bad"]))
    ex = "".join(rep.example(x["fen"], [x["best"]], [x["move"]],
                             f"{esc(x['path'])} · {esc(Y.name)} played {esc(san(x['fen'], x['move']))} in "
                             f"{x['recent_bad']} of their last {x['recent_n']} visits (last {esc(x['last'])}) · "
                             f"−{100 * x['loss']:.0f} · " + (f"<b>you reach it ({x['reach']}x)</b>" if x["reach"]
                                                              else "not in your games yet"),
                             labels=("best move", "their move")) for x in exploit[:10])
    parts.append("<h2>Opening</h2>")
    parts.append(Report.finding(f"Mistakes {Y.name} keeps making", f"{len(exploit)} positions where {esc(Y.name)}'s "
                                "latest game still went wrong and most of their recent visits did. Know the punishment; "
                                "the ones you already reach come first.", "exploratory", examples=ex))
    if fixed:
        parts.append(Report.finding("Fixed lines (not exploitable any more)",
                                    "; ".join(f"{esc(x['path'])} {esc(san(x['fen'], x['move']))}" for x in fixed[:10]),
                                    "info", open_=False))
    # their recent replies to X's first moves, and their recent results per opening
    rows = []
    for col, ply in (("white", 0), ("black", 1)):
        firsts = tx[tx.mine & (tx.color == col) & (tx.ply == ply)]
        for path_moves, k in firsts.groupby("path").size().sort_values(ascending=False).head(2).items():
            m = firsts[firsts.path == path_moves].move.value_counts().index[0]
            b0 = chess.Board()
            for t in path_moves.split():
                b0.push_san(t)
            b0.push_uci(m)
            resp = ty[yrec & ty.mine & (ty.epd == b0.epd())].move.value_counts()
            if len(resp):
                line = (path_moves + " " + b0.move_stack[-1].uci()).strip()
                rows.append([f"{esc(line)} (you)", ", ".join(f"{esc(san(b0.fen(), r))} {n}x" for r, n in resp.head(4).items())])
    if rows:
        parts.append(Report.finding(f"{Y.name}'s recent answers to your opening moves", "Their games in the last "
                                    f"{cc['recent_months']} months.", "info",
                                    Report.table(["position", "their replies"], rows)))
    yo = Y.outs[Y.outs.game_id.isin(Y.recent)]
    if len(yo):
        t = yo.groupby(["color", "family"]).agg(games=("game_id", "size"), score=("result", "mean")).reset_index()
        xo = X.outs.groupby(["color", "family"]).size()
        t["you_play_other_side"] = [int(xo.get(("black" if c == "white" else "white", f), 0)) for c, f in zip(t.color, t.family)]
        t = t[t.games >= 3].sort_values("score").head(12)
        parts.append(Report.finding(f"{Y.name}'s recent openings, weakest first", "Score per opening in their recent "
                                    "games; the last column says how often you've played the other side of it "
                                    "(you can steer there).", "exploratory",
                                    Report.table(["they play", "opening", "games", "their score", "you on the other side"],
                                                 [[r.color, esc(r.family), r.games, f"{r.score:.2f}", r.you_play_other_side]
                                                  for r in t.itertuples()])))
    # middlegame / endgame: where Y is weaker than X
    parts.append("<h2>Middlegame and endgame</h2>")
    if X.levels is not None and Y.levels is not None:
        cats = [c for c in X.levels.index if c in Y.levels.index and pd.notna(X.levels[c]) and pd.notna(Y.levels[c])
                and c != "all moves"]
        steer = sorted(((c, X.levels[c] - Y.levels[c]) for c in cats if X.levels[c] - Y.levels[c] >= cc["steer_gap"]),
                       key=lambda r: -r[1])
        parts.append(Report.finding(f"Where to steer the game against {Y.name}",
                                    "Kinds of positions where they play clearly below you (band scale, ≥ "
                                    f"{cc['steer_gap']}): " + (", ".join(f"<b>{esc(c)}</b> ({d:+.0f})" for c, d in steer[:8])
                                                                or "none ≥ the threshold."), "exploratory"))
    yw = sorted([f for f in Y.findings if f["tag"] in ("solid", "lead") and f["section"] in ("middlegame", "endgame")],
                key=lambda f: -(f.get("elo") or 0))[:6]
    for f in yw:
        parts.append(Report.finding(f"{Y.name}: {f['title']}", f["claim"], f["tag"], elo=f.get("elo"), open_=False))
    # study
    chapters = []
    for x in exploit[:40]:
        g = chapter(f"{X.name} vs {Y.name}: {x['path']}", "black" if chess.Board(x["fen"]).turn else "white", x["fen"])
        node = g.add_variation(chess.Move.from_uci(x["move"]))
        node.comment = (f"{Y.name} played this in {x['recent_bad']} of the last {x['recent_n']} visits "
                        f"(-{100 * x['loss']:.0f}); best was {san(x['fen'], x['best'])}. Find the punishment.")
        chapters.append(g)
    files = write_studies(chapters, out_dir / f"prep_{X.name.lower()}_to_{Y.name.lower()}") if chapters else []
    if files:
        parts.append(Report.finding("Prep study", "Import at lichess.org/study: " + ", ".join(esc(p.name) for p in files),
                                    "info", open_=False))
    return parts, len(exploit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--other", required=True, help="the other player's config")
    args = ap.parse_args()
    cfgA, cfgB = load_config(args.config), load_config(args.other)
    cc = {**CMP, **cfgA.get("compare", {})}
    A, B = Player(cfgA, cc), Player(cfgB, cc)
    out = data_root(cfgA) / "compare" / f"{A.name.lower()}_vs_{B.name.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    E = {**opening_engine(cfgB), **opening_engine(cfgA)}
    rep = Report(f"{A.name} vs {B.name}")

    O = rep.tab("overview", "Overview")
    h2h = head_to_head(A, B)
    if h2h:
        O.append(Report.finding("Your games against each other", f"{len(h2h)} games, your score {np.mean(h2h):.2f} "
                                f"({sum(x == 1 for x in h2h)} wins, {sum(x == 0.5 for x in h2h)} draws, "
                                f"{sum(x == 0 for x in h2h)} losses).", "info"))
    if A.levels is not None and B.levels is not None:
        cats = [c for c in A.levels.index if c in B.levels.index and pd.notna(A.levels[c]) and pd.notna(B.levels[c])]
        rows = sorted(((c, A.levels[c], B.levels[c], A.levels[c] - B.levels[c]) for c in cats), key=lambda r: -r[3])
        ph = [r for r in rows if r[0] in PHASES or r[0] == "all moves"]
        if not ph:
            O.append("<p class='muted'>No phase has a level for both players (flat across the bands).</p>")
        O.append(Report.finding("By phase", "Lichess rapid level whose loss per move matches each of you (same bands "
                                "for both; rough, ±100).", "info",
                                Report.table(["", esc(A.name), esc(B.name), f"{esc(A.name)} − {esc(B.name)}"],
                                             [[esc(c), f"{a:.0f}", f"{b:.0f}", Report.level_chip(d, f"{d:+.0f}")]
                                              for c, a, b, d in ph])))
        O.append(Report.finding("By kind of position", "Sorted from your biggest edge to theirs.", "info",
                                Report.table(["kind of position", esc(A.name), esc(B.name), "difference"],
                                             [[esc(c), f"{a:.0f}", f"{b:.0f}", Report.level_chip(d, f"{d:+.0f}")]
                                              for c, a, b, d in rows if c not in PHASES]), open_=False))
    else:
        O.append("<p>Run the multi-band stage (17) for both players to compare levels.</p>")
    n = {}
    for tid, X, Y in (("ab", A, B), ("ba", B, A)):
        T = rep.tab(tid, f"{X.name} → {Y.name}")
        T.append(f"<p class='muted'>How {esc(X.name)} can exploit {esc(Y.name)}. Exploratory: individual patterns from "
                 "their games, recent play weighted.</p>")
        parts, n[tid] = exploit_section(rep, X, Y, E, cc, out)
        T += parts
    M = rep.tab("method", "Method")
    M.append(f"<p>Recency: a mistake counts as exploitable only if the player's latest visit to the position still "
             f"went wrong and at least half of their last {cc['recent_k']} visits did; their recent repertoire uses the "
             f"last {cc['recent_months']} months (at least {cc['recent_min_games']} games). Levels: 17's band fits, "
             "computed for each player on the same rating bands. Engine values: both players' opening caches.</p>")
    rep.write(out / "report.html")
    print(f"{A.name} → {B.name}: {n['ab']} exploitable positions; {B.name} → {A.name}: {n['ba']}; "
          f"head-to-head games: {len(h2h)}; wrote {out / 'report.html'}")


if __name__ == "__main__":
    main()
