"""Practical value, 3 plies deep, with a model of YOUR follow-up (exploitation step 3 + self version).

Step 2 (12) assumed you punish every opponent mistake perfectly. Here the tree is

    your move m  ->  their reply r  ->  your follow-up f  ->  leaf: Stockfish value

  m: Stockfish candidates within practical.delta of the best (+ the move actually played)
  r: Maia at the opponent's level, top replies to tree.opp_mass (max tree.max_opp) + Stockfish's best r*
  f: Maia at YOUR level, top follow-ups to tree.self_mass (max tree.max_self) + Stockfish's best f*

Values (win prob for you):
  your node, perfect:   U_perf(m, r) = max_f L(m, r, f)
  your node, modelled:  U_self(m, r) = sum_f P_you(f) L(m, r, f) + (1 - sum P_you) min_f L(m, r, f)
  their node (restricted-Nash style mixture, p = weight on the model):
      O_p(m) = p [sum_r P_opp(r) U(m, r) + (1 - sum P_opp) U(m, r*)] + (1 - p) min_r U(m, r)
Unlisted probability is treated conservatively both ways (opponent finds r*; you play your
worst listed follow-up).

Reports
  - execution discount = O_1(m; perfect) - O_1(m; self): how much of a line's value you'd
    likely give back in the follow-up. Large for Stockfish's move = a sharp line that is hard
    for you; the coaching signal for "simplify".
  - practical choice with perfect follow-up (= step 2) vs with your follow-up modelled.
  - p sweep: expected gain under the model vs worst-case cost (p = 0), the safety trade-off.
  - validation (you): actual reply + your actual follow-up (from your next stored move):
    predicted U_self vs realized leaf value, binned by predicted discount.

    python scripts/13_practical_tree.py --config configs/v0.yaml            # both sets, tree.n each
    python scripts/13_practical_tree.py --config configs/v0.yaml --n 50     # quick test

Output: data/<set>/practical_tree.parquet, data/player/practical_tree_report.html
Config (optional block `tree:`; defaults below): n 600, opp_mass 0.85, max_opp 4,
self_mass 0.85, max_self 4, nodes 300000, p_grid [0, 0.25, 0.5, 0.75, 1].
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
from coach.handcrafted import phase
from coach.maia import Maia, maia_lists
from coach.report import page, position_card
from coach.results import load_analyzed

TREE = {"n": 600, "opp_mass": 0.85, "max_opp": 4, "self_mass": 0.85, "max_self": 4, "nodes": 300000,
        "p_grid": [0, 0.25, 0.5, 0.75, 1]}

_an: Analyzer | None = None
_cfg: dict = {}


def _init(ecfg, sf_path, c):
    global _an, _cfg
    _an, _cfg = Analyzer(ecfg, sf_path), c


def _cands(item):
    pos_id, fen, played = item
    try:
        cands, best = _an.candidates(fen, played, _cfg["multipv"], _cfg["nodes_root"], _cfg["delta"])
        return [{"pos_id": pos_id, "move": m, "wp_sf": w, "is_sf_best": m == best, "is_played": m == played}
                for m, w in cands]
    except Exception as e:
        print(f"  candidates failed on {pos_id}: {e}", flush=True)
        return []


def _expand(item):
    key, fen, prefix, moves, root_eval = item
    try:
        return key, _an.expand(fen, prefix, moves, _cfg["nodes"], root_eval)
    except Exception as e:
        print(f"  expand failed on {key}: {e}", flush=True)
        return key, None


def run_pool(fn, items, ecfg, sf, c, workers, label):
    out, t0 = [], time.time()
    with mp.Pool(workers, initializer=_init, initargs=(ecfg, sf, c)) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, items, chunksize=4), 1):
            out.append(r)
            if i % 500 == 0 or i == len(items):
                el = time.time() - t0
                print(f"  {label}: {i:,}/{len(items):,}  ETA {el / i * (len(items) - i) / 60:.0f} min", flush=True)
    return out


# ---------------------------------------------------------------- data
def select(cfg, name, n, seed):
    pc = cfg["practical"]
    df = load_analyzed(cfg, name)
    lo, hi = pc["contested"]
    m = (df.wp_best >= lo) & (df.wp_best <= hi)
    if name == "player":
        m &= df.mover_rating >= cfg["player"]["min_player_rating"]
    sub = df[m].sample(min(n, int(m.sum())), random_state=seed).reset_index(drop=True)
    sub["phase"] = [phase(chess.Board(f)) for f in sub.fen]
    return sub


def elos(cfg, name, df):
    """(opponent's Maia Elo, mover's Maia Elo) per row."""
    if name == "population":
        return df.opp_rating.to_numpy(float), df.mover_rating.to_numpy(float)
    mc = cfg["maia"]
    return (player_maia_elo(df.opp_rating.to_numpy(float), mc, df),
            player_maia_elo(df.mover_rating.to_numpy(float), mc, df))


def actual_lines(cfg, name, df):
    """pos_id -> (actual reply r, your actual follow-up f), from your next stored move (ply + 2)."""
    if name != "player":
        return {}
    pos = pd.read_parquet(data_dir(cfg, name) / "positions.parquet", columns=["game_id", "ply", "fen", "move"])
    nxt = {(g, p): (f, m) for g, p, f, m in zip(pos.game_id, pos.ply, pos.fen, pos.move)}
    out = {}
    for pid, g, p, fen, mv in zip(df.pos_id, df.game_id, df.ply, df.fen, df.move):
        n = nxt.get((g, p + 2))
        if n is None:
            continue
        b = chess.Board(fen)
        b.push_uci(mv)
        target = n[0].split()[:3]
        for r in b.legal_moves:
            b.push(r)
            if b.fen().split()[:3] == target:
                out[pid] = (r.uci(), n[1])
            b.pop()
            if pid in out:
                break
    return out


def push(fen, *moves):
    b = chess.Board(fen)
    for u in moves:
        b.push_uci(u)
    return b.fen()


def analyse_set(cfg, name, n, seed, workers, maia):
    tc = {**TREE, **cfg.get("tree", {})}
    pc = cfg["practical"]
    c = {**pc, **tc}
    ecfg, sf = dict(cfg["engine"]), str(resolve(cfg, cfg["paths"]["stockfish"]))
    df = select(cfg, name, n, seed)
    print(f"\n[{name}] {len(df):,} contested positions")
    row = {p: i for i, p in enumerate(df.pos_id)}
    opp_elo, my_elo = elos(cfg, name, df)
    actual = actual_lines(cfg, name, df)

    # 1. your candidates
    a = run_pool(_cands, list(zip(df.pos_id, df.fen, df.move)), ecfg, sf, c, workers, "candidates")
    C = pd.DataFrame([r for rows in a for r in rows])

    # 2. their replies (Maia at their level) + value after each, with your best follow-up f*
    def hist(i, *extra):
        return list(df.history_fens.iat[i]) + [df.fen.iat[i]] + list(extra)
    rn = []
    for pid, m, pl in zip(C.pos_id, C.move, C.is_played):
        i = row[pid]
        ra = actual.get(pid, (None, None))[0] if pl else None
        rn.append((push(df.fen.iat[i], m), hist(i), opp_elo[i], my_elo[i], [ra]))
    R = maia_lists(maia, rn, c["opp_mass"], c["max_opp"])
    items = [(k, df.fen.iat[row[pid]], [m], [r for r, _ in R[k][0]], True)
             for k, (pid, m) in enumerate(zip(C.pos_id, C.move))]
    e1 = dict(run_pool(_expand, items, ecfg, sf, c, workers, "their replies"))

    # 3. your follow-ups (Maia at your level) after each reply, valued by Stockfish
    nodes, keys = [], []
    for k, (pid, m, pl) in enumerate(zip(C.pos_id, C.move, C.is_played)):
        if e1.get(k) is None:
            continue
        i = row[pid]
        ra, fa = actual.get(pid, (None, None)) if pl else (None, None)
        _, _, vals = e1[k]
        for r, (v, fstar) in vals.items():
            extra = [fstar] + ([fa] if r == ra else [])
            nodes.append((push(df.fen.iat[i], m, r), hist(i, push(df.fen.iat[i], m)), my_elo[i], opp_elo[i], extra))
            keys.append((k, r))
    F = maia_lists(maia, nodes, c["self_mass"], c["max_self"])
    key_of = {kr: j for j, kr in enumerate(keys)}
    items = [(j, df.fen.iat[row[C.pos_id.iat[k]]], [C.move.iat[k], r], [f for f, _ in F[j][0]], False)
             for j, (k, r) in enumerate(keys)]
    e2 = dict(run_pool(_expand, items, ecfg, sf, c, workers, "your follow-ups"))

    # 4. assemble per candidate: per reply (P_opp, U_perf, U_self, listed self mass)
    per_reply = {}
    for j, (k, r) in enumerate(keys):
        res = e2.get(j)
        if res is None:
            continue
        leaves = {f: v for f, (v, _) in res[2].items()}
        if not leaves:                    # game over after r: value from step 2
            v = e1[k][2][r][0]
            per_reply.setdefault(k, {})[r] = (v, v, 1.0, {})
            continue
        listed = [(f, p) for f, p in F[j][0] if f in leaves]
        lo = min(leaves.values())
        u_self = sum(p * leaves[f] for f, p in listed) + max(0.0, 1 - sum(p for _, p in listed)) * lo
        per_reply.setdefault(k, {})[r] = (max(leaves.values()), u_self, sum(p for _, p in listed), leaves)
    recs = []
    for k in range(len(C)):
        if e1.get(k) is None or k not in per_reply:
            continue
        rstar, vstar, _ = e1[k]
        ur = per_reply[k]
        listed = [(r, p) for r, p in R[k][0] if r in ur]
        rs = rstar if rstar in ur else (min(ur, key=lambda r: ur[r][0]))
        rec = C.iloc[k].to_dict()
        rec.update({"vstar": vstar, "rstar": rstar, "opp_mass": sum(p for _, p in listed),
                    "self_mass": np.mean([ur[r][2] for r in ur]),
                    "p_opp_best": R[k][1].get(rstar, 0.0) if rstar else np.nan})
        for pol, idx in (("perf", 0), ("self", 1)):
            model = sum(p * ur[r][idx] for r, p in listed) + max(0.0, 1 - sum(p for _, p in listed)) * ur[rs][idx]
            worst = min(ur[r][idx] for r in ur)
            for p in c["p_grid"]:
                rec[f"O_{pol}_{p}"] = p * model + (1 - p) * worst
        pid = C.pos_id.iat[k]
        if C.is_played.iat[k] and pid in actual:
            ra, fa = actual[pid]
            if ra in ur and fa in ur[ra][3]:
                rec.update({"act_reply": ra, "act_follow": fa, "act_u_perf": ur[ra][0], "act_u_self": ur[ra][1],
                            "act_leaf": ur[ra][3][fa],
                            "act_follow_listed": fa in {f for f, _ in F[key_of[(k, ra)]][0][:c["max_self"]]}})
        recs.append(rec)
    T = pd.DataFrame(recs)
    T["set"] = name
    T.to_parquet(data_dir(cfg, name) / "practical_tree.parquet", index=False)
    return df, T, c


# ---------------------------------------------------------------- summary
def summarize(name, T, c):
    L = [f"[{name}]"]
    if T.empty:
        return pd.DataFrame(), L + ["  nothing to summarize"]
    one = max(c["p_grid"])          # the grid element itself, so the column names match
    zero = min(c["p_grid"])
    T = T.assign(discount=T[f"O_perf_{one}"] - T[f"O_self_{one}"])
    per = []
    for pid, g in T.groupby("pos_id"):
        safe = g[g.wp_sf >= g.wp_sf.max() - c["delta"]]
        sf = safe[safe.is_sf_best]
        if sf.empty:
            continue
        sf = sf.iloc[0]
        rec = {"pos_id": pid, "n_safe": len(safe), "sf_move": sf.move, "sf_discount": sf.discount,
               "best_discount": safe.discount.min()}
        for pol in ("perf", "self"):
            ch = safe.loc[safe[f"O_{pol}_{one}"].idxmax()]
            rec[f"{pol}_move"], rec[f"{pol}_gain"] = ch.move, ch[f"O_self_{one}"] - sf[f"O_self_{one}"]
            rec[f"{pol}_cost"], rec[f"{pol}_disc"] = sf.wp_sf - ch.wp_sf, ch.discount
        for p in c["p_grid"]:
            ch = safe.loc[safe[f"O_self_{p}"].idxmax()]
            rec[f"p{p}_differs"] = ch.move != sf.move
            rec[f"p{p}_gain"] = ch[f"O_self_{one}"] - sf[f"O_self_{one}"]
            rec[f"p{p}_worst"] = ch[f"O_self_{zero}"] - sf[f"O_self_{zero}"]
            rec[f"p{p}_cost"] = sf.wp_sf - ch.wp_sf
        per.append(rec)
    P = pd.DataFrame(per)
    L.append(f"  {len(P):,} positions; listed probability: their replies {T.opp_mass.mean():.0%}, "
             f"your follow-ups {T.self_mass.mean():.0%}")
    L.append(f"  execution discount on Stockfish's move (value you'd likely give back in the follow-up): "
             f"mean {100 * P.sf_discount.mean():.1f} pts, median {100 * P.sf_discount.median():.1f}, "
             f"> 5 pts in {(P.sf_discount > 0.05).mean():.0%} of positions")
    for pol, lab in (("perf", "assuming you follow up perfectly (= step 2)"), ("self", "with your follow-up modelled")):
        d = P[P[f"{pol}_move"] != P.sf_move]
        L.append(f"  practical move {lab}: differs from Stockfish's in {len(d) / len(P):.0%}; "
                 f"gain (your model) {100 * P[f'{pol}_gain'].mean():+.2f}/decision"
                 + (f", when it differs {100 * d[f'{pol}_gain'].mean():+.1f}, objective cost "
                    f"{100 * d[f'{pol}_cost'].mean():.1f}, its discount {100 * d[f'{pol}_disc'].mean():.1f} vs "
                    f"Stockfish's {100 * d.sf_discount.mean():.1f}" if len(d) else ""))
    ch = P.perf_move != P.self_move
    L.append(f"  modelling your follow-up changes the step-2 pick in {ch.mean():.0%} of positions")
    simp = (P.self_move != P.sf_move) & (P.sf_discount - P.self_disc >= 0.05)
    L.append(f"  'simplify' positions (your pick avoids >= 5 pts of execution risk in Stockfish's line): {simp.mean():.0%}")
    L.append("  safety trade-off, p = weight on the opponent model (0 = they always find the best reply):")
    L.append(f"    {'p':>5s} {'differs':>8s} {'gain (model)':>13s} {'worst-case':>11s} {'obj. cost':>10s}")
    for p in c["p_grid"]:
        L.append(f"    {p:5.2f} {P[f'p{p}_differs'].mean():8.0%} {100 * P[f'p{p}_gain'].mean():+13.2f} "
                 f"{100 * P[f'p{p}_worst'].mean():+11.2f} {100 * P[f'p{p}_cost'].mean():10.2f}")
    v = T.dropna(subset=["act_leaf"]) if "act_leaf" in T else pd.DataFrame()
    if len(v) >= 30:
        L.append(f"  validation on {len(v):,} of your moves with the actual reply + your actual follow-up:")
        L.append(f"    your follow-up was in Maia's list {v.act_follow_listed.mean():.0%}; mean predicted U_self "
                 f"{100 * v.act_u_self.mean():.1f} vs realized {100 * v.act_leaf.mean():.1f} "
                 f"(perfect {100 * v.act_u_perf.mean():.1f})")
        d = v.act_u_perf - v.act_u_self
        for a, b in [(-1, 0.01), (0.01, 0.03), (0.03, 0.08), (0.08, 1)]:
            m = (d >= a) & (d < b)
            if m.sum() >= 15:
                L.append(f"    predicted discount {100 * d[m].mean():5.1f} -> realized "
                         f"{100 * (v.act_u_perf[m] - v.act_leaf[m]).mean():5.1f}  (n {m.sum():,})")
    return P, L


def report(cfg, df, P):
    if P.empty:
        return
    fen_of = df.set_index("pos_id")
    ex = P[(P.self_move != P.sf_move) & (P.sf_discount - P.self_disc >= 0.03)]
    ex = ex.sort_values("self_gain", ascending=False).head(12)
    cards = []
    for r in ex.itertuples():
        row = fen_of.loc[r.pos_id].copy()
        row["move"], row["best_move"] = r.self_move, r.sf_move
        row["best_line"], row["played_line"] = [r.sf_move], [r.self_move]
        row["wp_loss"] = r.self_cost
        b = chess.Board(row.fen)
        cards.append(position_card(row, note=(
            f"Stockfish: {b.san(chess.Move.from_uci(r.sf_move))} (you'd likely give back "
            f"{100 * r.sf_discount:.0f} pts in the follow-up); simpler: {b.san(chess.Move.from_uci(r.self_move))} "
            f"(gives back {100 * r.self_disc:.0f}); net gain {100 * r.self_gain:+.1f}, objective cost "
            f"{100 * r.self_cost:.1f}")))
    body = ("<h1>Practical value with your follow-up</h1><p class='muted'>Positions where Stockfish's move leads "
            "to a line you would likely misplay (Maia at your level for your next move) and a move within a small "
            "margin is better in practice. Green = Stockfish's best, red = the practical move. Win-probability "
            f"points.</p><div class='grid'>{''.join(cards)}</div>")
    (data_dir(cfg, "player") / "practical_tree_report.html").write_text(page("Practical tree", body), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--set", choices=["player", "population", "both"], default="both")
    ap.add_argument("--n", type=int, default=None, help="positions per set (default tree.n)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = args.n or cfg.get("tree", {}).get("n", TREE["n"])
    workers = args.workers or cfg["engine"]["workers"]
    maia = Maia(cfg["maia"]["model"])
    lines = []
    for name in (["player", "population"] if args.set == "both" else [args.set]):
        df, T, c = analyse_set(cfg, name, n, args.seed, workers, maia)
        P, L = summarize(name, T, c)
        lines += L
        if name == "player":
            report(cfg, df, P)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    mp.freeze_support()
    main()
