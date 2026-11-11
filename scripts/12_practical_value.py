"""Practical value of candidate moves against a modelled opponent (exploitation step 2).

For a sample of contested positions (best-move win prob in practical.contested):
  A. Stockfish: candidate moves within practical.delta win probability of the best (+ the
     move actually played).                                             [multiprocess]
  B. Maia at the *opponent's* level: likely replies to each candidate, until their
     probability covers practical.reply_mass (max practical.max_replies).  [GPU]
  C. Stockfish: value after each listed reply, and the opponent's best reply r* with value
     V* (from the mover's side).                                        [multiprocess]
Then, for each candidate m:

    EV(m) = sum_r P_opp(r | m) V(m, r) + (1 - sum_r P_opp(r | m)) V*(m)

(unlisted probability is assumed to find the best reply: conservative). The practical move
is the highest-EV candidate; we report how often it differs from Stockfish's best, the
expected gain and its objective cost, with example positions. Validation: for played moves
whose actual reply is known (09d), does predicted "trap value" EV - V* match what happened?

    python scripts/12_practical_value.py --config configs/v0.yaml --n 2000

Output: data/<set>/practical.parquet, data/player/practical_report.html
"""
from __future__ import annotations

import argparse
import html
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
from coach.maia import Maia
from coach.report import page, position_card
from coach.results import load_analyzed

_an: Analyzer | None = None
_pc: dict = {}


def _init(ecfg, sf_path, pcfg):
    global _an, _pc
    _an, _pc = Analyzer(ecfg, sf_path), pcfg


def _stage_a(item):
    pos_id, fen, played = item
    try:
        cands, best = _an.candidates(fen, played, _pc["multipv"], _pc["nodes_root"], _pc["delta"])
        return [{"pos_id": pos_id, "move": m, "wp_sf": w, "is_sf_best": m == best, "is_played": m == played}
                for m, w in cands]
    except Exception as e:
        print(f"  stage A failed on {pos_id}: {e}", flush=True)
        return []


def _stage_c(item):
    key, fen, move, replies = item
    try:
        rstar, vstar, vals = _an.reply_values(fen, move, replies, _pc["nodes_reply"])
        return {"key": key, "rstar": rstar, "vstar": vstar, "vals": vals}
    except Exception as e:
        print(f"  stage C failed on {key}: {e}", flush=True)
        return None


def run_pool(fn, items, ecfg, sf_path, pcfg, workers, label):
    out, t0 = [], time.time()
    with mp.Pool(workers, initializer=_init, initargs=(ecfg, sf_path, pcfg)) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, items, chunksize=4), 1):
            out.append(r)
            if i % 200 == 0 or i == len(items):
                el = time.time() - t0
                print(f"  {label}: {i:,}/{len(items):,}  ETA {el / i * (len(items) - i) / 60:.0f} min", flush=True)
    return out


def select(cfg, name, n, seed):
    pc = cfg["practical"]
    df = load_analyzed(cfg, name)
    lo, hi = pc["contested"]
    m = (df.wp_best >= lo) & (df.wp_best <= hi)
    if name == "player":
        m &= df.mover_rating >= cfg["player"]["min_player_rating"]
    sub = df[m]
    sub = sub.sample(min(n, len(sub)), random_state=seed).reset_index(drop=True)
    sub["phase"] = [phase(chess.Board(f)) for f in sub.fen]
    return sub


def elos(cfg, name, df):
    if name == "population":
        return df.opp_rating.to_numpy(float), df.mover_rating.to_numpy(float)
    mc = cfg["maia"]
    return player_maia_elo(df.opp_rating.to_numpy(float), mc, df), player_maia_elo(df.mover_rating.to_numpy(float), mc, df)


def actual_replies(cfg, name) -> dict:
    p = data_dir(cfg, name) / "opp_model.parquet"
    if not p.exists():
        return {}
    o = pd.read_parquet(p, columns=["pos_id", "reply"])
    return dict(zip(o.pos_id, o.reply))


def maia_replies(maia, df, cands, opp_elo, mov_elo, pc, actual, batch=128):
    """Per candidate: [(reply, P)] covering reply_mass, + P of every legal reply kept for lookups."""
    torch = maia.torch
    row_of = {p: i for i, p in enumerate(df.pos_id)}
    lists, p_actual, full = [], [], []
    for b0 in range(0, len(cands), batch):
        sub = cands.iloc[b0:b0 + batch]
        T, M, fens, rows = [], [], [], []
        for pos_id, move in zip(sub.pos_id, sub.move):
            i = row_of[pos_id]
            b = chess.Board(df.fen.iat[i])
            b.push_uci(move)
            fa = b.fen()
            T.append(maia.tokens(fa, (list(df.history_fens.iat[i]) + [df.fen.iat[i]])[-7:]))
            M.append(maia.legal_mask(fa))
            fens.append(fa)
            rows.append(i)
        rows = np.array(rows)
        P = maia.policy(torch.stack(T), torch.stack(M), opp_elo[rows], mov_elo[rows]).cpu().numpy()
        for k, (fa, i) in enumerate(zip(fens, rows)):
            b = chess.Board(fa)
            if b.is_game_over():
                lists.append([])
                p_actual.append(np.nan)
                full.append({})
                continue
            probs = sorted(((float(P[k, maia.index(fa, m.uci())]), m.uci()) for m in b.legal_moves), reverse=True)
            chosen, mass = [], 0.0
            for p, m in probs:
                if mass >= pc["reply_mass"] or len(chosen) >= pc["max_replies"]:
                    break
                chosen.append((m, p))
                mass += p
            a = actual.get(df.pos_id.iat[i]) if sub.is_played.iat[k] else None
            if a and a not in {m for m, _ in chosen}:
                chosen.append((a, dict((m, p) for p, m in probs).get(a, 0.0)))
            lists.append(chosen)
            p_actual.append(dict((m, p) for p, m in probs).get(a, np.nan) if a else np.nan)
            full.append({m: p for p, m in probs if p >= 0.001})   # to look up P(Stockfish's best reply)
    return lists, p_actual, full


def analyse_set(cfg, name, n, seed, workers, maia):
    pc, ecfg = cfg["practical"], dict(cfg["engine"])
    sf = str(resolve(cfg, cfg["paths"]["stockfish"]))
    df = select(cfg, name, n, seed)
    print(f"\n[{name}] {len(df):,} contested positions")
    played = df.move if "move" in df else [None] * len(df)
    a = run_pool(_stage_a, list(zip(df.pos_id, df.fen, played)), ecfg, sf, pc, workers, "A candidates")
    cands = pd.DataFrame([r for rows in a for r in rows])
    opp, mov = elos(cfg, name, df)
    actual = actual_replies(cfg, name)
    lists, p_act, full = maia_replies(maia, df, cands, opp, mov, pc, actual)
    cands["replies"] = lists
    cands["p_actual_reply"] = p_act
    cands["actual_reply"] = [actual.get(p) if pl else None for p, pl in zip(cands.pos_id, cands.is_played)]
    fen_of = dict(zip(df.pos_id, df.fen))
    items = [(k, fen_of[p], m, [r for r, _ in rl]) for k, (p, m, rl) in
             enumerate(zip(cands.pos_id, cands.move, cands.replies))]
    c = run_pool(_stage_c, items, ecfg, sf, pc, workers, "C reply values")
    res = {r["key"]: r for r in c if r}
    ev, vstar, p_star, rstar, v_actual = [], [], [], [], []
    for k, rl in enumerate(cands.replies):
        r = res.get(k)
        if r is None:
            ev.append(np.nan); vstar.append(np.nan); p_star.append(np.nan); rstar.append(None); v_actual.append(np.nan)
            continue
        listed = sum(p for _, p in rl)
        e = sum(p * r["vals"].get(m, r["vstar"]) for m, p in rl) + max(0.0, 1 - listed) * r["vstar"]
        ev.append(e)
        vstar.append(r["vstar"])
        rstar.append(r["rstar"])
        p_star.append(full[k].get(r["rstar"], 0.0) if r["rstar"] else np.nan)   # < 0.1% counts as 0
        a = cands.actual_reply.iat[k]
        v_actual.append(r["vals"].get(a, np.nan) if a else np.nan)
    cands["ev"], cands["vstar"], cands["rstar"], cands["p_opp_best"], cands["v_actual"] = ev, vstar, rstar, p_star, v_actual
    cands["trap"] = cands.ev - cands.vstar      # expected gain from the opponent's likely errors
    cands["set"] = name
    out = data_dir(cfg, name) / "practical.parquet"
    cands.drop(columns=["replies"]).assign(replies=[str(r) for r in cands.replies]).to_parquet(out, index=False)
    return df, cands


def summarize(name, df, cands, delta):
    lines = [f"[{name}]"]
    c = cands.dropna(subset=["ev"])
    per = []
    for pos_id, g in c.groupby("pos_id"):
        best_wp = g.wp_sf.max()
        safe = g[g.wp_sf >= best_wp - delta]
        sf_row = safe[safe.is_sf_best]
        if sf_row.empty or safe.empty:
            continue
        sf_row = sf_row.iloc[0]
        ev_row = safe.loc[safe.ev.idxmax()]
        per.append({"pos_id": pos_id, "n_cands": len(safe), "differs": ev_row.move != sf_row.move,
                    "gain": ev_row.ev - sf_row.ev, "cost": sf_row.wp_sf - ev_row.wp_sf,
                    "sf_move": sf_row.move, "ev_move": ev_row.move, "sf_ev": sf_row.ev, "ev_ev": ev_row.ev,
                    "sf_p_best": sf_row.p_opp_best, "ev_p_best": ev_row.p_opp_best,
                    "played_ev": g[g.is_played].ev.max() if g.is_played.any() else np.nan})
    P = pd.DataFrame(per)
    if P.empty:
        return P, lines + ["  nothing to summarize"]
    d = P[P.differs]
    lines.append(f"  {len(P):,} positions; with more than one safe candidate: {(P.n_cands > 1).mean():.0%}")
    lines.append(f"  practical move differs from Stockfish's best in {P.differs.mean():.0%} of positions")
    if len(d):
        lines.append(f"  when it differs: expected gain {100 * d.gain.mean():+.1f} win-prob points "
                     f"(median {100 * d.gain.median():+.1f}), objective cost {100 * d.cost.mean():.1f}; "
                     f"opponent finds the best reply {d.sf_p_best.mean():.0%} after Stockfish's move vs "
                     f"{d.ev_p_best.mean():.0%} after the practical one")
    lines.append(f"  average over all positions: gain {100 * P.gain.mean():+.2f} win-prob points per decision")
    if P.played_ev.notna().any():
        lines.append(f"  your/their actual move vs the practical one: EV {100 * (P.played_ev - P.ev_ev).mean():+.1f} points")
    # validation: predicted trap value vs what the actual reply gave
    v = cands[cands.is_played & cands.v_actual.notna() & cands.trap.notna()]
    if len(v) >= 50:
        lines.append(f"  validation on {len(v):,} played moves with known replies (predicted vs realized gain over V*):")
        bins = [-1, 0.005, 0.02, 0.05, 0.1, 1]
        for a, b in zip(bins[:-1], bins[1:]):
            m = (v.trap >= a) & (v.trap < b)
            if m.sum() >= 20:
                lines.append(f"    predicted {100 * v.trap[m].mean():5.1f} -> realized "
                             f"{100 * (v.v_actual[m] - v.vstar[m]).mean():5.1f}  (n {m.sum():,})")
    return P, lines


def report(cfg, df, cands, P):
    if P.empty:
        return
    fen_of = df.set_index("pos_id")
    ex = P[P.differs & (P.gain >= 0.01)].sort_values("gain", ascending=False).head(12)
    cards = []
    for r in ex.itertuples():
        row = fen_of.loc[r.pos_id].copy()
        row["move"] = r.ev_move            # show the practical move as "played" (red) vs Stockfish's (green)
        row["best_move"] = r.sf_move
        row["best_line"] = [r.sf_move]
        row["played_line"] = [r.ev_move]
        row["wp_loss"] = r.cost            # card header "loss" = the practical move's objective cost
        b = chess.Board(row.fen)
        cards.append(position_card(row, note=(
            f"Stockfish: {b.san(chess.Move.from_uci(r.sf_move))} (opponent finds the best reply "
            f"{r.sf_p_best:.0%}); practical: {b.san(chess.Move.from_uci(r.ev_move))} (they find it "
            f"{r.ev_p_best:.0%}); expected gain {100 * r.gain:+.1f}, objective cost {100 * r.cost:.1f}")))
    body = ("<h1>Practical value</h1><p class='muted'>Positions where a move within a small margin of "
            "Stockfish's best gives a modelled opponent (Maia at their level) more chances to go wrong. "
            "Green = Stockfish's best, red = the practical move. Numbers are win-probability points.</p>"
            f"<div class='grid'>{''.join(cards)}</div>")
    (data_dir(cfg, "player") / "practical_report.html").write_text(page("Practical value", body), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--set", choices=["player", "population", "both"], default="both")
    ap.add_argument("--n", type=int, default=None, help="positions per set (default practical.n)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pc = cfg["practical"]
    workers = args.workers or cfg["engine"]["workers"]
    n = args.n or pc["n"]
    maia = Maia(cfg["maia"]["model"])
    sets = ["player", "population"] if args.set == "both" else [args.set]
    all_lines = []
    for name in sets:
        df, cands = analyse_set(cfg, name, n, args.seed, workers, maia)
        P, lines = summarize(name, df, cands, pc["delta"])
        all_lines += lines
        if name == "player":
            report(cfg, df, cands, P)
    print("\n" + "\n".join(all_lines))


if __name__ == "__main__":
    mp.freeze_support()
    main()
