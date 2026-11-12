"""Maia-3 difficulty pass + best-fit Elo.

For every analyzed position (player + population), with the 8-board history, compute
Maia's probability of the engine's best move and of the move actually played, at each
self-Elo on a grid (maia.grid, default 1100..2500 step 100). Opponent Elo = the real
opponent's (Lichess peers) or theirs shifted onto the Lichess scale (you); it barely
matters (probe: KL ~0.003 for -400). Resumable: parquet parts, skips done pos_ids.

Then:
  * best-fit Elo = the grid Elo that maximizes the summed log-likelihood of the played
    moves (quadratic refinement around the max). Validated on peers per rating bin
    (should land near their real Lichess ratings), then applied to you, with a 90%
    interval from resampling your games. This is a data-driven chess.com -> Lichess
    conversion to replace the maia.player_elo guess.

    python scripts/09b_maia_difficulty.py --config configs/v0.yaml

Output: data/<set>/maia_<lo>_<hi>_<step>/part-*.parquet (pos_id, p_best[grid], p_played[grid])
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow as pa

from coach.config import data_dir, load_config
from coach.difficulty import at_elo, grid_of, maia_dir, player_maia_elo
from coach.maia import Maia
from coach.parts import done_ids, next_part_index, read_parts, write_part
from coach.results import load_analyzed

SCHEMA = pa.schema([("pos_id", pa.string()),
                    ("p_best", pa.list_(pa.float32())), ("p_played", pa.list_(pa.float32()))])


def opponent_elos(df, name, mcfg) -> np.ndarray:
    if name == "population":
        return df.opp_rating.to_numpy()
    return player_maia_elo(df.opp_rating.to_numpy(float), mcfg, df).round()


def compute(maia: Maia, cfg: dict, name: str, chunk: int, batch: int) -> None:
    mcfg = cfg["maia"]
    grid = grid_of(mcfg)
    torch = maia.torch
    df = load_analyzed(cfg, name)
    out = maia_dir(cfg, name)
    done = done_ids(out)
    todo = df[~df.pos_id.isin(done)].reset_index(drop=True)
    print(f"[{name}] {len(df):,} positions, {len(done):,} done, {len(todo):,} to go; "
          f"{len(grid)} Elo values {grid[0]}-{grid[-1]}")
    if todo.empty:
        return
    opp = opponent_elos(todo, name, mcfg)
    idx = next_part_index(out)
    t0 = time.time()
    G = len(grid)
    for c0 in range(0, len(todo), chunk):
        sub = todo.iloc[c0:c0 + chunk]
        T = torch.stack([maia.tokens(f, list(h)) for f, h in zip(sub.fen, sub.history_fens)])
        M = torch.stack([maia.legal_mask(f) for f in sub.fen])
        bi = np.array([maia.index(f, m) for f, m in zip(sub.fen, sub.best_move)])
        pi = np.array([maia.index(f, m) for f, m in zip(sub.fen, sub.move)])
        o = opp[c0:c0 + chunk]
        pb = np.zeros((len(sub), G), np.float32)
        pp = np.zeros((len(sub), G), np.float32)
        for b0 in range(0, len(sub), batch):
            s = slice(b0, min(b0 + batch, len(sub)))
            n = s.stop - s.start
            # every position x every grid Elo in one forward pass
            Tb = T[s].repeat_interleave(G, 0)
            Mb = M[s].repeat_interleave(G, 0)
            se = np.tile(grid, n)
            oe = np.repeat(o[s], G)
            P = maia.policy(Tb, Mb, se, oe).cpu().numpy().reshape(n, G, -1)
            r = np.arange(n)
            pb[s] = P[r, :, bi[s]]
            pp[s] = P[r, :, pi[s]]
        write_part(out, idx, [{"pos_id": p, "p_best": b.tolist(), "p_played": q.tolist()}
                              for p, b, q in zip(sub.pos_id, pb, pp)], SCHEMA)
        idx += 1
        done_n = c0 + len(sub)
        el = time.time() - t0
        print(f"  {done_n:,}/{len(todo):,}  {done_n * G / el:,.0f} passes/s  "
              f"ETA {el / done_n * (len(todo) - done_n) / 60:.0f} min", flush=True)


def best_fit(ll_sum: np.ndarray, grid: np.ndarray) -> float:
    """Argmax of a summed log-likelihood curve, refined with a parabola through the peak."""
    k = int(np.argmax(ll_sum))
    if 0 < k < len(grid) - 1:
        y0, y1, y2 = ll_sum[k - 1], ll_sum[k], ll_sum[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(grid[k] + 0.5 * (y0 - y2) / den * (grid[1] - grid[0]))
    return float(grid[k])


def fit_offset(L: np.ndarray, rating: np.ndarray, game: np.ndarray, grid: np.ndarray,
               offsets=np.arange(-300, 501, 5), slopes=(0.0,), n_boot: int = 300):
    """(c0, c1) maximizing sum_i log P(played_i | Elo = r_i + c0 + c1 (r_i - 1800)): each game is
    scored at the player's rating when it was played, mapped onto Maia's (Lichess) scale.
    Returns (c0, c1, c0_lo, c0_hi, c1_lo, c1_hi) with a game-bootstrap 90% interval."""
    games, g = np.unique(game, return_inverse=True)
    combos = [(c0, c1) for c1 in slopes for c0 in offsets]
    per_game = np.zeros((len(games), len(combos)))
    for j, (c0, c1) in enumerate(combos):
        np.add.at(per_game[:, j], g, at_elo(L, rating + c0 + c1 * (rating - 1800), grid))
    best = lambda tot: combos[int(np.argmax(tot))]
    rng = np.random.default_rng(0)
    boots = np.array([best(per_game[rng.integers(0, len(games), len(games))].sum(0)) for _ in range(n_boot)])
    c0, c1 = best(per_game.sum(0))
    (a0, a1), (b0, b1) = np.percentile(boots, 5, axis=0), np.percentile(boots, 95, axis=0)
    return c0, c1, a0, b0, a1, b1


def load_maia(cfg: dict, name: str):
    """Analyzed positions joined with their Maia curves (rows without curves dropped)."""
    df = load_analyzed(cfg, name)
    q = read_parts(maia_dir(cfg, name))
    if q is None:
        return None
    return df.merge(q, on="pos_id", how="inner")


def fit_report(cfg: dict) -> None:
    grid = grid_of(cfg["maia"])
    pop = load_maia(cfg, "population")
    me = load_maia(cfg, "player")
    for d in (pop, me):
        if d is not None:
            d["ll"] = [np.log(np.clip(np.asarray(v, float), 1e-9, 1)) for v in d.p_played]

    if pop is not None:
        print("\nbest-fit Elo on peers (validation: should track their real Lichess rating)")
        bins = [2000, 2050, 2100, 2150, 2201]
        pop["bin"] = np.digitize(pop.mover_rating, bins) - 1
        for b in range(len(bins) - 1):
            sub = pop[pop.bin == b]
            if len(sub) < 200:
                continue
            fit = best_fit(np.sum(np.stack(sub.ll), 0), grid)
            print(f"  real {bins[b]}-{bins[b + 1] - 1}: mean {sub.mover_rating.mean():.0f}, "
                  f"best fit {fit:.0f}  (n {len(sub):,})")
        print(f"  all peers: mean {pop.mover_rating.mean():.0f}, best fit "
              f"{best_fit(np.sum(np.stack(pop.ll), 0), grid):.0f}")

    if me is not None:
        L = np.stack(me.ll)
        fit = best_fit(L.sum(0), grid)
        games, g = np.unique(me.game_id.to_numpy(), return_inverse=True)
        per_game = np.zeros((len(games), len(grid)))
        np.add.at(per_game, g, L)
        rng = np.random.default_rng(0)
        boots = [best_fit(per_game[rng.integers(0, len(games), len(games))].sum(0), grid)
                 for _ in range(300)]
        lo, hi = np.percentile(boots, [5, 95])
        edge = " (at the grid edge: widen maia.grid)" if fit in (grid[0], grid[-1]) else ""
        print(f"\nyou: best-fit Lichess-scale Elo {fit:.0f}  [90% {lo:.0f}, {hi:.0f}]{edge}  "
              f"(chess.com median {me.mover_rating.median():.0f}; config maia.player_elo "
              f"{cfg['maia']['player_elo']})")
        r = me.mover_rating.to_numpy(float)
        c, _, clo, chi, _, _ = fit_offset(L, r, me.game_id.to_numpy(), grid)
        when = pd.to_datetime(me.date.astype(str).str.replace(".", "-", regex=False), errors="coerce")
        order = np.argsort(when.to_numpy()) if when.notna().any() else np.arange(len(me))
        recent = r[order][-max(1, len(me) // 10):]   # latest 10% of moves
        print(f"you, game by game: Maia Elo = chess.com rating + {c:.0f}  [90% {clo:.0f}, {chi:.0f}]  "
              f"(config maia.player_offset {cfg['maia'].get('player_offset')}); "
              f"at your latest rating ~{np.median(recent):.0f} that is ~{np.median(recent) + c:.0f}")
        fitted = {"offset": float(c), "slope": 0.0, "kind": "constant"}
        if r.max() - r.min() >= 300:   # wide rating range: let the offset depend on rating
            c0, c1, a0, b0, a1, b1 = fit_offset(L, r, me.game_id.to_numpy(), grid,
                                                offsets=np.arange(-300, 601, 10),
                                                slopes=np.round(np.arange(-0.5, 0.51, 0.05), 2), n_boot=200)
            ex = ", ".join(f"{x} -> {x + c0 + c1 * (x - 1800):.0f}" for x in (1000, 1400, 1800, 1900))
            print(f"you, rating-dependent: Maia Elo = r + {c0:.0f} + {c1:.2f} (r - 1800)  "
                  f"[c0 {a0:.0f}..{b0:.0f}, slope {a1:.2f}..{b1:.2f}]  ({ex}); "
                  f"set maia.player_offset / player_offset_slope from this")
            fitted = {"offset": float(c0), "slope": float(c1), "kind": "rating-dependent"}
            print("  check per rating band (best fit vs mapped):")
            for lo, hi in ((0, 1200), (1200, 1500), (1500, 1750), (1750, 1850), (1850, 9999)):
                m = (r >= lo) & (r < hi)
                if me.game_id[m].nunique() >= 20:
                    print(f"    chess.com {lo}-{min(hi, 3000) - 1}: median {np.median(r[m]):.0f}, best fit "
                          f"{best_fit(L[m].sum(0), grid):.0f}, mapped {np.median(r[m]) + c0 + c1 * (np.median(r[m]) - 1800):.0f}"
                          f"  ({me.game_id[m].nunique()} games)")
        # saved per player; load_config uses it instead of maia.player_offset(_slope) from now on
        (data_dir(cfg, "player") / "maia_offset.json").write_text(json.dumps(fitted), encoding="utf-8")
        print(f"saved the fitted offset ({fitted['kind']}: {fitted['offset']:.0f}, slope {fitted['slope']:.2f}) "
              "-> used automatically by every later stage")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--set", choices=["population", "player", "both"], default="both")
    ap.add_argument("--chunk", type=int, default=2000, help="positions per saved part")
    ap.add_argument("--batch", type=int, default=None, help="positions per forward pass (x grid size)")
    ap.add_argument("--fit-only", action="store_true", help="skip compute, just report best-fit Elo")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["maia"]
    if not args.fit_only:
        maia = Maia(mcfg["model"])
        batch = args.batch or mcfg.get("batch", 64 if maia.device.startswith("cuda") else 16)
        print(f"Maia-3 {mcfg['model']} on {maia.device} (amp={maia.amp}), "
              f"{batch} positions x {len(grid_of(mcfg))} Elos per pass")
        sets = ["player", "population"] if args.set == "both" else [args.set]
        for name in sets:
            compute(maia, cfg, name, args.chunk, batch)
    fit_report(cfg)


if __name__ == "__main__":
    main()
