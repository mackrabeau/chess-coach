"""Probe Maia-3 on our data before building the difficulty pipeline.

Answers three questions on real positions:
  1. Does the 8-board history help? Mean log-likelihood of the move actually played,
     with vs without history, each position at its mover's own Elo (peers: Lichess
     rating; you: maia.player_elo). Use whichever predicts better.
  2. How does Maia respond to Elo near your level? Mean P(engine best move) and how
     often Maia's top move changes, over an Elo grid; self vs opponent Elo sensitivity.
  3. Speed on this machine (GPU if torch sees one).

    python scripts/diagnostics/09a_maia_probe.py --config configs/v0.yaml --n 3000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from coach.config import load_config
from coach.maia import Maia
from coach.results import load_analyzed


def batches(n, size):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--n", type=int, default=3000, help="positions per set")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["maia"]
    maia = Maia(mcfg["model"])
    torch = maia.torch
    print(f"Maia-3 {mcfg['model']} on {maia.device} (amp={maia.amp})")

    for name in ("population", "player"):
        df = load_analyzed(cfg, name)
        df = df.sample(min(args.n, len(df)), random_state=0).reset_index(drop=True)
        n = len(df)
        if name == "population":
            self_elo = df.mover_rating.to_numpy()
            oppo_elo = df.opp_rating.to_numpy()
        else:  # chess.com ratings -> Lichess scale via the calibrated offset
            off = mcfg["player_elo"] - df.mover_rating.median()
            self_elo = np.full(n, mcfg["player_elo"])
            oppo_elo = (df.opp_rating.to_numpy() + off).round()
        masks = torch.stack([maia.legal_mask(f) for f in df.fen])
        T = {h: torch.stack([maia.tokens(f, list(hf) if h else None) for f, hf in zip(df.fen, df.history_fens)])
             for h in (True, False)}
        played = torch.tensor([maia.index(f, m) for f, m in zip(df.fen, df.move)])
        best = torch.tensor([maia.index(f, m) for f, m in zip(df.fen, df.best_move)])
        ar = torch.arange(n)

        # 1. history vs none
        res = {}
        for h in (True, False):
            P = torch.cat([maia.policy(T[h][s], masks[s], self_elo[s], oppo_elo[s]).cpu() for s in batches(n, args.batch)])
            res[h] = P
        ll = {h: res[h][ar, played].clamp_min(1e-9).log().mean().item() for h in res}
        t1 = {h: (res[h].argmax(1) == played).float().mean().item() for h in res}
        agree = (res[True].argmax(1) == res[False].argmax(1)).float().mean().item()
        print(f"\n[{name}] {n:,} positions")
        print(f"  1. log-lik of played move: with history {ll[True]:.4f}, without {ll[False]:.4f} "
              f"(higher is better); top-1 = played {t1[True]:.1%} vs {t1[False]:.1%}; "
              f"top-1 agreement {agree:.0%}")

        # 2. Elo response (history input, self = grid, opponent = 2100)
        grid = list(range(1100, 2701, 100))
        pb, top = [], []
        for e in grid:
            P = torch.cat([maia.policy(T[True][s], masks[s], e, 2100).cpu() for s in batches(n, args.batch)])
            pb.append(P[ar, best].numpy())
            top.append(P.argmax(1).numpy())
        pb, top = np.stack(pb, 1), np.stack(top, 1)
        print("  2. Elo  P(best)  top-1=best")
        for j, e in enumerate(grid):
            if e % 200 == 100 or e in (2000, 2200):
                print(f"     {e}  {pb[:, j].mean():.3f}   {(top[:, j] == best.numpy()).mean():.3f}")
        j1, j2 = grid.index(1900), grid.index(2300)
        print(f"     top-1 changes between 1900 and 2300 in {(top[:, j1] != top[:, j2]).mean():.0%} of positions; "
              f"mean |dP(best)| {np.abs(pb[:, j2] - pb[:, j1]).mean():.3f}")
        P0 = res[True]

        def kl(Q):
            m = P0 > 0
            return (P0[m] * (P0[m].log() - Q[m].clamp_min(1e-12).log())).sum().item() / n
        Ps = torch.cat([maia.policy(T[True][s], masks[s], self_elo[s] - 400, oppo_elo[s]).cpu() for s in batches(n, args.batch)])
        Po = torch.cat([maia.policy(T[True][s], masks[s], self_elo[s], oppo_elo[s] - 400).cpu() for s in batches(n, args.batch)])
        print(f"     KL when self Elo -400: {kl(Ps):.3f}; when opponent Elo -400: {kl(Po):.3f}")

    # 3. speed
    s = slice(0, min(args.batch, n))
    maia.policy(T[True][s], masks[s], 2100, 2100)  # warm-up
    if maia.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0, reps = time.time(), 5
    for _ in range(reps):
        maia.policy(T[True][s], masks[s], 2100, 2100)
    if maia.device.startswith("cuda"):
        torch.cuda.synchronize()
    rate = reps * (s.stop - s.start) / (time.time() - t0)
    print(f"\n3. speed: {rate:,.0f} forward passes/s on {maia.device} "
          f"-> 15 Elo values x 25k positions ~ {15 * 25_000 / rate / 60:.0f} min")


if __name__ == "__main__":
    main()
