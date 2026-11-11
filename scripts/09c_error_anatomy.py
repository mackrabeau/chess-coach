"""Error anatomy for every mistake / blunder (see coach/anatomy.py), for player and peers.

Material-based depth of the punishment / missed gain, kind of the refuting reply, and Maia's
probability of that reply at the mover's own level (was it on a human's radar?). Your
games also get the seconds spent on each move. One Maia pass per error (minutes on GPU).

    python scripts/09c_error_anatomy.py --config configs/v0.yaml

Output: data/<set>/anatomy.parquet (pos_id + anatomy columns; player also think_s for all moves)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np

from coach.anatomy import anatomy_rows, think_times
from coach.config import data_dir, load_config
from coach.difficulty import own_elo
from coach.maia import Maia
from coach.results import load_analyzed


def reply_probs(maia: Maia, df, rows, elo, batch: int) -> np.ndarray:
    """Maia P(opponent's reply) from the position after the played move, at the mover's Elo."""
    torch = maia.torch
    out = np.full(len(rows), np.nan)
    have = [k for k, (i, r) in enumerate(rows.iterrows()) if r.reply]
    for b0 in range(0, len(have), batch):
        ks = have[b0:b0 + batch]
        T, M, idx = [], [], []
        for k in ks:
            i = rows.index[k]
            b = chess.Board(df.fen.iat[i])
            b.push_uci(df.played_line.iat[i][0])
            fen_after = b.fen()
            hist = (list(df.history_fens.iat[i]) + [df.fen.iat[i]])[-7:]
            T.append(maia.tokens(fen_after, hist))
            M.append(maia.legal_mask(fen_after))
            idx.append(maia.index(fen_after, rows.reply.iat[k]))
        e = np.array([elo[rows.index[k]] for k in ks])
        P = maia.policy(torch.stack(T), torch.stack(M), e, e).cpu().numpy()
        out[ks] = P[np.arange(len(ks)), idx]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config(args.config)
    maia = Maia(cfg["maia"]["model"])
    swing = cfg["ideas"]["tactical_material"]
    for name in ("player", "population"):
        df = load_analyzed(cfg, name)
        rows = anatomy_rows(df, swing)
        elo = own_elo(df, name, cfg["maia"])
        rows["p_reply"] = reply_probs(maia, df, rows, elo, args.batch)
        rows["pos_id"] = df.pos_id.to_numpy()[rows.index]
        out = rows.reset_index(drop=True)
        if name == "player":
            t = think_times(df)
            tt = df[["pos_id"]].assign(think_s=t)
            out = tt.merge(out, on="pos_id", how="left")   # all moves (think time), errors annotated
        path = data_dir(cfg, name) / "anatomy.parquet"
        out.to_parquet(path, index=False)
        e = rows
        print(f"[{name}] {len(e):,} errors; punished next move {np.mean(e.refute_ply <= 2):.0%}, "
              f"ply 3-4 {np.mean((e.refute_ply > 2) & (e.refute_ply <= 4)):.0%}, ply 5+ {np.mean(e.refute_ply > 4):.0%}; "
              f"reply quiet {np.mean(e.reply_kind == 'quiet'):.0%}; median Maia P(reply) {np.nanmedian(e.p_reply):.2f}"
              + (f"; think time known for {np.isfinite(out.think_s).mean():.0%} of moves" if name == "player" else "")
              + f" -> {path}")


if __name__ == "__main__":
    main()
