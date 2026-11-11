"""Validate Maia as an opponent model (step 1 of the exploitation idea).

For each analyzed position where we also know what the opponent actually replied:
  you    : the reply is read off your next stored position (same game, ply + 2)
  peers  : the reply is the move of the sampled position at ply + 1 (positions_sae)
we ask Maia, set to the *opponent's* level, from the position after the mover's move
(with history):
  * how likely was the reply actually played?           -> log-lik, top-1
  * how likely was the engine's best reply (2nd move of the settled played line)?
    and did they play it?                               -> calibration table, Brier score
If predicted P(opponent finds the best reply) is well calibrated, it can be used to pick
moves that make the opponent's task hard (the later "practical value" step).

    python scripts/09d_opponent_model.py --config configs/v0.yaml

Output: data/<set>/opp_model.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.difficulty import player_maia_elo
from coach.maia import Maia
from coach.results import load_analyzed


def _key(fen: str) -> str:
    return " ".join(fen.split()[:4])   # board, side to move, castling, en passant


def player_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Your positions with the opponent's reply, from your next stored position."""
    idx = pd.Series(range(len(df)), index=pd.MultiIndex.from_arrays([df.game_id, df.ply]))
    nxt = idx.reindex(pd.MultiIndex.from_arrays([df.game_id, df.ply + 2])).to_numpy()
    rows = []
    for i in np.flatnonzero(~pd.isna(nxt)):
        b = chess.Board(df.fen.iat[i])
        b.push_uci(df.move.iat[i])
        target = _key(df.fen.iat[int(nxt[i])])
        for m in b.legal_moves:
            b.push(m)
            hit = _key(b.fen()) == target
            b.pop()
            if hit:
                rows.append((i, m.uci()))
                break
    return pd.DataFrame(rows, columns=["row", "reply"])


def population_pairs(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Peer positions whose ply + 1 position (the opponent's move) was also sampled."""
    sae = pd.read_parquet(data_dir(cfg, "population") / "positions_sae.parquet",
                          columns=["game_id", "ply", "fen", "move"])
    sae = sae.set_index(["game_id", "ply"])
    want = pd.MultiIndex.from_arrays([df.game_id, df.ply + 1])
    have = want.isin(sae.index)
    rows = []
    nx = sae.loc[want[have]]
    for i, fen_next, reply in zip(np.flatnonzero(have), nx.fen, nx.move):
        b = chess.Board(df.fen.iat[i])
        b.push_uci(df.move.iat[i])
        if _key(b.fen()) == _key(fen_next):    # consistency check
            rows.append((i, reply))
    return pd.DataFrame(rows, columns=["row", "reply"])


def score(maia: Maia, df, pairs, opp_elo, mover_elo, batch: int) -> pd.DataFrame:
    torch = maia.torch
    out = []
    for b0 in range(0, len(pairs), batch):
        sub = pairs.iloc[b0:b0 + batch]
        T, M, ir, ib, keep = [], [], [], [], []
        for k, (i, reply) in enumerate(zip(sub.row, sub.reply)):
            pl = list(df.played_line.iat[i])
            b = chess.Board(df.fen.iat[i])
            b.push_uci(df.move.iat[i])
            fa = b.fen()
            T.append(maia.tokens(fa, (list(df.history_fens.iat[i]) + [df.fen.iat[i]])[-7:]))
            M.append(maia.legal_mask(fa))
            ir.append(maia.index(fa, reply))
            best = pl[1] if len(pl) >= 2 else None
            ib.append(maia.index(fa, best) if best else 0)
            keep.append(best)
        rows = sub.row.to_numpy()
        P = maia.policy(torch.stack(T), torch.stack(M), opp_elo[rows], mover_elo[rows]).cpu().numpy()
        r = np.arange(len(sub))
        for k in r:
            out.append({"pos_id": df.pos_id.iat[rows[k]], "reply": sub.reply.iat[k],
                        "best_reply": keep[k], "opp_elo": float(opp_elo[rows[k]]),
                        "p_reply": float(P[k, ir[k]]),
                        "p_best_reply": float(P[k, ib[k]]) if keep[k] else np.nan,
                        "maia_top": bool(P[k].argmax() == ir[k])})
    o = pd.DataFrame(out)
    o["played_best"] = o.reply == o.best_reply
    return o


def summarize(name: str, o: pd.DataFrame) -> None:
    ok = o.best_reply.notna()
    p, y = o.p_best_reply[ok].to_numpy(), o.played_best[ok].to_numpy(float)
    base = y.mean()
    brier, brier0 = np.mean((p - y) ** 2), np.mean((base - y) ** 2)
    print(f"\n[{name}] {len(o):,} (move, reply) pairs; opponent Elo median {o.opp_elo.median():.0f}")
    print(f"  actual reply: mean log-lik {np.log(np.clip(o.p_reply, 1e-9, 1)).mean():.3f}, "
          f"Maia top-1 = reply {o.maia_top.mean():.1%}")
    print(f"  best reply found {base:.1%} of the time; Maia predicted {p.mean():.1%} on average")
    print(f"  Brier score {brier:.4f} vs {brier0:.4f} for a constant guess "
          f"(skill {1 - brier / brier0:+.1%}; higher = Maia knows when the reply is hard)")
    print("  calibration (Maia's P(opponent finds the best reply) -> how often they did):")
    bins = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001]
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 20:
            print(f"    {a:.0%}-{min(b, 1):.0%}: predicted {p[m].mean():5.1%}, actual {y[m].mean():5.1%}  (n {m.sum():,})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["maia"]
    maia = Maia(mcfg["model"])
    for name in ("player", "population"):
        df = load_analyzed(cfg, name)
        pairs = player_pairs(df) if name == "player" else population_pairs(cfg, df)
        if pairs.empty:
            print(f"[{name}] no (move, reply) pairs found")
            continue
        if name == "player":
            opp = player_maia_elo(df.opp_rating.to_numpy(float), mcfg, df)
            mov = player_maia_elo(df.mover_rating.to_numpy(float), mcfg, df)
        else:
            opp, mov = df.opp_rating.to_numpy(float), df.mover_rating.to_numpy(float)
        o = score(maia, df, pairs, opp, mov, args.batch)
        o.to_parquet(data_dir(cfg, name) / "opp_model.parquet", index=False)
        summarize(name, o)


if __name__ == "__main__":
    main()
