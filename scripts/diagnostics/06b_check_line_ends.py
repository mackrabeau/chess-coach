"""How settled are the line ends behind each idea?

For the top positions of every atom (and all mistakes as a baseline):
  pend_*   share of line ends where the opponent just captured and we can take back on that square
  dmat     mean material (P1 N3 B3 R5 Q9, mover's POV) at best-line end minus played-line end
  |dmat|>=2  share with a gap of 2+ pawns (large for moves that lose only 0.03-0.20 win prob)
Run after 06; compare --raw (02 lines) with the default (settled lines from 03).

    python scripts/diagnostics/06b_check_line_ends.py --config configs/v0.yaml
    python scripts/diagnostics/06b_check_line_ends.py --config configs/v0.yaml --raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess
import numpy as np
import pandas as pd

from coach.config import data_dir, load_config
from coach.ideas import IdeaModel
from coach.results import load_analyzed

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def end_info(fen: str, line) -> tuple[bool, int]:
    b = chess.Board(fen)
    me, last = b.turn, None
    for u in line:
        m = chess.Move.from_uci(u)
        last = (m, b.is_capture(m))
        b.push(m)
    pend = bool(last and last[1] and any(
        m.to_square == last[0].to_square for m in b.legal_moves if b.is_capture(m)))
    mat = sum(VAL[p.piece_type] * (1 if p.color == me else -1)
              for p in b.piece_map().values() if p.piece_type in VAL)
    return pend, mat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--raw", action="store_true", help="use the unsettled 02 lines")
    ap.add_argument("--top", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)

    model = IdeaModel.load(data_dir(cfg, "ideas") / "model.pkl")
    z = np.load(data_dir(cfg, "ideas") / "pop_codes.npz")
    df = load_analyzed(cfg, "population", raw_lines=args.raw).set_index("pos_id").loc[z["pos_id"]]
    r = pd.DataFrame([(*end_info(f, b), *end_info(f, p))
                      for f, b, p in zip(df.fen, df.best_line, df.played_line)],
                     columns=["pb", "mb", "pp", "mp"])
    r["dmat"] = r.mb - r.mp
    W = z["W"]

    def row(name, ix):
        s = r.iloc[ix]
        return (f"{name:>8}  pend_best {s.pb.mean():.2f}  pend_played {s.pp.mean():.2f}  "
                f"either {(s.pb | s.pp).mean():.2f}  dmat {s.dmat.mean():+.2f}  "
                f"|dmat|>=2 {(s.dmat.abs() >= 2).mean():.2f}")

    print(f"lines: {'raw (02)' if args.raw else 'as used by 05 (settled if quiesce.use)'}")
    print(row("ALL", np.arange(len(r))))
    for a in np.argsort(-model.stability):
        top = np.argsort(-W[:, a])[: args.top]
        d = ", ".join(f"{ar}{n}" for ar, n, _ in model.describe(a, top=2))
        print(row(f"atom {a:2d}", top), f" stab {model.stability[a]:.2f} | {d}")


if __name__ == "__main__":
    main()
