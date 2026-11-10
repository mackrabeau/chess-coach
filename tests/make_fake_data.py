"""Generate small fake inputs (Lichess-style .pgn.zst + chess.com-style archive JSON)
so the pipeline can be smoke-tested without downloading anything.

Games are played by a weak, noisy Stockfish so they contain real mistakes.

    python tests/make_fake_data.py --config tests/test.yaml --games 40
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess.engine
import chess.pgn
import zstandard

from coach.config import data_dir, load_config, resolve


def play_game(eng, rng: random.Random, base: int, inc: int) -> chess.pgn.Game:
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    clocks = {chess.WHITE: float(base), chess.BLACK: float(base)}
    while not board.is_game_over() and board.ply() < 90:
        if rng.random() < 0.25:
            move = rng.choice(list(board.legal_moves))
        else:
            move = eng.play(board, chess.engine.Limit(nodes=rng.choice([200, 2000, 20000]))).move
        spent = rng.uniform(1, 25)
        clocks[board.turn] = max(0.0, clocks[board.turn] - spent + inc)
        node = node.add_variation(move)
        node.set_clock(clocks[board.turn])
        board.push(move)
    game.headers["Result"] = board.result(claim_draw=True)
    return game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tests/test.yaml")
    ap.add_argument("--games", type=int, default=40)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = random.Random(1)
    eng = chess.engine.SimpleEngine.popen_uci(str(resolve(cfg, cfg["paths"]["stockfish"])))

    # Lichess-style dump, mostly in-band rapid games plus some that must be filtered out.
    buf = io.StringIO()
    for i in range(args.games):
        tc = rng.choice(["600+0", "600+5", "900+10", "180+2"])  # 180+2 is blitz -> filtered
        base, inc = map(int, tc.split("+"))
        g = play_game(eng, rng, base, inc)
        g.headers["Event"] = "Rated Rapid game" if base >= 600 else "Rated Blitz game"
        g.headers["Site"] = f"https://lichess.org/fake{i:05d}"
        g.headers["TimeControl"] = tc
        g.headers["WhiteElo"] = str(rng.randint(1980, 2220))
        g.headers["BlackElo"] = str(rng.randint(1980, 2220))
        g.headers["Termination"] = "Normal"
        print(g, file=buf, end="\n\n")
    dump = resolve(cfg, cfg["paths"]["lichess_dump"])
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue().encode()))

    # chess.com-style monthly archive for the player.
    user = cfg["player"]["username"]
    games = []
    for i in range(args.games // 2):
        g = play_game(eng, rng, 600, 0)
        white = i % 2 == 0
        g.headers["White"] = user if white else "someone"
        g.headers["Black"] = "someone" if white else user
        g.headers["TimeControl"] = "600"
        pgn = str(g)
        games.append({
            "url": f"https://www.chess.com/game/live/{1000 + i}",
            "uuid": f"uuid-{i}",
            "pgn": pgn,
            "time_control": "600",
            "time_class": "rapid" if i % 5 else "blitz",  # some get filtered
            "rules": "chess",
            "rated": True,
            "white": {"username": g.headers["White"], "rating": 1900 if i % 4 else 1600},
            "black": {"username": g.headers["Black"], "rating": 1900 if i % 4 else 1600},
        })
    arch = data_dir(cfg, "player", "archives")
    (arch / "2026-10.json").write_text(json.dumps({"games": games}), encoding="utf-8")
    eng.quit()
    print(f"wrote {dump} and {arch / '2026-10.json'}")


if __name__ == "__main__":
    main()
