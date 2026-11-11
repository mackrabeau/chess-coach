"""Download the player's chess.com games and mine the player's decision positions.

Output: data/player/positions.parquet (only moves made by the player, only games
where the player's rating >= player.fetch_min_rating, standard filters applied).
fetch_min_rating defaults to 0 (whole history, used for progress over time);
comparisons with the peer band use only games >= player.min_player_rating (08).

    python scripts/00_fetch_player.py --config configs/v0.yaml
    python scripts/00_fetch_player.py --config configs/v0.yaml --offline   # reuse cached archives
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach.config import data_dir, load_config
from coach.fetch_chesscom import fetch_archives, iter_player_games
from coach.mine import ParquetSink, extract_positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--offline", action="store_true", help="don't hit the API, use cached archives")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pcfg, fcfg = cfg["player"], cfg["filters"]
    user = pcfg["username"].lower()
    arch_dir = data_dir(cfg, "player", "archives")

    if args.offline:
        paths = sorted(arch_dir.glob("*.json"))
    else:
        paths = fetch_archives(user, arch_dir)
    print(f"{len(paths)} monthly archives")

    out = data_dir(cfg, "player") / "positions.parquet"
    sink = ParquetSink(out, flush_rows=20_000)
    n_games = n_used = n_skipped_rating = 0
    for g in iter_player_games(paths, pcfg):
        n_games += 1
        is_white = g["white"]["username"].lower() == user
        if not is_white and g["black"]["username"].lower() != user:
            continue
        # chess.com JSON rating is the player's rating after the game; close enough for a cutoff.
        rating = g["white" if is_white else "black"].get("rating", 0)
        if rating < pcfg.get("fetch_min_rating", pcfg["min_player_rating"]):
            n_skipped_rating += 1
            continue
        game_id = g.get("uuid") or g["url"].rstrip("/").split("/")[-1]
        headers = {
            "TimeControl": g.get("time_control", ""),
            "WhiteElo": str(g["white"].get("rating", -1)),
            "BlackElo": str(g["black"].get("rating", -1)),
            **_headers(g["pgn"]),  # PGN headers win when present
        }
        rows = extract_positions(
            g["pgn"],
            headers,
            source="chesscom",
            game_id=game_id,
            fcfg=fcfg,
            mover_ok=lambda w, iw=is_white: w == iw,
        )
        if rows:
            n_used += 1
            sink.add(rows)
    sink.close()
    print(
        f"{n_games} {pcfg['time_class']} games; {n_skipped_rating} below rating cutoff; "
        f"{n_used} games used; {sink.n} positions -> {out}"
    )


def _headers(pgn: str) -> dict[str, str]:
    import re

    return dict(re.findall(r'^\[(\w+)\s+"(.*)"\]\s*$', pgn, flags=re.M))


if __name__ == "__main__":
    main()
