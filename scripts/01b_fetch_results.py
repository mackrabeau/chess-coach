"""Fetch final results for the peer games (positions were mined before results were stored).

Uses the public Lichess endpoint POST /api/games/export/_ids (up to 300 ids per request,
one request at a time, waits a minute on HTTP 429). Resumable: ids already in the output
are skipped.

    python scripts/01b_fetch_results.py --config configs/v0.yaml

Output: data/population/results.parquet (game_id, white_score: 1 / 0.5 / 0)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from coach.config import data_dir, load_config

URL = "https://lichess.org/api/games/export/_ids"
HEADERS = {"Accept": "application/x-ndjson",
           "User-Agent": "chess-coach research script (results lookup)"}


def white_score(game: dict) -> float | None:
    """Lichess game JSON -> White's score; None if the game has no result (aborted etc.)."""
    if game.get("status") in ("created", "started", "aborted", "unknownFinish", "noStart"):
        return None
    w = game.get("winner")
    if w == "white":
        return 1.0
    if w == "black":
        return 0.0
    return 0.5   # finished without a winner: draw, stalemate, ...


def fetch_batch(ids: list[str], session) -> dict[str, float]:
    while True:
        r = session.post(URL, data=",".join(ids), headers=HEADERS,
                         params={"moves": "false", "clocks": "false", "evals": "false",
                                 "opening": "false"}, timeout=120)
        if r.status_code == 429:
            print("  rate limited, waiting 60 s", flush=True)
            time.sleep(60)
            continue
        r.raise_for_status()
        out = {}
        for line in r.text.splitlines():
            if line.strip():
                g = json.loads(line)
                s = white_score(g)
                if s is not None:
                    out[g["id"]] = s
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = data_dir(cfg, "population")
    ids = pd.read_parquet(d / "positions_engine.parquet", columns=["game_id"]).game_id.unique().tolist()
    path = d / "results.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["game_id", "white_score"])
    have_ids = set(have.game_id)
    todo = [g for g in ids if g not in have_ids]
    print(f"{len(ids):,} peer games, {len(have):,} results on file, {len(todo):,} to fetch")
    rows = have.to_dict("records")
    with requests.Session() as s:
        for i in range(0, len(todo), 300):
            batch = todo[i:i + 300]
            got = fetch_batch(batch, s)
            # ids without a result (aborted, deleted) are stored as NaN so re-runs skip them
            rows += [{"game_id": k, "white_score": got.get(k, float("nan"))} for k in batch]
            pd.DataFrame(rows).to_parquet(path, index=False)   # save after every batch
            print(f"  {min(i + 300, len(todo)):,}/{len(todo):,} requested, {len(got)} results", flush=True)
            time.sleep(1)
    print(f"done -> {path}")


if __name__ == "__main__":
    main()
