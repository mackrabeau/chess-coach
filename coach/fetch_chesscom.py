"""Download a player's games from the chess.com public API (no auth needed).

Monthly archives are cached as JSON, one file per month. Past months never change,
so they're only downloaded once; the current month is always refreshed.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import requests

API = "https://api.chess.com/pub/player/{user}/games/archives"
HEADERS = {
    "User-Agent": "chess-coach/0.1 (personal research)",
    "Accept": "application/json",
}


def _get(url: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:  # rate limited: back off
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"giving up on {url}")


def fetch_archives(username: str, out_dir: Path) -> list[Path]:
    # chess.com asks for a descriptive User-Agent; name the account whose own games are fetched
    HEADERS["User-Agent"] = f"chess-coach/0.1 (personal research; chess.com user {username.lower()})"
    out_dir.mkdir(parents=True, exist_ok=True)
    archives = _get(API.format(user=username.lower()))["archives"]
    now = dt.datetime.now(dt.timezone.utc)
    current = f"{now.year:04d}/{now.month:02d}"
    paths = []
    for url in archives:
        ym = "/".join(url.rstrip("/").split("/")[-2:])  # "YYYY/MM"
        p = out_dir / f"{ym.replace('/', '-')}.json"
        if not p.exists() or ym == current:
            data = _get(url)
            p.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.5)  # sequential + polite; chess.com rate-limits parallel requests
        paths.append(p)
    return paths


def iter_player_games(archive_paths: list[Path], pcfg: dict):
    """Yield (game_json) for games matching the player config (rapid, rated, standard)."""
    for p in archive_paths:
        for g in json.loads(p.read_text(encoding="utf-8")).get("games", []):
            if g.get("rules") != "chess":
                continue
            if g.get("time_class") != pcfg["time_class"]:
                continue
            if pcfg.get("rated_only", True) and not g.get("rated", False):
                continue
            if not g.get("pgn"):
                continue
            yield g
