"""Opening tree helpers shared by 14_openings and 18_opening_traps."""
from __future__ import annotations

import json
import re

import chess
import pandas as pd

from .config import data_dir
from .parts import read_parts
from .fetch_chesscom import iter_player_games
from .mine import mover_score

_COMMENT = re.compile(r"\{[^}]*\}|\([^)]*\)|;[^\n]*")
_SKIP = re.compile(r"^(\d+\.+|\$\d+|1-0|0-1|1/2-1/2|\*)$")


def san_tokens(text: str) -> list[str]:
    body = "\n".join(l for l in text.splitlines() if not l.startswith("["))
    out = []
    for t in _COMMENT.sub(" ", body).split():
        t = re.sub(r"^\d+\.+", "", t).rstrip("!?")
        if t and not _SKIP.match(t):
            out.append(t)
    return out


def family(name: str) -> str:
    if not name:
        return "unknown"
    m = re.match(r"^(.*?\b(?:Defense|Defence|Opening|Game|Gambit|Attack|System|Variation))\b", name)
    return m.group(1) if m else " ".join(name.split()[:2])


def chesscom_name(g: dict) -> str:
    url = g.get("eco") or ""
    tail = url.rstrip("/").split("/")[-1]
    tail = re.split(r"-\d", tail)[0]
    return tail.replace("-", " ")


def band_labels(bands):
    return [f"{a}-{b - 1}" if b < 4000 else f"{a}+" for a, b in zip(bands[:-1], bands[1:])]


def player_tree(cfg, oc):
    pcfg = cfg["player"]
    user = pcfg["username"].lower()
    paths = sorted(data_dir(cfg, "player", "archives").glob("*.json"))
    rows, outs = [], []
    for g in iter_player_games(paths, pcfg):
        white = g["white"]["username"].lower() == user
        if not white and g["black"]["username"].lower() != user:
            continue
        gid = g.get("uuid") or g["url"].rstrip("/").split("/")[-1]
        hdr = dict(re.findall(r'^\[(\w+)\s+"(.*)"\]\s*$', g["pgn"], flags=re.M))
        res = mover_score(hdr.get("Result"), white)
        rating = g["white" if white else "black"].get("rating", 0)
        name = chesscom_name(g)
        b = chess.Board()
        path = []
        try:
            toks = san_tokens(g["pgn"])
            for ply, t in enumerate(toks[: max(oc["max_ply"], oc["out_ply"] + 2)]):
                if ply == oc["out_ply"] + (0 if white else 1):
                    outs.append({"game_id": gid, "color": "white" if white else "black", "rating": rating,
                                 "result": res, "date": hdr.get("Date", ""), "family": family(name),
                                 "fen": b.fen()})
                if ply < oc["max_ply"]:
                    epd, fen, mine = b.epd(), b.fen(), (b.turn == chess.WHITE) == white
                    mv = b.push_san(t)
                    rows.append({"game_id": gid, "color": "white" if white else "black", "rating": rating,
                                 "result": res, "date": hdr.get("Date", ""), "name": name, "family": family(name),
                                 "ply": ply, "epd": epd, "fen": fen, "move": mv.uci(), "san": t, "mine": mine,
                                 "path": " ".join(path)})
                    path.append(t)
                else:
                    b.push_san(t)
        except ValueError:
            continue
    return pd.DataFrame(rows), pd.DataFrame(outs)


def engine_dir_name(nodes: int) -> str:
    """Opening engine cache per search size ("engine" = the original 300k-node pass)."""
    return "engine" if nodes == 300000 else f"engine_{nodes}"


def opening_engine(cfg) -> dict:
    """pos_id -> (best, {move: wp}) from every opening engine cache; the deepest search wins."""
    base = data_dir(cfg, "openings")
    dirs = []
    for p in base.glob("engine*"):
        marker = p / "nodes.txt"                    # written by 14; the name is only a fallback
        if marker.exists():
            n = int(marker.read_text().strip())
        elif p.name == "engine":
            n = 300000
        else:
            n = int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else None
        if n:
            dirs.append((n, p))
    out = {}
    for _, p in sorted(dirs):
        E = read_parts(p)
        if E is not None:
            out.update({i: (b, json.loads(v)) for i, b, v in zip(E.pos_id, E.best, E["values"])})
    return out
