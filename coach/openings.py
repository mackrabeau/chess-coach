"""Opening tree helpers shared by 14_openings and 18_opening_traps."""
from __future__ import annotations

import json
import multiprocessing as mp
import re
import time

import chess
import pandas as pd
import pyarrow as pa

from .config import data_dir, resolve
from .engine import Analyzer
from .parts import done_ids, next_part_index, read_parts, write_part

ENGINE_SCHEMA = pa.schema([("pos_id", pa.string()), ("fen", pa.string()), ("best", pa.string()),
                           ("values", pa.string())])
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


def opening_engine(cfg, with_nodes: bool = False) -> dict:
    """pos_id -> (best, {move: wp}) from every opening engine cache; the deepest search wins.
    with_nodes: values are (best, {move: wp}, nodes) instead."""
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
            out.update({i: (b, json.loads(v), n) for i, b, v in zip(E.pos_id, E.best, E["values"])})
    return out if with_nodes else {k: v[:2] for k, v in out.items()}


_an = None
_nodes = 0


def _init(ecfg, sf, nodes):
    global _an, _nodes
    _an, _nodes = Analyzer(ecfg, sf), nodes


def _eval(item):
    pid, fen, moves, mpv = item
    try:
        best, vals = _an.move_values(fen, moves, _nodes, mpv)
        return {"pos_id": pid, "fen": fen, "best": best, "values": json.dumps(vals)}
    except Exception as e:
        print(f"  engine failed on {pid}: {e}", flush=True)
        return None


def evaluate(cfg, items, nodes, workers, with_nodes=False):
    """Stockfish (resumable) on items = [(pos_id, fen, moves to value, multipv)] at `nodes`, cached in
    data/openings/<engine_dir_name(nodes)>/. Returns opening_engine(cfg) (deepest result per position)."""
    part_dir = data_dir(cfg, "openings", engine_dir_name(nodes))
    (part_dir / "nodes.txt").write_text(str(nodes))
    done = done_ids(part_dir)
    todo = [it for it in items if it[0] not in done]
    print(f"  engine: {len(items) - len(todo):,} done, {len(todo):,} to go ({nodes:,} nodes)")
    if todo:
        sf = str(resolve(cfg, cfg["paths"]["stockfish"]))
        idx, buf, t0 = next_part_index(part_dir), [], time.time()
        with mp.Pool(workers, initializer=_init, initargs=(dict(cfg["engine"]), sf, nodes)) as pool:
            for i, r in enumerate(pool.imap_unordered(_eval, todo, chunksize=4), 1):
                if r:
                    buf.append(r)
                if len(buf) >= 200 or i == len(todo):
                    write_part(part_dir, idx, buf, ENGINE_SCHEMA)
                    idx, buf = idx + 1, []
                    el = time.time() - t0
                    print(f"    {i:,}/{len(todo):,}  ETA {el / i * (len(todo) - i) / 60:.0f} min", flush=True)
    return opening_engine(cfg, with_nodes)
