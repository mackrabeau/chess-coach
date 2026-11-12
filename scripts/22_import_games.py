"""Import games from PGN (over-the-board classical, 30+30, any other source) and analyse each one.

    python scripts/22_import_games.py --config configs/v0.yaml --pgn my_otb_games.pgn --name "Rabeau"
    python scripts/22_import_games.py --config configs/v0.yaml            # re-analyse what's imported

--name: text that identifies you in the White/Black tags (case-insensitive; several allowed). Games
without your name are skipped. Imported games are kept separate (data/imported/), never mixed into
the chess.com comparisons; they reuse the rapid peers/bands as reference (no separate classical
model yet). Positions from move 1 (no opening cut), no clock filter (most OTB PGNs have no clocks).

Per game, in the report's Games tab:
  game rating + per phase (opening / middlegame / endgame) from the multi-band lines (rough: one game
  is a small sample), mistakes/blunders, your worst moments as boards, and how the game compares with
  your online rapid games (percentile of loss per move).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from coach.categories import annotate
from coach.config import data_dir, load_config
from coach.findings import example, finding, write as write_findings
from coach.game_rating import level, lines
from coach.mine import SCHEMA, extract_positions
from coach.pgn_stream import iter_raw_games
from coach.results import load_analyzed
from coach.runner import engine_pass

SET = "imported"


def import_pgn(cfg, paths, names, label):
    fcfg = {**cfg["filters"], "min_ply": 0, "min_clock_seconds": None, "min_game_plies": 10}
    d = data_dir(cfg, SET)
    meta_p = d / "games.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    rows, n, skipped = [], 0, 0
    for p in paths:
        for headers, text in iter_raw_games(p):
            w, b = headers.get("White", ""), headers.get("Black", "")
            side = ("white" if any(x.lower() in w.lower() for x in names)
                    else "black" if any(x.lower() in b.lower() for x in names) else None)
            if side is None:
                skipped += 1
                continue
            gid = "imp" + hashlib.sha1(text.encode()).hexdigest()[:12]
            r = extract_positions(text, headers, source=label, game_id=gid, fcfg=fcfg,
                                  mover_ok=lambda iw, s=side: iw == (s == "white"))
            if not r:
                skipped += 1
                continue
            rows += r
            n += 1
            meta[gid] = {"white": w, "black": b, "date": headers.get("Date", ""), "event": headers.get("Event", ""),
                         "result": headers.get("Result", ""), "time_control": headers.get("TimeControl", ""),
                         "color": side, "label": label}
    out = d / "positions_engine.parquet"
    if not rows:
        print(f"no games with your name found ({skipped} skipped); check --name")
        return
    new = pd.DataFrame(rows)
    if out.exists():
        new = pd.concat([pd.read_parquet(out), new], ignore_index=True)
    new = new.drop_duplicates("pos_id")
    pq.write_table(pa.Table.from_pandas(new, schema=SCHEMA, preserve_index=False), out)
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"imported {n} games ({skipped} skipped: your name not found or too short); {len(new):,} positions in total")


def analyse(cfg):
    d = data_dir(cfg, SET)
    if not (d / "positions_engine.parquet").exists():
        print("nothing imported yet (python scripts/22_import_games.py --pgn games.pgn --name \"Your Name\")")
        return
    meta = json.loads((d / "games.json").read_text())
    df = annotate(load_analyzed(cfg, SET, raw_lines=True, refined=False))
    online = load_analyzed(cfg, "player", raw_lines=True)
    online = online[online.mover_rating >= cfg["player"]["min_player_rating"]]
    online_pg = online.groupby("game_id").wp_loss.mean().to_numpy()
    fits = lines(cfg)
    items = []
    for gid, g in df.groupby("game_id"):
        m = meta.get(gid, {})
        rows, lv = [], {}
        for ph in ("opening", "middlegame", "endgame"):
            s = g[g.ph == ph]
            if len(s):
                cmp = s[s.ply >= 20]          # the bands start at move 10: same window for the level
                lv[ph] = level(cmp.wp_loss.mean(), fits.get(ph)) if len(cmp) >= 5 else np.nan
                rows.append([ph, len(s), f"{100 * s.wp_loss.mean():.1f}", int((s.label == "mistake").sum()),
                             int((s.label == "blunder").sum()), f"~{lv[ph]:.0f}" if np.isfinite(lv[ph]) else "–"])
        overall = level(g.wp_loss.mean(), fits.get("all"))
        pct = float((online_pg > g.wp_loss.mean()).mean()) if len(online_pg) else np.nan
        worst = g[g.err.to_numpy()].sort_values("wp_loss", ascending=False).head(3)
        exs = [example(x.fen, x.best_line, x.played_line, f"move {x.ply // 2 + 1} · lost {100 * x.wp_loss:.0f} points",
                       x.pos_id)
               for x in worst.itertuples()]
        opp = m.get("black") if m.get("color") == "white" else m.get("white")
        title = f"{m.get('date', '')} · {m.get('event', '')} · vs {opp} ({m.get('result', '')})"
        claim = (f"Game rating <b>~{overall:.0f}</b> (Lichess rapid scale, rough). " if np.isfinite(overall) else "") + \
                (f"Fewer losses per move than {pct:.0%} of your online rapid games." if np.isfinite(pct) else "")
        items.append(finding(f"game:{gid}", "games", title, claim, "info",
                             evidence={"header": ["phase", "your moves", "loss/100", "mistakes", "blunders", "level"],
                                       "rows": rows}, examples=exs,
                             drill={"theme": f"imported game {m.get('date', '')}",
                                    "pos_ids": worst[worst.critical.astype(bool)].pos_id.tolist()}))
    items.sort(key=lambda f: f["title"], reverse=True)
    write_findings(cfg, "imported", items)
    print(f"{len(items)} imported games analysed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--pgn", nargs="*", default=[], help="PGN file(s) or folder(s)")
    ap.add_argument("--name", nargs="*", default=None, help="your name as it appears in the PGN tags")
    ap.add_argument("--label", default="otb", help="source label for these games (otb, classical, ...)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.pgn:
        paths = []
        for p in map(Path, args.pgn):
            paths += sorted(p.glob("*.pgn")) if p.is_dir() else [p]
        names = args.name or cfg["player"].get("otb_names") or [cfg["player"]["username"]]
        import_pgn(cfg, paths, names, args.label)
    if (data_dir(cfg, SET) / "positions_engine.parquet").exists():
        engine_pass(cfg, SET, workers=args.workers)
    analyse(cfg)


if __name__ == "__main__":
    mp.freeze_support()
    main()
