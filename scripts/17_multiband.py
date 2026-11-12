"""Multi-band peers: which measures change with rating (skill) and which don't (style), and
where you sit on each.

  1. mine   Lichess rapid games from the local dump in several rating bands (both players
            inside the band), multiband.per_game positions per game, until multiband.target
            positions per band -> data/band_<lo>/positions_engine.parquet. The existing peer set
            (population, 2000-2200) is the middle band.
  2. engine Stockfish on every band position with the same settings as the peers (resumable,
            same as 02). Several hours for 4 x 4,000 positions at 1M nodes; it can be stopped and
            restarted any time.
  3. fit    per category (phase, eval band, critical, best-move kind, clock, endgame type, ...)
            and metric (error rate, blunder rate, loss per 100 moves): band values vs the bands'
            mean rating, weighted line  m = a + s (r - 2100). Game bootstrap in every band.
              skill-relevant: the slope's 90% CI excludes 0
              your level: invert the line at your value (games >= player.min_player_rating)
              -> "plays like Lichess ~X", with a 90% CI; relative to your level on all moves.
            Flat measures are style: no implied level (dividing by ~0 slope is noise).

Caveats: you are chess.com 10|0, the bands are Lichess rapid; you never resign, which lowers
error rates in lost positions (use the contested rows); implied levels outside the band range
are extrapolations and flagged.

    python scripts/17_multiband.py --config configs/v0.yaml                # all stages
    python scripts/17_multiband.py --config configs/v0.yaml --stage mine   # one stage
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coach.bands import band_name, band_ranges, resolve_sets, settings
from coach.categories import annotate, categories
from coach.config import data_dir, load_config, resolve
from coach.mine import ParquetSink, _int, extract_positions, is_rapid
from coach.pgn_stream import iter_raw_games
from coach.results import load_analyzed
from coach.runner import engine_pass

METRICS = {"error rate": "err", "blunder rate": "blunder", "loss per 100 moves": "wp_loss"}


# ---------------------------------------------------------------- 1. mine
def mine(cfg, mb):
    """Mine until every band has multiband.target positions. Incremental: existing positions are
    kept, their games skipped, so raising the target (refine.py does) just adds more."""
    mb = {**mb, "bands": band_ranges(cfg, mb)}
    have, seen = {}, set()
    for lo, _ in mb["bands"]:
        f = data_dir(cfg, band_name(lo)) / "positions_engine.parquet"
        old = pd.read_parquet(f) if f.exists() else None
        have[lo] = old
        if old is not None:
            seen |= set(old.game_id)
    todo = [(lo, hi) for lo, hi in mb["bands"] if have[lo] is None or len(have[lo]) < mb["target"]]
    if not todo:
        print("mine: all bands at target")
        return
    fcfg = cfg["filters"]
    rng = random.Random(cfg["population"]["seed"] + len(seen))
    tmp = {lo: data_dir(cfg, band_name(lo)) / "positions_new.parquet.tmp" for lo, _ in todo}
    sinks = {lo: ParquetSink(tmp[lo], flush_rows=5_000) for lo, _ in todo}
    count = {lo: 0 if have[lo] is None else len(have[lo]) for lo, _ in todo}
    t0, n_seen = time.time(), 0
    for headers, text in iter_raw_games(resolve(cfg, cfg["paths"]["lichess_dump"])):
        n_seen += 1
        if n_seen % 1_000_000 == 0:
            print(f"  scanned {n_seen:,} ({n_seen / (time.time() - t0):,.0f}/s): "
                  + ", ".join(f"{lo}: {c:,}" for lo, c in count.items()), flush=True)
        if "Rated" not in headers.get("Event", "") or not is_rapid(headers.get("TimeControl"), fcfg):
            continue
        if headers.get("Termination", "Normal") not in ("Normal", "Time forfeit"):
            continue
        w, b = _int(headers.get("WhiteElo")), _int(headers.get("BlackElo"))
        band = next((lo for lo, hi in todo if lo <= w < hi and lo <= b < hi and count[lo] < mb["target"]), None)
        if band is None:
            continue
        gid = headers.get("Site", "").rstrip("/").split("/")[-1] or f"g{n_seen}"
        if gid in seen:
            continue
        rows = extract_positions(text, headers, source="lichess", game_id=gid, fcfg=fcfg,
                                 mover_ok=lambda _w: True, rng=rng, max_per_game=mb["per_game"])
        if rows:
            sinks[band].add(rows)
            count[band] += len(rows)
        if all(count[lo] >= mb["target"] for lo, _ in todo):
            break
    for lo, s in sinks.items():
        s.close()
        f = data_dir(cfg, band_name(lo)) / "positions_engine.parquet"
        new = pd.read_parquet(tmp[lo]) if tmp[lo].exists() and s.n else None
        parts = [d for d in (have[lo], new) if d is not None]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(f.with_suffix(".parquet.tmp2"), index=False)
            f.with_suffix(".parquet.tmp2").replace(f)          # atomic: a kill never leaves a broken file
        tmp[lo].unlink(missing_ok=True)
    print(f"mine: scanned {n_seen:,} games; " + ", ".join(f"band {lo}: {c:,} positions" for lo, c in count.items()))


# ---------------------------------------------------------------- 3. fit
def load_set(cfg, which):
    try:
        # bands are never refined: compare everything at the base search depth
        df = load_analyzed(cfg, which, raw_lines=True, refined=False)
    except FileNotFoundError:
        return None
    return annotate(df)


def per_game(df, masks, col):
    """games x categories arrays: (moves, sum of metric)."""
    gids, g = np.unique(df.game_id.to_numpy(), return_inverse=True)
    vals = df[col].to_numpy(float)
    M, S = np.zeros((len(gids), len(masks))), np.zeros((len(gids), len(masks)))
    for j, m in enumerate(masks):
        np.add.at(M[:, j], g[m], 1)
        np.add.at(S[:, j], g[m], vals[m])
    return M, S


def fit_line(r, m, w):
    """Weighted least squares m = a + s (r - 2100) -> (a, s)."""
    x = r - 2100
    W = np.sum(w)
    xm, ym = np.sum(w * x) / W, np.sum(w * m) / W
    s = np.sum(w * (x - xm) * (m - ym)) / max(np.sum(w * (x - xm) ** 2), 1e-9)
    return ym - s * xm, s


def analyse(cfg, mb):
    sets = resolve_sets(cfg, mb)
    bands = []
    for name, rng_ in sets:
        d = load_set(cfg, name)
        if d is None:
            print(f"  {name}: no engine results yet, skipped")
            continue
        bands.append((name, rng_, d))
    bands.sort(key=lambda t: t[1][0])
    if len(bands) < 3:
        print("need >= 3 bands with engine results for the multi-band fit; skipped (17b/23/24 work without it)")
        return
    me = load_set(cfg, "player")
    me = me[me.mover_rating >= cfg["player"]["min_player_rating"]].reset_index(drop=True)
    names = list(categories(me).keys())
    cats_me = categories(me)
    rng = np.random.default_rng(0)
    B = mb["boot"]
    print("bands: " + ", ".join(f"{n} (mean rating {d.mover_rating.mean():.0f}, {len(d):,} moves)" for n, _, d in bands))
    print(f"you: {len(me):,} moves in {me.game_id.nunique()} games >= {cfg['player']['min_player_rating']}\n")
    out = []
    for metric, col in METRICS.items():
        # per band: per-game sums for all categories + bootstrap draws of the category means
        draws, point, ratings, weights = [], [], [], []
        for name, _, d in bands:
            cm = categories(d)
            M, S = per_game(d, [cm[k] for k in names], col)
            picks = rng.integers(0, len(M), (B, len(M)))
            draws.append(S[picks].sum(1) / np.maximum(M[picks].sum(1), 1))       # B x cats
            point.append(S.sum(0) / np.maximum(M.sum(0), 1))
            weights.append(M.sum(0))
            ratings.append(d.mover_rating.mean())
        draws, point, weights, ratings = np.array(draws), np.array(point), np.array(weights), np.array(ratings)
        Mm, Sm = per_game(me, [cats_me[k] for k in names], col)
        pm = rng.integers(0, len(Mm), (B, len(Mm)))
        me_draw = Sm[pm].sum(1) / np.maximum(Mm[pm].sum(1), 1)
        me_point = Sm.sum(0) / np.maximum(Mm.sum(0), 1)
        scale = 100 if col == "wp_loss" else 1
        print(f"=== {metric} ===")
        print(f"   {'category':40s} {'bands (low -> high)':>34s} {'slope/100':>10s} {'you':>7s}  your level (90% CI)")
        lvl_all = None
        for j, name in enumerate(names):
            if Mm[:, j].sum() < mb["min_moves"] or (weights[:, j] < mb["min_moves"]).any():
                continue
            a, s = fit_line(ratings, point[:, j], weights[:, j])
            sb = np.array([fit_line(ratings, draws[:, b, j], weights[:, j])[1] for b in range(B)])
            s_lo, s_hi = np.percentile(sb, [5, 95])
            skill = s_lo > 0 or s_hi < 0
            band_txt = " ".join(f"{scale * v:5.1f}" if scale == 100 else f"{v:5.1%}" for v in point[:, j])
            you_txt = f"{scale * me_point[j]:5.2f}" if scale == 100 else f"{me_point[j]:5.1%}"
            rec = {"metric": metric, "category": name, "slope_per_100": 1e4 * s, "slope_lo": 1e4 * s_lo,
                   "slope_hi": 1e4 * s_hi, "skill": skill, "you": me_point[j], "level": np.nan,
                   "level_lo": np.nan, "level_hi": np.nan, **{f"band_{i}": v for i, v in enumerate(point[:, j])},
                   **{f"rating_{i}": float(r) for i, r in enumerate(ratings)}}
            lv_txt = "flat across ratings: style, no level"
            if skill:
                lv = 2100 + (me_point[j] - a) / s
                lvb = []
                for b in range(B):
                    ab, sbb = fit_line(ratings, draws[:, b, j], weights[:, j])
                    if abs(sbb) > 1e-9:
                        lvb.append(2100 + (me_draw[b, j] - ab) / sbb)
                lo, hi = np.percentile(lvb, [5, 95])
                ext = " (extrapolated)" if lv < ratings.min() - 50 or lv > ratings.max() + 50 else ""
                rel = ""
                if name == "all moves":
                    lvl_all = lv
                elif lvl_all is not None:
                    rel = f"  ({lv - lvl_all:+.0f} vs your all-moves level)"
                lv_txt = f"Lichess ~{lv:.0f} [{lo:.0f}, {hi:.0f}]{ext}{rel}"
                rec.update({"level": lv, "level_lo": lo, "level_hi": hi})
            print(f"   {name:40s} {band_txt:>34s} {1e4 * s:+10.2f} {you_txt:>7s}  {lv_txt}")
            out.append(rec)
        print()
    pd.DataFrame(out).to_parquet(data_dir(cfg, "player") / "multiband.parquet", index=False)
    print("slope/100 = change per +100 rating, in percentage points (rates) or loss units (negative = stronger "
          "players err less; skill-relevant if its CI "
          "excludes 0).\nyour level = the band rating whose fitted value equals yours. Prefer contested rows "
          "(never resigning lowers your error rate in lost positions).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--stage", choices=["mine", "engine", "fit", "all"], default="all")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--target", type=int, default=None, help="positions per band (default multiband.target)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mb = settings(cfg)
    if args.target:
        mb["target"] = args.target
    if args.stage in ("mine", "all"):
        mine(cfg, mb)
    if args.stage in ("engine", "all"):
        for lo, _ in band_ranges(cfg, mb):
            if not (data_dir(cfg, band_name(lo)) / "positions_engine.parquet").exists():
                print(f"  band {lo}: no positions mined (none in the dump?), skipped")
                continue
            engine_pass(cfg, band_name(lo), workers=args.workers)
    if args.stage in ("fit", "all"):
        analyse(cfg, mb)


if __name__ == "__main__":
    mp.freeze_support()
    main()
