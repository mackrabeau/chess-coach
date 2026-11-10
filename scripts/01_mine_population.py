"""Stream a Lichess dump and mine peer positions in the configured rating band.

Outputs:
  data/population/positions_sae.parquet     every kept position (SAE training pool)
  data/population/positions_engine.parquet  random subset for the Stockfish pass

Stops as soon as sae_target_positions is reached, so you rarely read the whole dump.

    python scripts/01_mine_population.py --config configs/v0.yaml
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach.config import data_dir, load_config, resolve
from coach.mine import ParquetSink, _int, extract_positions, is_rapid
from coach.pgn_stream import iter_raw_games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--dump", default=None, help="override paths.lichess_dump")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pop, fcfg = cfg["population"], cfg["filters"]
    dump = resolve(cfg, args.dump or cfg["paths"]["lichess_dump"])
    lo = pop["rating_center"] - pop["rating_halfwidth"]
    hi = pop["rating_center"] + pop["rating_halfwidth"]
    target = pop["sae_target_positions"]
    p_engine = min(1.0, pop["engine_target_positions"] / target)
    rng = random.Random(pop["seed"])

    out = data_dir(cfg, "population")
    sae = ParquetSink(out / "positions_sae.parquet")
    eng = ParquetSink(out / "positions_engine.parquet", flush_rows=5_000)

    n_seen = n_kept_games = 0
    t0 = time.time()
    for headers, text in iter_raw_games(dump):
        n_seen += 1
        if n_seen % 500_000 == 0:
            rate = n_seen / (time.time() - t0)
            print(f"  scanned {n_seen:,} games ({rate:,.0f}/s), kept {n_kept_games:,} games, "
                  f"{sae.n + len(sae.buf):,} positions", flush=True)
        # Cheap header filters first; almost every game dies here.
        if "Rated" not in headers.get("Event", ""):
            continue
        if not is_rapid(headers.get("TimeControl"), fcfg):
            continue
        w, b = _int(headers.get("WhiteElo")), _int(headers.get("BlackElo"))
        if not (lo <= w <= hi and lo <= b <= hi):
            continue
        if headers.get("Termination", "Normal") not in ("Normal", "Time forfeit"):
            continue  # abandoned / rules infraction

        game_id = headers.get("Site", "").rstrip("/").split("/")[-1] or f"g{n_seen}"
        rows = extract_positions(
            text, headers, source="lichess", game_id=game_id, fcfg=fcfg,
            mover_ok=lambda _w: True, rng=rng, max_per_game=pop["max_positions_per_game"],
        )
        if not rows:
            continue
        n_kept_games += 1
        sae.add(rows)
        eng.add([r for r in rows if rng.random() < p_engine])
        if sae.n + len(sae.buf) >= target:
            break

    sae.close()
    eng.close()
    print(f"done: scanned {n_seen:,} games, kept {n_kept_games:,}; "
          f"{sae.n:,} SAE positions, {eng.n:,} engine positions -> {out}")


if __name__ == "__main__":
    main()
