"""Resumable parquet part files (used by the engine passes 02 and 03)."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def done_ids(part_dir: Path) -> set[str]:
    """pos_ids already written; also cleans up leftovers from a killed run."""
    done: set[str] = set()
    for t in part_dir.glob("part-*.parquet.tmp"):
        t.unlink()
    for p in sorted(part_dir.glob("part-*.parquet")):
        try:
            done.update(pq.read_table(p, columns=["pos_id"]).column("pos_id").to_pylist())
        except Exception:
            print(f"  removing unreadable part {p.name} (interrupted write)")
            p.unlink()
    return done


def next_part_index(part_dir: Path) -> int:
    idx = [int(p.stem.split("-")[1]) for p in part_dir.glob("part-*.parquet")]
    return max(idx, default=-1) + 1


def write_part(part_dir: Path, idx: int, rows: list[dict], schema: pa.Schema) -> None:
    tmp = part_dir / f"part-{idx:06d}.parquet.tmp"
    rows = [{k: r.get(k) for k in schema.names} for r in rows]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    tmp.replace(part_dir / f"part-{idx:06d}.parquet")  # atomic: no half-written parts


def read_parts(part_dir: Path):
    """All parts concatenated as a DataFrame (None if there are none), last write wins."""
    import pandas as pd

    parts = sorted(part_dir.glob("part-*.parquet"))
    if not parts:
        return None
    df = pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)
    return df.drop_duplicates("pos_id", keep="last")
