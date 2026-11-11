"""Download the Syzygy 3-4-5 piece endgame tablebases (WDL + DTZ, ~1 GB) from the Lichess mirror.

Resumable: files already present with the right size are skipped.

    python scripts/01c_download_syzygy.py --config configs/v0.yaml
    python scripts/01c_download_syzygy.py --config configs/v0.yaml --limit 4   # quick test

Target folder: endgames.syzygy_path in the config.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from coach.config import load_config, resolve

BASE = "https://tablebase.lichess.ovh/tables/standard/"
DIRS = ["3-4-5-wdl/", "3-4-5-dtz/"]
UA = {"User-Agent": "chess-coach research script (tablebase download)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--limit", type=int, default=None, help="only the first N files per folder (testing)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve(cfg, cfg["endgames"]["syzygy_path"])
    out.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        for d in DIRS:
            listing = s.get(BASE + d, headers=UA, timeout=60).text
            files = sorted(set(re.findall(r'href="([^"/]+\.rtb[wz])"', listing)))[: args.limit]
            print(f"{d}: {len(files)} files")
            for i, name in enumerate(files, 1):
                dest = out / name
                head = s.head(BASE + d + name, headers=UA, timeout=60)
                size = int(head.headers.get("Content-Length", -1))
                if dest.exists() and dest.stat().st_size == size:
                    continue
                with s.get(BASE + d + name, headers=UA, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    tmp.replace(dest)
                if i % 20 == 0 or i == len(files):
                    print(f"  {i}/{len(files)}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
