# chess-coach

Find the *ideas* you miss: patterns in how the engine's preferred move changes the position vs. the move you played, learned from peer games, then measured on your own games. Full design: `chess-coaching-engine-plan.md` in the claude.ai "chess" project.

Done so far: data, the Stockfish pass, handcrafted features, idea learning, and both reports (plan §2–8 with handcrafted features). Next: the Maia-3 SAE as a replacement feature extractor.

## Setup (Windows 10)

1. **Python 3.11 or 3.12** from python.org (tick "Add to PATH"). In a terminal in this folder:
   ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Stockfish**: download the Windows build from https://stockfishchess.org/download/, pick the **x86-64-avx2** version, unzip to `C:\tools\stockfish\`. Make sure `paths.stockfish` in `configs/v0.yaml` points at the `.exe` (forward slashes are fine).
3. **Lichess dump**: from https://database.lichess.org/ download one recent "Standard" month (`lichess_db_standard_rated_YYYY-MM.pgn.zst`, ~30 GB). Put it in `data/raw/` and set `paths.lichess_dump`. Don't unzip it; it's streamed.

## Run order

All commands from this folder with the venv active.

```
# 1. Your chess.com rapid games -> data/player/positions.parquet   (~1 min)
python scripts/00_fetch_player.py

# 2. Peer positions from the Lichess dump -> data/population/      (~10–30 min)
python scripts/01_mine_population.py

# 3. Quick engine check before the long run (should take ~1 min)
python scripts/02_engine_lines.py --set population --limit 100

# 4. The long runs (overnight). Safe to Ctrl+C and restart; finished work is kept.
python scripts/02_engine_lines.py --set player
python scripts/02_engine_lines.py --set population

# 5. Compare your loss profile to the peer band (rating calibration)
python scripts/02b_engine_summary.py

# 6. Feature deltas along the engine lines (a few minutes each)
python scripts/05_build_deltas.py --set population
python scripts/05_build_deltas.py --set player

# 7. Learn ideas from peer mistakes + split-half stability (minutes)
python scripts/06_learn_ideas.py

# 8. Look at the ideas: data/ideas/ideas_report.html   <- the go/no-go check
python scripts/07_inspect_ideas.py

# 9. Your report: data/player/player_report.html
python scripts/08_player_report.py
```

### Reading the reports

- **ideas_report.html**: one section per idea, stable ones first. ↑/↓ chips say which features the end of the engine's line has more/less of than the end of the played line. Ignore the labels and look at the boards: an idea is real if most of its top positions share something you can name. "Unstable" ideas didn't reappear when fitting on two halves of the games separately.
- **player_report.html**: ideas where you lose more win probability per 100 moves than peers, with a 90% interval from resampling your games. Only rows marked **clear** are real differences; "within noise" ones are leads. Each idea shows your own positions (with links to the games) and peer positions as puzzles.
- `data/ideas/sweep.csv`: how many atoms survive the stability check at each dictionary size. If 50 atoms gives barely more stable ones than 30, stay at 30.

Known limitation (handcrafted features): a line can end mid-exchange, so material features sometimes reflect a capture that's about to be recaptured. Ideas dominated by "↑ your pawns / knights / …" counts are often this, not a real idea.

Step 4 timing on a Ryzen 5 3600 with 10 workers at 1M nodes is roughly 2–3 positions/s, so 20k population positions ≈ 2–3 h. The script prints a live ETA. If it's much slower, set `engine.nodes: 300000`.

Do the player run first: if your positions look fine in step 5, the population run is the one to leave overnight.

## What each step produces

| file | contents |
|---|---|
| `data/player/archives/*.json` | raw chess.com monthly archives (cached) |
| `data/player/positions.parquet` | your moves only, rapid, rated, rating ≥ `min_player_rating`, move ≥ 10, clock ≥ 30 s |
| `data/population/positions_sae.parquet` | ~1.5M peer positions (SAE training pool, later) |
| `data/population/positions_engine.parquet` | ~20k random subset for Stockfish |
| `data/*/engine/part-*.parquet` | per position: best move, 2nd best, evals, win-prob loss, label (ok/mistake/blunder), critical flag, 6-ply lines for best / played / 2nd-best move |

Evals and win probabilities are from the mover's point of view. Lines are always an even number of plies so the end position has the same side to move as the root.

## Calibrating the peer band

`population.rating_center: 2100` is a guess for your chess.com 1922. After step 5:
- if your mean loss is clearly **lower** than the population's, you're playing above the band: raise `rating_center` by ~100 and re-run steps 2 and 4 (population only)
- if clearly **higher**, lower it.

Delete `data/population/` before re-mining with a new band, otherwise the engine step will mix old and new results.

## Smoke test (no downloads)

Generates fake games with a weak Stockfish and runs the pipeline on them. Edit `tests/test.yaml` so `paths.stockfish` points at your Stockfish first.

```
python tests/make_fake_data.py --config tests/test.yaml
python scripts/01_mine_population.py --config tests/test.yaml
python scripts/00_fetch_player.py --config tests/test.yaml --offline
python scripts/02_engine_lines.py --config tests/test.yaml --set population
python scripts/02b_engine_summary.py --config tests/test.yaml
python scripts/05_build_deltas.py --config tests/test.yaml --set population
python scripts/02_engine_lines.py --config tests/test.yaml --set player
python scripts/05_build_deltas.py --config tests/test.yaml --set player
python scripts/06_learn_ideas.py --config tests/test.yaml
python scripts/07_inspect_ideas.py --config tests/test.yaml
python scripts/08_player_report.py --config tests/test.yaml
```
On fake games every player difference should come out "within noise".
