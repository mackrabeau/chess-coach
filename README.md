# chess-coach

A personal chess analysis pipeline: which positions and ideas cost you points, how sure we are, and
what to train. It compares your chess.com games with Lichess players at your level (and above and
below it), using Stockfish for accuracy and Maia-3 for how human-like / how difficult each move is.
Two layers:

1. **Analysis**: you vs a reference, opening → middlegame → endgame, with confidence intervals.
2. **Exploitation** (exploratory): where opponents at your level go wrong, opening traps, practical
   move choice given what you and they are likely to play.

Design notes: `docs/report_design.md` (report + drills, proposal), `docs/generalize.md` (what is
tailored to one player). Detailed method/status log: `CLAUDE.md`.

## Setup

1. **Python 3.11 / 3.12**, then in this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate            (Linux/macOS: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
   For the Maia stages (GPU recommended): `pip install torch` (CUDA build from pytorch.org) and
   `pip install git+https://github.com/CSSLab/maia3`. Without them, run with `--skip-maia`.
2. **Stockfish** from https://stockfishchess.org/download/.
3. **A Lichess monthly dump** (`lichess_db_standard_rated_YYYY-MM.pgn.zst`, ~30 GB) from
   https://database.lichess.org/. Don't unzip it; it's streamed.
4. **Config**: copy `configs/example.yaml` to `configs/local/<you>.yaml` (ignored by git, so your username and
   paths stay on your machine) and fill in the fields marked `<-` (username, Stockfish path, dump path, rating band).

## Run everything

```
python main.py --config configs/local/<you>.yaml
```

Runs every stage that is missing or out of date, in order (fetch your games → mine peers → Stockfish
→ settle lines → features → Maia → analyses → exploitation). Long passes are resumable: stop it any
time (Ctrl+C, power cut) and rerun the same command. `python main.py --list` shows each stage and
whether it would run; `--only a,b`, `--from stage`, `--force stage`, `--update` (refetch your newest
games), `--skip-maia`, `--offline`.

First run on a 6-core PC: roughly a day of engine time (most of it the peer and band Stockfish passes
at 1M nodes), plus ~1 h of GPU for Maia.

## More compute = more accuracy

```
python refine.py --config configs/local/<you>.yaml            # until the planned levels are done
python refine.py --config configs/local/<you>.yaml --hours 8  # time budget
python refine.py --config configs/local/<you>.yaml --status
```

Re-searches positions at 4x, 16x, 64x the nodes (priority first: mistakes, threshold cases,
critical positions; you and the peers by the same rule), deepens the opening tree, and mines more
multi-band data. Everything is saved as it goes. Then `python main.py` rebuilds what changed.

## Outputs

| file | what |
|---|---|
| `data/player/player_report.html` | main report so far (tracks, ideas, tactics, Maia difficulty, progress) |
| `data/player/openings_report.html` | opening lines that cost you (engine-only), must-know positions |
| `data/player/endgame_report.html`, `balanced_endgames.html` | endgames by outcome, tablebase, balanced endgames |
| stdout of 15 / 16 / 17 | score accounting, skill profile (Maia + engine), multi-band skill vs style |
| `data/player/opening_traps.html`, `practical_tree_report.html` | exploitation (exploratory) |
| `data/ideas/*.html` | idea and tactics-detector inspection |

`scripts/diagnostics/` holds one-off checks (rating calibration, Maia probe, line-end check,
clock confound, step-2 practical value); they are not part of the main run.

## Calibrating the peer band

`population.rating_center` should be the Lichess equivalent of your chess.com rating. After the
engine pass, `python scripts/diagnostics/02b_engine_summary.py` compares your loss profile with the
band; if yours is clearly lower, raise the center by ~100 (and the reverse). Delete
`data/population/` before re-mining with a new band.

## Smoke test (no downloads)

Fake games played by a weak Stockfish; set `paths.stockfish` in `tests/test.yaml` first.

```
python tests/make_fake_data.py --config tests/test.yaml
python main.py --config tests/test.yaml --offline
```

On fake games every you-vs-peers difference should come out "within noise", and Maia fits run to
the edge of the Elo grid (engine-played games aren't human).

## License

The code in this repository is MIT-licensed (see `LICENSE`). It does not include or
redistribute any third-party engine or model; those keep their own licenses:

- **Stockfish** (GPL-3.0) is a separate program you download yourself. The pipeline only
  talks to it over the UCI protocol as a subprocess (via python-chess).
- **Maia-3** (code and weights: AGPL-3.0, per the maia3 repository) is installed with pip
  and imported by `coach/maia.py`; the weights are downloaded from Hugging Face on first
  use. If you distribute this project together with Maia-3, or run it as a network
  service, the AGPL's terms apply to that combined work.

## Citations

If you use or build on this project, please also cite the tools it depends on.

**Maia-3** (human move prediction model used for the difficulty measures):

```bibtex
@inproceedings{monroe2026chessformer,
  title     = {Chessformer: A Unified Architecture for Chess Modeling},
  author    = {Daniel Monroe and George Eilender and Philip Chalmers and Zhenwei Tang and Ashton Anderson},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=2ltBRzEHyd}
}
```

Code: https://github.com/CSSLab/maia3 · Weights: https://huggingface.co/UofTCSSLab

**Stockfish** (engine lines and evaluations), per its `CITATION.cff`:

```bibtex
@software{stockfish,
  author  = {{The Stockfish developers}},
  title   = {Stockfish},
  url     = {https://stockfishchess.org/},
  note    = {Source: https://github.com/official-stockfish/Stockfish. GPL-3.0. Authors listed in the AUTHORS file.}
}
```
