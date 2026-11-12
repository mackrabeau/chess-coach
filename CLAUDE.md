# chess-coach — context for Claude Code

Chess coaching pipeline: finds what costs a player rating points, how sure we are, and what to train, from their
chess.com games compared with Lichess players (peers at their level and rating bands above/below), using Stockfish
(accuracy) and Maia-3 (human-likeness / difficulty). Two layers, kept separate:
1. **Analysis** (player vs reference; engine + stats; opening → middlegame → endgame).
2. **Exploitation** (exploratory; models of opponents: traps, practical moves, head-to-head prep).

**Personal context lives outside git**: `CLAUDE.local.md` (the player's results, preferences, environment;
Claude Code loads it next to this file when present), `docs/local/status_log.md` (chronological log),
`configs/local/*.yaml` (configs with usernames/paths). Never commit personal data, usernames, results or
paths; keep shared docs generic. Other docs: `docs/agents.md` (modules publish findings via
`coach/findings.py`; the report only aggregates), `docs/future_work.md`, `docs/report_design.md`,
`docs/generalize.md`; each script documents itself in its docstring.

## Run
- `python main.py --config configs/local/<you>.yaml [--list|--only a,b|--from x|--force x|--update|--skip-maia|--offline]`:
  all stages, make-style (outputs newer than inputs are skipped; resumable passes always invoked). Writes
  `data/logs/main_<time>.log` and rebuilds the hub `data/index.html` (every player's report, comparisons, logs).
- `python refine.py --config ...`: compute pouring (level k = engine.nodes × 4^k on priority positions → opening
  tree / repertoire deeper → more band data → all positions; `data/<set>/engine_L<k>/`, deepest wins in
  `load_analyzed`; `refined=False` for comparisons with never-refined sets). Runs `main.py --update` (new games +
  stale stages) at start, every 6 h and after each level.
- Head-to-head: `python scripts/23_compare_players.py --config A --other B` → `data/compare/<a>_vs_<b>/report.html`.
- Second player = small config with `extends: <first>.yaml` (see `configs/example_second_player.yaml`); quick
  head-to-head mode: `configs/example_quick.yaml`.
- Tests: fake data via `tests/make_fake_data.py` + `tests/test.yaml` (player `test_player`, opponent `someone`).
  Locally: `sed 's#C:/tools/stockfish/stockfish-windows-x86-64-avx2.exe#/usr/games/stockfish#' tests/test.yaml > tests/local.yaml`
  (ignored by git). Fake games are engine-played: Maia fits hit the grid edge, you-vs-peers "within noise".

## Data layout (coach/config.py)
`data/players/<username>/{player,openings,findings,imported}` per player · `data/population_<lo>_<hi>/` peers per
Lichess band (+ `ideas/`), shared · `data/band_<lo>/` multi-band samples · `data/compare/` · `data/index.html`.
The chess.com → Maia offset is fitted per player by 09b (`player/maia_offset.json`) and overrides the config.

## Pipeline (scripts/, numbered in run order; diagnostics/ = one-off checks)
00 fetch chess.com games · 01 mine Lichess peers for the band (dump streamed) · 01b peer results + status · 01c
Syzygy 3-4-5 · 02 Stockfish (`coach/runner.engine_pass`) · 03 settle lines (quiescence) · 05 feature deltas · 06
ideas (concept groups; NMF path kept for SAE) · 06c deepen no-swing tactics · 07/07b idea + tactics inspection · 08
detailed player report (+ tactics findings) · 09b Maia-3 curves + offset fit · 09c error anatomy · 09d Maia opponent
model · 11 endgames (outcomes, tablebase) · 11b balanced endgames · 13 practical tree · 14 openings (engine-only;
book from the dump) · 14b opening findings · 15 score accounting · 16 skill profile · 17 multi-band (skill vs style;
reuses any peer set covering a band) · 17b category findings · 18 opening traps · 19 repertoire ("your lines,
extended", Lichess studies via `coach/study.py`) · 22 import PGN games · 20 report (aggregates findings: Summary +
Opening/Middlegame/Endgame by expected Elo gain, Exploitation, Games, Drills, Method, Files) · 23 head-to-head.
Shared code: `coach/` (runner, results, categories, openings, maia, difficulty, stats, findings, why, game_rating,
webreport, hub, report, study). `pos_id = "{source}:{game_id}:{ply}"` joins everything. Evals from the mover's POV;
win prob = Lichess logistic; ok < 0.03 ≤ mistake ≤ 0.20 < blunder; critical = best beats 2nd by ≥ 0.10.

## Method rules (why things are the way they are)
- Peers: Lichess (bulk dump; Maia trained there); player: chess.com. "vs peers" only on games ≥
  `player.min_player_rating`; whole history for you-vs-yourself and Maia-relative measures (each game at its own
  level via the fitted offset).
- "clear" = peer value outside the player's 90% game-bootstrap CI (+ ratio ≥ 1.25 / ≤ 0.8, expected count ≥ 5 in
  08). Many slices tested → treat borderline lower bounds as one lead.
- Players who rarely resign: lost positions get played out and rarely register errors → endgame / losing /
  low-clock error rates look too good (17b/20 detect it from game endings). Prefer contested (0.2–0.8) rows;
  judge endgames by outcomes.
- Maia level ≠ accuracy (human-likeness at an Elo). Call a weakness only when the engine agrees; Maia-only = style.
- Peers have ~12 sampled positions per game; the player has every move.
- Per-game averages vs pooled per-move rates can disagree: state the weighting.
- Positional loss is not a skill metric (flat across ratings) → style.
- Elo gain = 695 × expected score per game (win-prob loss as proxy; estimates overlap; upper bound).
- Exploitation: looser evidence, labelled exploratory; individual opponents judged on RECENT play (a line counts
  only if their latest visit still went wrong).

## Conventions
- Windows-safe multiprocessing (`__main__` guards, top-level workers, `mp.freeze_support()`), pathlib everywhere.
- Prefer targeted edits; exact commands (PowerShell for the Windows user) one per block; point out flaws, don't
  overstate results, retract clearly; math in LaTeX.
- `pyflakes coach scripts main.py refine.py` clean and a fake-data run before pushing.
