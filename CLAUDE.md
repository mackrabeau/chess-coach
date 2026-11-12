# chess-coach — context for Claude Code

Personal chess coaching pipeline for Mack (chess.com `bananaman987654321`, rapid ~1920, ~1,650 rapid games, climbed from 960). Finds what costs him points, how sure we are, and what to train. Two layers, kept separate:
1. **Analysis** (you vs reference; engine + stats; opening → middlegame → endgame).
2. **Exploitation** (exploratory; models of a player: traps, practical moves, later scouting / self-exploitability).

More detail when needed: `docs/status_log.md` (full chronological log, real-data numbers), `docs/future_work.md` (parked ideas), `docs/report_design.md` (report/drills proposal), `docs/generalize.md` (what's tailored to Mack; keep adding), script docstrings (each script documents itself).

## Run
- `python main.py [--list|--only a,b|--from x|--force x|--update|--skip-maia|--offline]`: all stages, make-style; resumable passes always invoked.
- `python refine.py [--hours H|--status]`: compute pouring; level k = engine.nodes × 4^k on priority positions (errors, threshold cases, critical; player + peers by the same rule) → opening tree deeper → more band data → all positions. Results in `data/<set>/engine_L<k>/`; `load_analyzed` uses the deepest (refined rows keep their own lines); `refined=False` when comparing with never-refined sets (17's bands).
- Tests: fake data via `tests/make_fake_data.py` + `tests/test.yaml`. Locally: `sed 's#C:/tools/stockfish/stockfish-windows-x86-64-avx2.exe#/usr/games/stockfish#' tests/test.yaml > tests/local.yaml`, venv `/tmp/claude-0/venv/bin/python`, delete local.yaml after. Fake games are engine-played: Maia fits hit the grid edge, you-vs-peers "within noise".

## Pipeline (scripts/, numbered in run order; diagnostics/ = one-off checks)
00 fetch chess.com games (whole history) · 01 mine Lichess peers 2000–2200 (dump streamed) · 01b peer results + status (Lichess API) · 01c Syzygy 3-4-5 · 02 Stockfish (1M nodes; `coach/runner.engine_pass`) · 03 settle lines (quiescence) · 05 feature deltas · 06 ideas (concept groups; NMF path kept for SAE) · 06c deepen no-swing tactics · 07/07b idea + tactics inspection · 08 player report · 09b Maia-3 P(best/played) over Elo grid 700–2500 + offset fit · 09c error anatomy · 09d Maia opponent model · 11 endgames (outcomes, tablebase) · 11b balanced endgames · 13 practical tree (3 plies, your follow-up modelled) · 14 openings (engine-only; book from dump per band) · 15 score accounting · 16 skill profile (Maia level + engine error ratio per category) · 17 multi-band (skill vs style) · 18 opening traps (layer 2).
Diagnostics: 02b calibration, 06b line ends, 09a Maia probe, 10 checks, 11c clock, 12 practical value (step 2, superseded by 13).
Shared code: `coach/runner.py`, `results.py` (load_analyzed), `categories.py`, `openings.py`, `maia.py` (+ maia_lists), `difficulty.py`, `stats.py`, `report.py` (HTML/SVG boards, hover end positions). `pos_id = "{source}:{game_id}:{ply}"` joins everything. Evals from the mover's POV; win prob = Lichess logistic; labels ok < 0.03 ≤ mistake ≤ 0.20 < blunder; critical = best beats 2nd by ≥ 0.10.

## Method rules (why things are the way they are)
- Peers: Lichess (bulk dump; Maia trained there); player: chess.com. "vs peers" only on games ≥ `player.min_player_rating` (1750); whole history for you-vs-yourself and Maia-relative measures (each game at its own level: Maia Elo = r + 160 − 0.10 (r − 1800) in Mack's local config).
- "clear" = peer value outside your 90% game-bootstrap CI (+ ratio ≥ 1.25 / ≤ 0.8 and expected count ≥ 5 in 08). Many slices tested → treat borderline lower bounds as one lead.
- **Never resigns**: lost/played-out positions rarely register errors → endgame / losing / low-clock error rates look too good. Prefer contested (0.2–0.8) rows; judge endgames by outcomes.
- **Maia level ≠ accuracy**: it measures how human-like-at-Elo his choices are. Only call something a weakness when the engine agrees; Maia-only differences = style.
- Peers have ~12 sampled positions per game (no per-move think time, no move-by-move execution); his data has every move.
- Per-game averages vs pooled per-move rates can disagree (balanced-endgame "1.5×" vanished pooled). State the weighting.
- Positional loss is not a skill metric (flat across ratings; rises in his history) → describes style.
- Exploitation section: looser evidence, labelled exploratory.

## Current picture (real data; details + numbers in docs/status_log.md)
- **Overall**: correctly rated, no big hole. Contested positions at par (error ratio 1.04); blunders par (1.89 vs 1.93 per 100). The peer band was calibrated to him, so par is expected.
- **Real leads**: (1) critical positions while pressing / forcing best move / middlegame: error ratio 1.16–1.19 (clear), at par with ≥ 3 min on the clock → time + concrete shots; (2) rook endings with time: Maia −188, engine 1.45 [0.79, 2.15] (small n); (3) long thinks in easy positions go wrong (34% vs 7%).
- **Strengths**: converts winning endgames (score 0.97 vs 0.88 by outcome); fewer one-move punishments (0.67×); Maia level high in critical/capture positions.
- **Openings** (engine-only): no overall weakness out of the opening; fixable lines: Vienna 3.Bc4 Nf6 4.f4? (4.d3), vs King's Gambit take (2...exf4), 2.Nc3 Nc6 3.Nf3 ...Nf6 not ...Bc5, vs Vienna Gambit 4...g5. Score accounting: moves ~10–15 are his only net-losing phase (−0.035/game) but his per-move accuracy there is par → opponents' book knowledge.
- **Time**: not a leak in results (9.8% of games end on time vs 9.5%); he reaches late phases with less clock than peers.
- **Exploitation**: Maia opponent model calibrated (Brier skill +44%); predicted trap value matches realized; 13's model of his own follow-ups validated (follow-up in Maia's list 90%; predicted 45.5 vs realized 46.2) but you ≈ peers → generic until an individual model exists.

## Open / next
- Mack to run on real data: 17 (multi-band; hours of Stockfish), 18 (traps), then main.py/refine.
- Report generator: waiting on Mack's answers to the 3 questions in `docs/report_design.md` (recommendation: one static HTML organized Analysis/Exploitation/Drills + Lichess study PGN files).
- Parked ideas: `docs/future_work.md` (time management model and clean-endgame plan execution are next candidates).

## Environment
- Mack: Windows 10, PowerShell, Python 3.12 `.venv`, repo `D:\chess\chess-coach\chess-coach`, Ryzen 5 3600 (engine.workers 10), GTX 1660 Super (Maia ~200 passes/s). His `configs/v0.yaml` has local values (paths, offset 160/−0.10); he pulls with `git pull --no-edit origin claude/inspiring-bohr-c5jekh`.
- Windows-safe multiprocessing (`__main__` guards, top-level workers, `mp.freeze_support()`), pathlib everywhere.
- Cloud session env `chess-coach` allows huggingface.co (Maia weights), tablebase.lichess.ovh, explorer.lichess.ovh (explorer now needs a token; unused). lichess.org API not reachable here.
- Branch: `claude/inspiring-bohr-c5jekh`. Commits end with the attribution lines; no PRs unless asked.

## Working with Mack
- Direct, casual tone; no corporate phrasing.
- Prefer targeted edits; give exact PowerShell commands, one per block.
- Point out specific flaws; don't overstate results; retract clearly when a finding doesn't hold.
- Math in LaTeX.
- He runs everything on Windows himself. No script-runner files unless he asks (main.py/refine.py were requested).
