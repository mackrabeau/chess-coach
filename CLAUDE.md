# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# chess-coach — context for Claude Code

Chess coaching pipeline: finds what costs a player rating points, how sure we are, and what to train, from their
chess.com games compared with Lichess players (peers at their level and rating bands above/below), using Stockfish
(accuracy) and Maia-3 (human-likeness / difficulty). Two layers, kept separate:
1. **Analysis** (player vs reference; engine + stats; opening → middlegame → endgame).
2. **Exploitation** (exploratory; models of opponents: traps, practical moves, head-to-head prep).

## Vision (the plan every change should serve)
**Goal**: show a player the flaws in their play, compared with the average at their Elo, ranked by how much rating
fixing each one is worth. Fully automatic: inputs = a chess.com or Lichess username (+ optionally an opponent);
everything else (peer band, Maia offset, compute budget) is derived. (Lichess usernames: later, small job.) "Most valuable" = expected Elo gain ×
trainability (how often it comes up, whether a drill/line fixes it); state both, never rank on a p-value.
Headlines must be drillable (a tactic motif, an endgame type, an opening line), not "blunder less" / "critical
positions": broad slices are diagnostic context only.
Sub-goals, in priority order:
1. **Weakness patterns** (core). Supervised first: engineered, human-readable concepts (`coach/categories.py`)
   → error rates vs peers. Direction: one model of $P(\text{error} \mid \text{position features, rating, clock})$
   fit on peers across bands; the player's shrunken residual per concept = weakness, its rating slope = Elo
   value (replaces many separate slice tests, handles multiple comparisons; design in `docs/weakness_model.md`,
   next major build). Unsupervised (NMF now, SAE later)
   = exploratory pattern discovery, promoted to a named concept only once a human can describe it.
   Phases by game state (material, queens, development), not move number; split finer (opening book → opening
   → transition → middlegame → transition → endgame → theoretical endgame) only where the data shows a
   difference, else merge.
2. **Openings**: the player's own mistakes and the lines they meet most, ranked by frequency × cost at their
   band; recommend practical lines (human data at their level, Maia) checked by the engine; "deepen good lines"
   = prep where opponents deviate most. Must work at any level and repertoire.
3. **Lessons**: every finding → puzzles from the player's own games → lessons grouped by concept. Language
   layer later (future work, free open-weight model): an LLM narrates from structured findings + engine lines;
   it never judges chess itself.
4. **Exploitation** (exploratory): same pipeline on an opponent's games, recency-weighted; automatic prep
   (their weak openings/concepts × your repertoire).
5. **Research**: reuse what exists (Maia, Lichess data, published methods) before building; record sources and
   what's known in `docs/research.md`; new models are labelled exploratory and documented.
**Success check** (closed loop): re-runs over time must show whether trained weaknesses shrink; a finding that
never predicts anything out of sample gets dropped.

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
  `--only/--from/--force` take the stage names from `STAGES` in `main.py` (e.g. `report`, `opening-findings`,
  `engine-player`), not script numbers. A new script must be registered there with its input/output globs
  (relative to the player's data dir) or main.py never runs it. Every script also runs standalone:
  `python scripts/<nn>_<name>.py --config ...`.
- `--config configs/v0.yaml` (the default everywhere) also resolves `configs/local/v0.yaml` (`coach/config.py`).
- `python refine.py --config ...`: compute pouring (level k = engine.nodes × 4^k on priority positions → opening
  tree / repertoire deeper → more band data → all positions; `data/<set>/engine_L<k>/`, deepest wins in
  `load_analyzed`; `refined=False` for comparisons with never-refined sets). Runs `main.py --update` (new games +
  stale stages) at start, every 6 h and after each level.
- Head-to-head: `python scripts/23_compare_players.py --config A --other B` → `data/compare/<a>_vs_<b>/report.html`.
- Second player = small config with `extends: <first>.yaml` (see `configs/example_second_player.yaml`); quick
  head-to-head mode: `configs/example_quick.yaml`.
- Tests (no test suite; a fake-data end-to-end run is the test): `python tests/make_fake_data.py --config tests/test.yaml`
  then `python main.py --config tests/test.yaml --offline` (set `paths.stockfish` first). Fake data via
  `tests/make_fake_data.py` + `tests/test.yaml` (player `test_player`, opponent `someone`).
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
tactics findings + middlegame in-depth tables · 09b Maia-3 curves + offset fit · 09c error anatomy · 09d Maia opponent
model · 11 endgames (outcomes, tablebase) · 11b balanced endgames · 13 practical tree · 14 openings (engine-only;
book from the dump) · 14b opening findings · 15 score accounting · 16 skill profile · 17 multi-band (skill vs style;
reuses any peer set covering a band) · 17b category findings · 18 opening traps · 19 repertoire ("your lines,
extended", Lichess studies via `coach/study.py`) · 22 import PGN games · 24 weakness model (stage `weakness`:
peer error model across bands + your excess per concept, `coach/weakness.py` + `coach/motifs.py`; runs beside
08/16/17/17b until validated, then those slice findings get retired) · 20 report (aggregates findings: Summary +
Opening/Middlegame/Endgame by `rank` (Elo × trainability) else Elo, each with a collapsed "In depth" part,
Exploitation, Games (analysis board → PGN → `imported/inbox/`), Drills (copy/download PGN), How sure are we, Files) ·
23 head-to-head.
Shared code: `coach/` (runner, results, categories, openings, maia, difficulty, stats, findings, why, game_rating,
webreport, hub, report, study, lines, bands, motifs, weakness). `coach/lines.follow_up` turns a move into move + engine continuation (cached);
`findings.write` applies it to every one-move example / puzzle (best move = the plan, mistake = the punishment),
and puzzle chapters carry the wrong move's refutation as a side line. `pos_id = "{source}:{game_id}:{ply}"` joins everything. Evals from the mover's POV;
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
- `pyflakes coach scripts main.py refine.py` clean and a fake-data run before pushing (pyflakes may not be in the
  `.venv`: check early, ask before installing; `python -m py_compile` is the fallback).
- Read and write text files with an explicit `encoding='utf-8'`.
- When rewriting a code path (e.g. puzzle rendering), remove the old/unused path in the same edit; no dead code.

## Report UI (do this, don't re-derive it)
- No standalone detail pages: a module's section-specific tables/boards are findings with `detail=True`
  (`findings.detail()` + `findings.table()`; cells are plain text, the report escapes them), shown under "In depth"
  in their tab. `coach/report.py` only serves the developer pages (07/07b, hub's Developer tab).
- Where: `coach/webreport.py` holds `BASE_CSS` (tokens, type, tables; also used by `coach/report.py` developer
  pages), `CSS`, `JS` (boards, tabs, theme) and the components (`finding`, `table`, `kpis`, `example`) used by 20,
  23 and the hub. Display names: `PLAIN` in `coach/categories.py` + `plain_title()` (category ids stay the data
  keys). Before changing how a label renders, grep every call site (`Report.finding`, `Report.table`, `esc(c)` in
  20/23/hub) so all reports change together.
- Adopted rules (sources already read: web.dev typography, GOV.UK type scale/details, NN/g scanning; no need to
  research again): 16px / line-height 1.6 / prose ≤ 72ch; each finding = claim → what to do → numbers → boards;
  plain title with the raw category as a grey subline; tags = word + colour; numbers right-aligned; don't hide what
  most readers need in collapsed blocks; boards show side to move + green (best) / red (played) arrows.
- Chess UI rules (from Lichess source: lila `ui/lib/css/theme/board`, `ui/lib/css/tree/_tree.scss`, chessground
  `src/svg.ts`, lila `ui/puzzle/src/view/feedback.ts`; no need to research again):
  - Coordinates inside the board, bold, in the opposite square's colour; ranks top-right of the h-file, files
    bottom-left of rank 1; size 0.18–0.2 × square (≥ 8px). Flipped: reverse order, same edges, recompute colours.
  - Move lists: `grid-template-columns: 3ch 1fr 1fr`, dim move numbers, small right-aligned eval; glyph glued to
    the move ("Nf6?") and coloured: inaccuracy `hsl(202 78% 62%)`, mistake `hsl(41 100% 45%)`, blunder
    `hsl(0 69% 60%)`; key move = tinted background + bold.
  - Arrows: green `#15781B`, red `#882020`, stroke 10/64 square, head scaled to stroke, layer `opacity: .6`; draw red
    first, green on top; tip stops 10/64 short of centre (20/64 when both arrows end on the same square).
  - Puzzles: solver's side at the bottom, king icon in solver's colour + "Find the best move for White.";
    `<details>` "View the solution" → best move + follow-up, then "You played X?" + refutation.
  - Static boards ≥ 200px (`clamp(200px, 40vw, 320px)`), last move's from/to squares tinted, one-line caption
    (move no., side to move, best vs played with evals). This rule is inferred, not from source; tint colour unverified.
- Loop: regenerate (`python scripts/20_report.py --config configs/v0.yaml` is seconds; 23 for head-to-head;
  in-depth items need `main.py --force <their stages>`), then look at it, then test it. Check several tabs/pages,
  not just one, and every board both normal and flipped (labels and light/dark squares must stay correct):
  - Screenshot a tab: copy the HTML, inject `<script>show('<tab>')</script>` before `</body>`, then
    `chrome.exe --headless=new --disable-gpu --hide-scrollbars --user-data-dir=<scratch> --window-size=1280,2000
    --screenshot=<png> <file uri>`; `data-theme='dark'` on `<html>` for dark. Chrome is at
    `C:\Program Files\Google\Chrome\Application\chrome.exe` (Edge headless wrote nothing).
  - Headless Chrome's minimum width is 500px: a 400px screenshot is just cropped, not a layout bug. Check real
    overflow by comparing `document.documentElement.scrollWidth` to `innerWidth`.
  - JS smoke test: inject a `window.onerror` collector in `<head>` and a test script that clicks/keys and writes
    results to a `data-` attribute on `<body>`; run with `--virtual-time-budget=3000 --dump-dom` and regex it out.
  - Set `PYTHONIOENCODING=utf-8` for any Python that prints report text (→, ✓, …) from the Bash tool.
  - Verify UI changes by opening the real generated report, not headless Chrome screenshots (they have width limits).
  - Test the board in both orientations (normal and flipped) and check coordinate label colours on each.

## Mistakes log (running list; `/reflect` appends here; once a lesson is a rule above, delete it here)
- Personal notes were double-encoded (UTF-8 read as cp437, saved again) by a PowerShell write → mojibake for
  every session. Write files with the Write tool or Python `encoding="utf-8"`, never `Set-Content` defaults.
- Copied into the scratchpad before it existed: create it first (`mkdir -p`).
- `Path(rel).as_uri()` fails on relative paths: `.resolve()` first.
- Printed Unicode from Python in the Bash tool without `PYTHONIOENCODING=utf-8` → cp1252 crash, rerun (twice).
- Tried Edge headless first (silent no-output); go straight to Chrome.
- Faked tab switching by clicking the `href='#tab'` link → browser anchor jump hid the heading under the sticky
  header; the real page had the same bug. In-page `#` links must `preventDefault` + `history.pushState`.
- Shipped JS with leftover no-op junk and a flipped-board coordinate colour bug; re-read generated JS against the
  board geometry (a1 dark, orientation) before the first render.
- Reused a generic class name (`.bar`) already styled in the header: namespace new component classes.
- Fixed labels in the main report but missed the comparison tables (raw `phase: endgame`) until a screenshot:
  grep all call sites first.
- Low-yield research (generic "lichess ui" / open-source review app searches, a guessed lila file URL → 404):
  go straight to primary guidance (web.dev, GOV.UK, NN/g) and the current rendered page.

### 2026-11-12
- Installed pyflakes into `.venv` without asking, against the rule above: the plan came from a stale view of this
  file. Re-read CLAUDE.md before acting on tooling/install rules.
- Wrote a "new" `coach/motifs.py` over the existing one (06c/07b/08 import `tactic_labels` from it), and
  re-invented its cash-in rule. Before creating a module: `ls coach/`, grep for the concept (e.g. "motif",
  "fork"), and reuse what's there.

