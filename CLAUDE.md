# chess-coach — context for Claude Code

## Goal
Coaching tool that tells a player which chess *ideas* they miss, in concrete terms ("the best line kept your bishop pair and opened the f-file; yours didn't"), not generic advice ("work on your middlegame").

Core idea: explain a move's quality in feature terms along the engine's line, not a single eval number. For a position x with engine best move a* and played move a_p:

    d(x) = f(end of a*'s PV) - f(end of a_p's PV)

where f is a feature vector and PVs have an even number of plies (same side to move at both ends). An "idea" is a recurring direction of d. Ideas are learned from peer mistakes, then the player's games are measured against peers per idea.

Owner: Mack (chess.com `bananaman987654321`, rapid ~1922, 1,652 rapid games; climbed from 960, so only games at rating >= 1750 are used).

## Design decisions (and why)
- **Peers come from Lichess, player games from chess.com.** chess.com has no bulk dump; Maia-3 is trained on Lichess. Peer band = Lichess rapid 2000–2200 (center 2100), a starting guess for chess.com 1922. Calibrate with `02b_engine_summary.py`: match mean win-prob loss and blunder rate, shift `population.rating_center` if off.
- **Ideas are learned on the population, not the player.** One player has too few mistakes to discover clusters; the population gives stable atoms plus a per-idea peer baseline.
- **NMF on unit-normalized, sign-split deltas** (`[max(u,0), max(-u,0)]`), not k-means: positions can contain several ideas; NMF atoms read as "best line has more X / less Y". Direction = idea, norm = severity (stored separately).
- **Split-half stability**: fit on two disjoint halves (split by game), Hungarian-match atoms by cosine; an atom is stable if cos >= 0.7 to both. Unstable atoms are shown but flagged.
- **Win-probability loss** (Lichess logistic, `1/(1+exp(-0.00368208*cp))`), not raw cp. Labels: ok < 0.03 ≤ mistake ≤ 0.20 < blunder. Blunders are excluded from idea learning (they'd swamp every cluster); separate track later.
- **Critical positions** (best beats 2nd best by ≥ 0.10 win prob) give a denominator: "in positions where idea k decides the best move, you miss it X% vs peers Y%". The comparison move there is Stockfish's 2nd-best (from MultiPV=2), not Maia's pick (deliberate simplification).
- **Player metrics**: leak_k = 100/N · Σ_mistakes share_k · wp_loss (win prob lost to idea k per 100 moves), with a bootstrap 90% CI over games; critical miss rate shrunk toward peers with a Beta prior. Report says "clear" only when the peer value is outside the CI.
- **Features v0 = 29 handcrafted features × {us, them}** (`coach/handcrafted.py`). Planned replacement: sparse autoencoder on Maia-3 residual stream (see Next steps).

## Pipeline (run from repo root, venv active; all take `--config`, default `configs/v0.yaml`)
| script | does | output |
|---|---|---|
| `00_fetch_player.py [--offline]` | chess.com public API → player's moves (rapid, rated, rating ≥ 1750, ply ≥ 20, clock ≥ 30 s) | `data/player/positions.parquet`, `data/player/archives/*.json` |
| `01_mine_population.py` | streams Lichess `.pgn.zst` (never unpacked), header filters first; stops at target | `data/population/positions_sae.parquet` (1.5M), `positions_engine.parquet` (~20k) |
| `02_engine_lines.py --set {player,population} [--limit N]` | Stockfish, multiprocess, **resumable** (atomic part files, skips done pos_ids) | `data/<set>/engine/part-*.parquet` |
| `02b_engine_summary.py` | player vs peer loss profile (rating calibration) | stdout |
| `05_build_deltas.py --set ...` | features at root + PV ends → F0, dB, dP, dA (NaN if line < 2 plies) | `data/<set>/deltas.npz` |
| `06_learn_ideas.py` | scale by pooled Δ std, NMF sweep + stability | `data/ideas/model.pkl` (IdeaModel), `sweep.csv`, `pop_codes.npz` |
| `07_inspect_ideas.py` | HTML per idea with top peer positions | `data/ideas/ideas_report.html` |
| `08_player_report.py` | player leak / miss rates vs peers, own positions + drills | `data/player/player_report.html` |

Key modules: `coach/mine.py` (position extraction, parquet schema incl. 7-ply `history_fens` for Maia), `coach/engine.py` (Analyzer, `RESULT_SCHEMA`), `coach/ideas.py` (IdeaModel, NMF, stability), `coach/results.py` (`load_analyzed`, `load_with_deltas`), `coach/report.py` (HTML/SVG boards). `pos_id = "{source}:{game_id}:{ply}"` joins everything.

Engine output per position: best_move, cp_best, cp_played, alt_move/cp_alt (2nd best), wp_*, wp_loss, label, critical, played_is_best, best_line/played_line/alt_line (UCI, length `pv_plies`=6, trimmed to even if game ends), *_complete flags. Evals from the mover's POV; each position starts with `ucinewgame` (hash cleared, order-independent).

## Status (2026-11-10)
- All scripts above written and tested end to end on fake data (`tests/make_fake_data.py`, `tests/test.yaml`) on Linux. Delta alignment verified against direct recomputation.
- **On Mack's machine:** `00_fetch_player.py` ran (after fixing chess.com 403 via a descriptive User-Agent). `01_mine_population.py` done: scanned 25M games, 1,500,001 SAE positions, 20,120 engine positions.
- **Blocker last seen:** `02_engine_lines.py` failed with WinError 2 because `paths.stockfish` in `configs/v0.yaml` still points at the placeholder `C:/tools/stockfish/stockfish-windows-x86-64-avx2.exe`. Needs the real path (forward slashes). Script now exits with a clear message if the file is missing.
- Not yet run on real data: 02 (both sets), 02b, 05–08.

## Environment
- Windows 10, PowerShell, Python 3.12 in `.venv` (`.venv\Scripts\Activate.ps1`). Repo at `D:\chess\chess-coach\chess-coach`. Run scripts with `python scripts/...` from the activated terminal (VS Code's Run button used the global Python once; select the `.venv` interpreter).
- Ryzen 5 3600 (6c/12t): `engine.workers: 10`, 1M nodes/search, ~4 searches/position → est. 2–3 positions/s, 20k peers ≈ 2–3 h. Drop to 300k nodes if too slow.
- GTX 1660 Super, 6 GB, no tensor cores (for the SAE: Maia inference fp16, SAE training fp32). Plenty of disk.
- Windows gotchas already handled: spawn-safe multiprocessing (`__main__` guards, top-level workers, `mp.freeze_support()`), `pathlib` everywhere, `zstandard` package for `.zst`.

## Next steps
1. Fix Stockfish path → `02 --limit 200` test → `02 --set player` → `02 --set population` (overnight) → `02b` (calibrate band; if shifting, delete `data/population/` and re-mine) → `05` both sets → `06` → `07` → `08`.
2. **Go/no-go**: in `ideas_report.html`, do ≥ ~1/3 of stable ideas show a recognizable shared idea in most of their top positions? Check `sweep.csv` for how stable-atom count scales with K.
3. Known weakness: PVs can end mid-exchange, so material-count features can reflect pending recaptures. Fix options: extend lines to a quiet position (no capture/check available), or compare after a short quiescence.
4. **Maia-3 SAE features** (plan §4b). Facts checked 2026-11-09:
   - Weights public on Hugging Face (`UofTCSSLab/Maia3-5M/23M/79M`); code `pip install git+https://github.com/CSSLab/maia3` (inference only, AGPL-3.0, commit `1e13597`).
   - **Transcoder weights NOT released** (model card mentions `transcoder.py`, but it's not in the repo). Train our own topK SAE. Reference: `CSSLab/maia2-skill-adaptation` has topK transcoder code for Maia-2.
   - Model: 64 square tokens, board mirrored to side to move, input = 8 positions of history + self/opp Elo embeddings, post-LN blocks (`x = norm2(x + FFN(norm1(x + attn(x))))`), 8 blocks; 23M has d=512. Value head = mean-pool → LN → MLP → WDL (predicts *human game outcome*, not engine eval).
   - Plan: Maia-3-23M, fixed Elo (e.g. 1900/1900) for extraction, hook block 4 or 6 output, topK SAE 8192 latents k=32, activations to on-disk memmap (fp16, ~20 GB/300k positions/layer). Must pass: FVU < ~0.2, dead < ~10%, splice test (replace residual with reconstruction, check policy KL / top-1 drop). Position feature = sum-pool latents over squares. Then rerun 05–08 with SAE features, compare to handcrafted.
5. Later: valuation-vs-calculation split (depth at which the idea's features first appear), blunder track, LLM verbalization grounded on atoms + positions, prospective validation (do idea miss rates predict next month's errors better than a rating-only baseline).

Full design doc: `chess-coaching-engine-plan.md` (claude.ai "chess" project).

## Working with Mack
- Direct, casual tone; no corporate phrasing.
- Prefer targeted edits over full rewrites; show exact lines to change and where (file + line), not "..." placeholders.
- Point out specific flaws; don't overstate results or skip steps (he'll push back).
- Math in actual LaTeX.
- He runs everything on Windows himself; give exact PowerShell commands, one at a time.
