# Future work and parked ideas

Ideas worth exploring that aren't built yet, roughly grouped. Each: what, why, how, and the known risk.
"Status" says how far it got. Keep adding; move an item out when it's built (and note it in CLAUDE.md).

## Layer 1: analysis

### Concept groups / clustering, all ML in the system (2026-11-12) — status: next big topic after the report rework
- What: the middlegame and endgame groups (08's NMF "ideas", 17's slices, 24's motif × phase concepts) are
  ambiguous to a player. Revisit the whole concept layer as one piece: what a group is, how it's found
  (engineered features vs learned clusters vs SAE), and how it's named.
- Why: findings are only coachable if a player recognises the group ("you allow forks in endgames" yes;
  "idea 7" or "critical while pressing" no). 24's first real run found no stable concept-level difference,
  which may mean the concepts are too coarse/wrong, not that there's nothing to find.
- How: start from `docs/research.md` (to write: chess concept detection, Lichess puzzle themes, SAE on
  chess models), then test candidate groupings with 24's machinery (peer-calibrated O/E + temporal holdout):
  a grouping is good if it's nameable AND its player residuals replicate across time.
- Risk: more concepts = more multiple-comparison noise; 24's shrinkage + holdout must stay the gate.
- First try (2026-11-12, stage 25 `clusters`): k-means (k=32) on Maia-3 23m block-3 hidden states averaged over
  squares, fixed Elo 1900, fitted on all peer bands, your positions matched. Held-out peer log-loss gain over 24's
  features +1.7e-4 [-8.1, +13.0] (a permuted label of the same size costs ~-30e-4; readable-feature k-means
  -8e-4): the clusters are redundant with 24's features, not noise. Naming rules are material / move number /
  queens slices (F1 mostly 0.3-0.6; rook endings 0.80). Square-averaging likely throws away the spatial pattern.
  Next variants: `--layer -1`, a later block, k=64; per-square or attention-based pooling; then an SAE.
  `--layer -1` (final pooled vector): gain +34.9e-4 [+18.8, +48.4] but positive in only 3 of 5 bands, and the
  rules lean on win chance / critical / legal moves = a difficulty signal, not nameable patterns. Verdict:
  position clustering is retired as a concept source.
- **Rethink (2026-11-12, second research round): concepts = named line patterns.** Opportunity (best line has
  pattern X, 2nd-best doesn't, gap ≥ 0.05; label confidence from that gap + agreement at 4× nodes) → choice
  (played move starts X; record the alternative chosen) → cost (win prob lost when declining). Estimand: uptake
  vs a peer logistic model with difficulty / rating / clock as controls; rating slope of peer uptake decides
  skill (Elo) vs style (no Elo claim); Elo ≈ $695 \cdot f \cdot \delta \cdot \bar c$ (opportunities per game ×
  shrunk uptake deficit × mean cost when declined). Reverse errors (greedy grab, premature queen trade, unsound
  sac) use 24's observed / expected event path. Peer frequencies on settled lines (2000–2200 band): any
  sacrifice 1.25% of positions (uptake 58%), endgame pawn sac 0.25%, pawn break 180 / 17k (49%, rising
  0.28 → 0.75 across bands vs 0.42 → 0.55 for best-move rate), queen trade 562 (73%), endgame king walk 139 (78%).
  Next steps, ranked:
  1. Settle the extra bands (03 on `band_*`, register in main.py; ~20–40 min); 24 should load settled lines.
  2. `coach/linepatterns.py`: per-ply replay of best / 2nd / played lines (material trajectory, move events,
     existing motifs) → ~13 rule-defined patterns, each with an advice sentence; unit tests, ≥ 85% precision
     by eye on sampled boards, a negative-control pattern.
  3. Maia difficulty (P(best), entropy) as a stage-1 control in 24 (~5 min GPU).
  4. Uptake model + honest gate in 24: family → pattern shrinkage, MDE column, peer bootstrap in
     `calibration()`, BH, simulated null for the holdout correlation, injected-effect test.
  5. Label stability: re-search a sample at 4× nodes (flip rate < 25%).
  6. Mine ~3× more peers (per-band opportunity counts of 10–90 are the real limit).
  7. Report wording; later, PrefixSpan mining of peer lines as a candidate generator only.
- **Built (2026-11-13, steps 1–5)**: `settle-bands` stage (03 `--set bands`); 24 on settled lines + Maia
  difficulty control; `coach/linepatterns.py` (15 rules incl. negative control `ctrl_light_square`,
  `<set>/linepatterns.parquet`); uptake model + gate in 24; `scripts/diagnostics/lp_stability.py` (4× nodes).
  - Precision by eye (Sonnet reviewers, ~20 peer opportunities per pattern, 3 rounds of rule fixes): ≥ 85%
    retreat, quiet_defence, passer_push, forcing_checks; 75–80% queen_trade, pawn_break, king_activation, fork;
    below 70% (exploratory, never claimed): sacrifices (35–55%), line_tactic 50%, simplify_ahead 35%. The
    sacrifice rules fail on exchanges/desperados read as sacs and on lines that still end mid-tactic; a
    sacrifice needs a "voluntary + compensation" test the material trajectory alone doesn't give (try:
    the sacrificed piece was safe before the move, the 2nd line keeps it, eval holds at 4× nodes).
  - Stability at 4× nodes (25 per pattern): overall flip rate 28% (target < 25%); ≤ 24% only for
    forcing_checks, fork, quiet_defence, retreat; queen_trade / pawn_break 32%, king_activation 56% (n 9).
  - First real run: no family passes the gate (an overall decline ratio and two families are leads only;
    numbers in `docs/local/status_log.md`); the negative control includes 1.
  - Peer opportunities are the binding limit: ~2,200 over 5 bands (~7 per 100 positions with the control
    excluded); the player ~1,300 inside the band range. Injected ×1.5 decline rate is found 60–90% of the
    time per family, the king family (44 opportunities) only ~7%. Results: `docs/local/status_log.md`.
  - Next: (a) mine more peers (step 6) — every family is under-powered at pattern level; (b) better sacrifice
    rules (above); (c) reverse errors (rows with opportunity False, took True) via 24's O/E path;
    (d) peer bootstrap for the concept calibration too (only the line-pattern path has it).

### Position clusters for puzzles (2026-11-12) — status: built (stage 25), exploratory
- What: stage 25 now picks puzzles. It groups the player's critical errors (whole history) into recurring kinds of
  position and drills a few representative, non-duplicate ones per group, plus "more like this" peer positions
  from the same cluster. Clusters decide WHICH positions to drill. Line patterns (Concept groups) decide WHAT to
  train. Clusters make no concept, weakness or Elo claims.
- How: Maia-3 block-3 states at Elo 1900, "move" pooling = [square mean, best move's from-square, to-square];
  k=64 k-means on his critical errors + peers' critical positions; sets ranked by total win prob lost; inside a
  set `coach.clusters.pick_puzzles` (typical + recent + costly, one per game / board / opening line, embedding
  distance) picks 12; peers' cluster-mates (cost = gap to the 2nd move) give 8 more. Output: in-depth items +
  `player/clusters_drills_*.pgn`.
- Judged only by coherence: pairwise tag agreement vs size-matched random sets (`coherence()`), plus a blind
  eyeball check. First real run: the sets are "same piece moves the solution" (99.6% vs 18%) × move kind ×
  phase, not tactic motifs (motif 40% vs 35%). Blind reviewers called all 6 top sets partly/yes and both random sets
  no ("Pawn push hits a piece" 11/12, "Knight tactics" 10/12). Good enough for drill selection, but the names are
  shallow.
- Next:
  1. Once `linepatterns.parquet` exists, add the pattern as a coherence label and try sets = pattern × cluster
     (the pattern names the set; the cluster splits a big pattern into position types; `pick_puzzles` picks).
  2. Pooling without the piece identity (e.g. the to-square state only, or the state minus its token's piece
     embedding), so clusters follow what the move does rather than which piece moves.
  3. Better set names from the eyeball labels: a small rule on the best line (hits a piece with tempo, check,
     capture, retreat) would cover most of what the reviewers named.
  4. Closed loop: re-run after he drills a set and track whether the set's errors per 100 moves fall (puzzle
     selection only; any Elo claim still goes through 24).
  5. Attention pooling or an SAE only if 1-3 still leave the sets shallow.

### Time management model (2026-11-17) — status: designed, not built
- **Why**: critical positions with a forcing move while pressing are missed more than peers only with
  < 3 min left (at par with time); long thinks in "easy" positions go wrong often (34% vs 7% at the
  same Maia difficulty); he reaches late phases with less clock than peers.
- **How**: P(error | think time, Maia difficulty at his level, critical, forcing/quiet, stakes) from
  ~38k moves → marginal value of a second per position type → optimal allocation of 600 s/game
  (equalize marginal value, like an engine time manager) vs his actual allocation. Turn it into
  over-the-board rules from visible board features (opponent just captured/checked, many captures
  available, king exposed) via a small "time pays off here" classifier.
- **Risk**: think time is endogenous (he thinks long when something feels off). Use clock remaining as
  an instrument (same difficulty, different clock → forced shorter thinks) to separate "time helps"
  from "hard positions take time". Peers' per-move think times need a clock-only re-mine of the dump.
- Maia lines as a model of what he calculates: speculative, parked.

### Clean endgame / plan execution (2026-11-17) — status: idea
- **What**: detect the endgame's idea (king route to a target pawn, supported passer push, rook behind
  passer, trade into won K+P, breakthrough; opposition/triangulation via tablebase) from features
  changing along a long engine line, then score how well his next moves follow it.
- **How**: deep line (06c-style, 20+ plies) from the first contested/winning endgame position;
  classify by endgame_features trajectories; execution = his moves moving the same features the same
  way + eval stability + moves to convert (tablebase DTZ gives an exact yardstick ≤ 5 pieces).
- **Risk**: plan detection needs eyeball validation first (prototype: print "idea: Kg5 then xh5" next to
  his moves for 20 rook/K+P endings). Peers have only ~12 sampled positions per game → you-only unless
  full peer endgames are re-mined (more engine time).

### Repertoire extension: "your lines, extended" (2026-11-17) — status: designed, not built
- **Why**: in openings he plays a lot, a move can be fine and still not the best; and past the point
  where he stops knowing theory he should know the follow-ups (essential at 2000+). Engine-only, so it
  belongs in the main opening analysis (14), not the exploitation layer. Example: after 1.e4 e5
  2.Nc3 Nc6 3.Bc4 Nf6 he played 4.f4 (−11.6; ...Nxe4 tricks); the right move (our 300k run: 4.d3, with
  4.Nf3 0.8 behind) and then the right answer to each common Black reply.
- **How**: roots = his-to-move positions he reaches often (recent games). His nodes: Stockfish best at
  deep search (1M+, refine deepens), alternatives within ~2 points listed as "also fine", his usual
  move with its cost even when small ("good, but X is better by 1.5"). Opponent nodes: replies covering
  ~90% of the book mass at his opponents' band (option: the band he's aiming for, e.g. 2200), Maia when
  the book is thin; frequent mistakes flagged with the punishment. Depth: to move ~15 or 3–4 of his moves
  past the point where he leaves his usual moves; prune replies < 5%. Rank by frequency × improvement.
- **Output**: report section with the trees + one Lichess study per opening (chapters with variations,
  comments: "you usually play X (−1.2)", "40% of 2000s play Y here"), shared with the drills export.
- **Cost**: a few thousand positions at 1M nodes (~1–2 h), resumable; reuses 14's book and engine cache.

### Smaller layer-1 follow-ups
- **Critical × clock row**: add "critical, clock < 3 min" directly to 16 (currently only implied).
- **Overthinking check**: on long thinks in easy positions, does he deviate from Maia's top move?
- **Per-game level from time** (rolling Maia fit) instead of rating + fitted offset.
- **Per-opening peer baseline** in 14 A (store the Lichess Opening header in peer_out; needs `--rebuild`).
- **Opening coverage beyond move 12**: repeated-position approach stops there; moves 10–15 need a
  structure-level method (cluster by pawn structure / opening family, not exact positions).
- **6–7 piece tablebases** via the Lichess API for real rook endings (rate limits; cache).
- **Valuation vs calculation split**: depth at which the idea's features first appear along the line.
- **Maia-3 SAE features** to replace handcrafted ones (plan in CLAUDE.md "Next steps" §4: topK SAE,
  FVU/dead/splice acceptance tests, NMF must beat the column-shuffled null).
- **Prospective validation**: do idea/category miss rates predict next month's errors better than a
  rating-only baseline? The real test that any of this is more than description.
- **Grow the peer set** in refine.py (population positions beyond 20k; needs 02 + 03 on the new ones,
  since load_analyzed refuses partial settled coverage).

### Single-game analysis tab (2026-11-18) — status: idea
- Upload one game (PGN) → run the pipeline on just that game → a report tab tailored to it: game
  rating + per-phase ratings (red → green, as chess.com), best moves / mistakes / blunders, opening ideas
  (vs the repertoire), tactics, endgame ideas, the idea concepts — and how this game compares with the
  rest of his games (percentiles per measure). Needs: a single-game path through 02/03/05/09b (engine
  at depth, Maia), the per-game rating (quantile mapping onto the band per-game distributions), and
  the report generator (option A HTML) to render one game. A local web app (option B) would make the
  upload step natural; with option A it's a command (`python analyse_game.py game.pgn`) that writes the tab.

### Over-the-board games: classical + 30+30 (2026-11-18) — status: idea
- ~10 classical and ~15 30+30 games played in person, as PGN. Same upload path as the single-game tab,
  plus a batch import (`import_pgn.py games.pgn --source otb`) producing the standard position schema
  with `source = "otb"`, so every stage can include or exclude them.
- Differences to handle: no clock data in most OTB PGNs (skip clock / think-time measures; keep them
  NaN, not 0); different time control → don't pool with chess.com rapid for "vs peers" (peers would be
  Lichess classical, a separate band) — or report them as their own small section; ratings are
  FIDE/national (own mapping, or Maia best-fit per game only). n ≈ 25 games → per-game analysis and
  description (game ratings, worst moments, opening lines vs the repertoire), not statistical claims.
- Useful comparison: same measures OTB vs online (does slow play change the critical-position gap?),
  with the small n made explicit.

## Layer 2: exploitation (exploratory)

- **Safety budget over a whole game** (Ganzfried & Sandholm: only risk what the opponent has gifted):
  needs sequential positions; his games have every move, so it can be simulated on his games.
- **Deeper expectimax** (13 is 3 plies): 4–5 plies with a p-mixture, pruned by Maia mass.
- **Opponent scouting** from any chess.com username: their tree + mistakes → prep sheet (traps 18
  applied to their repertoire, shrunk toward the Elo model for opponents with few games).
- **Self-exploitability**: run the exploitation machinery on himself — how would an opponent who
  knows his tendencies beat him? Needs a personalized model.
- **Individual Maia** (fine-tune on his ~38k positions; McIlroy-Young et al. style) — only if it beats
  base Maia on held-out games; then everything in layer 2 becomes personal.
- **Endgame tricks**: stalemate / fortress / perpetual resources from the same tree search (13), for
  lost positions (and the trap side when winning).

## Language layer (2026-11-18) — status: idea
### Repo-local coach skill (no training)
- `.claude/skills/coach/SKILL.md`: before answering, read `data/players/<user>/findings/*.json` (+ status
  log); quote numbers with intervals; no generic advice without a finding; commands: "review this game"
  (22_import_games on the PGN, walk the worst moments with the why-text), "plan my week" (top 2–3 Elo-gain
  findings → drill sets with spaced repetition), method rules from CLAUDE.md (Maia-only = style, never-
  resign caveat). Third-party skills: read their files before installing (`npx skills add` runs remote code).
### Puzzle-explanation fine-tune (learning project)
- Narrow job: structured facts (eval, best line SAN, played move + loss, top feature deltas, motif,
  weakness category) → 2–4 sentence coaching explanation. Engine = source of truth; the LM only narrates.
- Baselines first: [Second-1.0-2B](https://huggingface.co/joebos/Second-1.0-2B) (engine facts → coaching
  prose, Qwen-based, 0.8–9B variants) and C1 (CSSLab, arXiv 2603.20510, 4B puzzle explanations). Related:
  MATE (1M annotated positions, Llama-3-8B), ChessGPT (2023 data/models).
- Data: a few thousand items from our pipeline, targets written by a strong model (distillation),
  automatically filtered (every move mentioned legal and in the engine line; claims match the facts).
- Train: QLoRA on a 1–4B model (Qwen); GTX 1660 6 GB can just about do 1–2B slowly; a rented GPU for a
  few hours is easier. Eval: automatic hallucination rate (unsupported moves/claims) + the player rates 50
  samples vs baseline. Plug-in: drill/report puzzle comments call it (one setting to swap models).

## Modelling human search (2026-11-13) — status: stage 1 done (error anatomy)
- Stage 2: fitted search model (Maia policy as candidate generator for both sides, depth/breadth/leaf
  noise, think time from board features), validated by held-out likelihood vs plain Maia. New branch,
  only if stage 1 shows he searches differently from peers (it showed little difference so far).
- Identifiability: one move per position → only aggregate search parameters are estimable.

## Engine-side ideas (2026-11-13) — status: parked
- Abstract planning / endgame concepts to guide search, long-term planning in engines. Research
  direction, outside the coaching tool.

## Report / tooling
- Report generator per `docs/report_design.md` (waiting on the player's 3 answers).
- LLM verbalization of findings grounded on the computed numbers + positions (never free text).
- Generalize for other players: `docs/generalize.md`.
