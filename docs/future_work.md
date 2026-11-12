# Future work and parked ideas

Ideas worth exploring that aren't built yet, roughly grouped. Each: what, why, how, and the known risk.
"Status" says how far it got. Keep adding; move an item out when it's built (and note it in CLAUDE.md).

## Layer 1: analysis

### Time management model (Mack, 2026-11-17) — status: designed, not built
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

### Clean endgame / plan execution (Mack, 2026-11-17) — status: idea
- **What**: detect the endgame's idea (king route to a target pawn, supported passer push, rook behind
  passer, trade into won K+P, breakthrough; opposition/triangulation via tablebase) from features
  changing along a long engine line, then score how well his next moves follow it.
- **How**: deep line (06c-style, 20+ plies) from the first contested/winning endgame position;
  classify by endgame_features trajectories; execution = his moves moving the same features the same
  way + eval stability + moves to convert (tablebase DTZ gives an exact yardstick ≤ 5 pieces).
- **Risk**: plan detection needs eyeball validation first (prototype: print "idea: Kg5 then xh5" next to
  his moves for 20 rook/K+P endings). Peers have only ~12 sampled positions per game → you-only unless
  full peer endgames are re-mined (more engine time).

### Repertoire extension: "your lines, extended" (Mack, 2026-11-17) — status: designed, not built
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

### Single-game analysis tab (Mack, 2026-11-18) — status: idea
- Upload one game (PGN) → run the pipeline on just that game → a report tab tailored to it: game
  rating + per-phase ratings (red → green, as chess.com), best moves / mistakes / blunders, opening ideas
  (vs the repertoire), tactics, endgame ideas, the idea concepts — and how this game compares with the
  rest of his games (percentiles per measure). Needs: a single-game path through 02/03/05/09b (engine
  at depth, Maia), the per-game rating (quantile mapping onto the band per-game distributions), and
  the report generator (option A HTML) to render one game. A local web app (option B) would make the
  upload step natural; with option A it's a command (`python analyse_game.py game.pgn`) that writes the tab.

### Over-the-board games: classical + 30+30 (Mack, 2026-11-18) — status: idea
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

## Language layer (Mack, 2026-11-18) — status: idea
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
  few hours is easier. Eval: automatic hallucination rate (unsupported moves/claims) + Mack rates 50
  samples vs baseline. Plug-in: drill/report puzzle comments call it (one setting to swap models).

## Modelling human search (Mack, 2026-11-13) — status: stage 1 done (error anatomy)
- Stage 2: fitted search model (Maia policy as candidate generator for both sides, depth/breadth/leaf
  noise, think time from board features), validated by held-out likelihood vs plain Maia. New branch,
  only if stage 1 shows he searches differently from peers (it showed little difference so far).
- Identifiability: one move per position → only aggregate search parameters are estimable.

## Engine-side ideas (Mack, 2026-11-13) — status: parked
- Abstract planning / endgame concepts to guide search, long-term planning in engines. Research
  direction, outside the coaching tool.

## Report / tooling
- Report generator per `docs/report_design.md` (waiting on Mack's 3 answers).
- LLM verbalization of findings grounded on the computed numbers + positions (never free text).
- Generalize for other players: `docs/generalize.md`.
