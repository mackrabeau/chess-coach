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
