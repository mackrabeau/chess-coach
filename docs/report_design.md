# Report and drills: design proposal

Status: proposal, not built. Agree on this before writing the report generator.

## What the report is for

One question per section: *what costs you points, how sure are we, and what do you do about it?*
Every finding uses the same block:

1. **Claim** in one sentence ("when pressing in the middlegame with one forcing move, you miss it more often than peers").
2. **Evidence**: the numbers, n and 90% CI, and which measures agree (engine error rate, Maia level, outcomes).
3. **Confidence tag**:
   - *solid*: several measures agree and the CI is clear.
   - *lead*: one measure, or a borderline CI.
   - *style*: different from peers but not costing points.

   Style findings are shown in their own box, so they never read like weaknesses.
4. **Examples**: 2–3 of your own positions (board, your move, the best move, end of both lines on hover).
5. **What to do**: a drill set (link to the Lichess study chapter group) or a concrete line to learn.

## Structure

```
Summary            (one page: rating picture, top 3 fixes, top 2 strengths, what's style)
1. Analysis        (layer 1: you vs reference, engine + stats)
   1.1 Opening     (engine-only: lines that cost you, must-know positions, out-of-opening eval)
   1.2 Middlegame  (critical positions, tactics track, positional concepts, think time)
   1.3 Endgame     (conversion / holding by outcome, tablebase, balanced endgames, by type)
   1.4 Across phases: score accounting (where points go), clock, progress over time,
       skill profile (Maia level + engine), multi-band (skill vs style)
2. Exploitation    (layer 2, exploratory; looser standard of evidence, labelled as such)
   2.1 Opening traps (book error rates at your opponents' level + Maia for rare lines)
   2.2 Practical moves (12/13: traps within a small margin, lines you can execute)
   2.3 Endgame tricks (later: stalemate / fortress resources from the same tree search)
3. Drills          (puzzles from your games, grouped by finding; the Lichess study file)
Appendix           (method notes, every table in full)
```

## Format: static HTML + a Lichess study (recommended)

- **Report: one self-contained HTML file** (what we have now, reorganized). It opens in any browser, needs no server and can be shared as a single file. A small amount of inline JS for tabs/collapsible sections and the hover popups we already have. Interpretability comes from the finding blocks, not from interactivity.
- **Drills: a Lichess study PGN** (`data/player/drills.pgn`, import at lichess.org/study → "import PGN"). Each puzzle is a chapter: your position (FEN), the solution line, and a comment with *why* (the feature difference along the best line, e.g. "best line wins the e5 pawn / keeps the bishop pair"). Lichess's interactive lesson mode handles "find the move" with feedback, so we don't need to build a puzzle UI. Studies hold 64 chapters, so write one file per theme (critical/pressing, rook endings, opening lines, …).
- A full web app (server, database, accounts) would be more work for little gain in interpretability; worth it only if this becomes a tool for other people.

## Why this shape (learning evidence)

- **Practice testing (retrieval) and spaced practice** are the two techniques rated "high utility" in Dunlosky et al.'s review, and a 2021 meta-analysis of the ten techniques (242 studies) reached the same ranking ([Frontiers in Education 2021](https://www.doi.org/10.3389/feduc.2021.581216)). → Drills are *find the move* (retrieval), never "read the answer". The study file gets reused over weeks (spacing), and small themed sets keep sessions short.
- **Interleaving** helps later, not during first learning (blocked first, then mixed) ([summary](https://digitalpromise.org/2015/02/07/five-learning-strategies-that-work/)). → Each theme gets its own chapter group first, then a mixed review chapter set.
- **Self-explanation / elaborative interrogation** are rated moderate. → Every puzzle carries a *why* comment generated from the feature delta along the engine line (the core idea of this project).
- **Serious study alone** is the strongest predictor of chess skill among measured activities ([Charness et al. 2005](https://zendy.io/title/10.1002/acp.1106); reanalysis [Burgoyne et al. 2019](https://englelab.gatech.edu/articles/2019/Burgoyne_et_al_2019_AJP.pdf)). Deliberate practice explains a sizeable but partial share of skill ([APS summary](https://www.psychologicalscience.org/news/releases/deliberate-practice-necessary-but-not-sufficient.html)). → The report's job is to point the study hours at the positions that cost points.
- **Repetition of the same puzzle set** (Woodpecker method) has only observational / vendor evidence ([Disco Chess data](https://lichess.org/@/DiscoChessApp/blog/woodpecker-method-training-431000-puzzles-60000-rated-games-and-35-elo/yrWsFF1N); [skeptical take](https://zwischenzug.substack.com/p/the-woodpecker-method)). → Repetition is offered, but with expanding intervals (spacing) rather than the shrinking ones the method uses.
- **Own-game positions** are not tested directly in that literature. They are used because they are the positions you actually reach (transfer), and they double as the eyeball check of each finding.

## Puzzle selection

- Only from findings tagged *solid* or *lead* (never style).
- One clear best move (critical: best beats 2nd by ≥ 10 points), solution verified at a deeper search (refine pass), and a loss ≥ the mistake threshold in your game.
- Grouped by finding (e.g. "pressing, forcing move, < 3 min"). Each group holds at most 15, chosen for variety (different openings/structures), with recent games first.
- Exploitation section: a few illustrative positions, not drills.

## Decisions (Mack, 2026-11-18)

1. **Format**: whatever looks and feels best; open to an interactive website (feasibility discussed:
   v1 = one self-contained HTML with inline JS (tabs, collapsible findings, boards you can step through,
   puzzle mode via chess.js), no server; v2 option = local web app (Streamlit/FastAPI over the parquet
   files) for free exploration; hosted site not planned (private data, hosting cost)).
2. **Drills**: several Lichess studies, one (or more) theme per study, many puzzles each.
3. **Ratings**: include rating estimates (from the multi-band fits), relative statements as fallback.
   Plus a **game rating** like chess.com's: per game, a rating for the game and for opening /
   middlegame / endgame on a red → green scale (from per-phase loss mapped onto the band distributions
   by quantile, plus the Maia best-fit Elo of the game's moves; per-game values are noisy, so show the
   range and aggregate by month).
