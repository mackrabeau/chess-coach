# Weakness model (design, 2026-11-12) — status: v1 built (`scripts/24_weakness_model.py`), runs beside 17b

## v1 as built (what differs from the design below)
- Motifs: the existing `coach/motifs.line_motifs` (already used by 06c/07b/08; fork / pin / skewer /
  discovered count only when cashed in). missed = mover's moves of the best line, allowed = opponent's replies
  in the played line. Lichess's puzzle tagger is AGPL (this repo is MIT), so it isn't ported.
- Stage 1: sklearn HistGradientBoosting (no new dependency), early stopping; rare-event models shallow.
  5-fold cross-fit on peer games; per-concept calibration factors make peers' out-of-fold O/E = 1 per concept
  (without it rare "allowed" events were under-predicted 1.3–2×, which inflated exactly those findings).
- Lost positions (wp_best < 0.15) are dropped for everyone; player games outside the bands' rating range
  (Lichess scale) are dropped, which replaces `min_player_rating` here.
- No Maia difficulty feature yet (bands have no 09b curves); difficulty comes from engine features (gap to
  2nd best, criticality, eval, legal moves).
- Elo value is shrunk with the same weight as the count ratio; diagnostic concepts never carry Elo.


One model replaces the many separate player-vs-peer slice tests (08 idea tests, 16/17 per-category fits, 17b
findings) with: **what error rate would a typical player at your rating have in exactly your positions, and where
do you beat or miss that?** Same idea as indirect standardisation in epidemiology (observed / expected).

## Why
- Dozens of slices × bootstrap CIs → some look "clear" by chance; borderline leads get over-read.
- Slices confound: "you err more in critical positions" may just mean your critical positions are harder.
  Conditioning on difficulty (Maia p(best), eval, clock) removes that.
- Slices are coarse diagnostics ("critical positions"), not trainable. Findings should resolve to drillable
  concepts (a tactic motif, an endgame type, an opening line).

## Data
- Rows = analysed positions. Peers: Lichess bands (now ~20k in the home band + 4k in each of 4 other bands;
  more is cheap to mine, Stockfish is the cost). Player: every move, each game at its own level (Maia offset).
- Targets: `err` (mistake/blunder, 0/1) for frequency; win-prob loss for value.
- Features:
  - Controls (not findings): rating r, clock, eval band, Maia p(best) / difficulty, phase, move number.
  - Concepts (findings): `coach/categories.py` + handcrafted positional features + **tactic motifs** (new).

## Concept library (stage 0 — the trainability lever)
- Motifs from the engine lines, in two directions:
  - **missed**: the best line contains motif m and the player didn't play it (offence: fork, pin, skewer,
    discovered attack, deflection, back rank, promotion, mate-in-n...).
  - **allowed**: the refutation of the played move contains motif m (defence: walked into a fork...).
- Reuse, don't rebuild: Lichess's puzzle theme tagger (lichess-puzzler, open source) detects these themes on a
  line; port or call it on best line / refutation. Check its licence and what it needs as input first.
- Endgame types (`coach/endgame_features.material_class`), opening lines (ECO / tree node), positional
  features stay as concepts.
- Each concept carries a trainability tag: motif / endgame type / opening line = drillable; broad slices
  (critical, contested, phase) = diagnostic only, shown as context, never as the headline.

## Stage 1: peer model (prediction only)
- $\hat p(x, r) = P(\text{err} \mid x, r)$ fit on peers across all bands. Gradient boosting (LightGBM) or a GAM;
  it only has to predict well, interpretability comes from stage 2.
- Validate: calibration per band and per concept on held-out peer games (reliability curve, Brier).
- Rating enters as a feature, so $\partial \hat p / \partial r$ per concept = how much that concept improves
  with skill (skill vs style, what 17 does with lines today).

## Stage 2: player residuals per concept
- For concept k over the player's positions $i \in k$ (games ≥ the comparison cut-off, rating = own level):
  $O_k = \sum_i \text{err}_i$, $E_k = \sum_i \hat p(x_i, r_i)$; same with win-prob loss for value.
- Excess loss per game $\Delta_k = (L^{obs}_k - L^{exp}_k) / n_{games}$ → Elo value $\approx 695\,\Delta_k$ (the
  existing proxy; upper bound).
- Shrinkage: empirical Bayes across concepts (gamma-Poisson on $O_k / E_k$), so rare concepts are pulled to 1
  instead of producing extreme ratios. CI by game bootstrap of the whole stage 2.
- "Plays like X in concept k": the rating at which the peer model's expected rate equals the player's observed
  rate (only where the concept has a clear rating slope; flat = style).

## Ranking
value_k = Elo value × frequency (positions per game) × trainability. Headline = top few drillable concepts
with CI excluding 0; each links to puzzles from the player's own games (missed/allowed positions in k).

## Built-in validation (the success check)
- Temporal holdout: find weaknesses on the older 70% of games, test $O_k/E_k$ on the newest 30%. Concepts that
  don't replicate are not reported as weaknesses.
- Later, re-runs after training: did the trained concept's ratio move?

## Caveats
- chess.com player vs Lichess peers: mapped via the fitted Maia offset; residual bias shows up as an overall
  ratio ≠ 1, so report concepts relative to the player's overall ratio as well.
- Peers are ~12 sampled positions per game: fine as long as sampling doesn't depend on the outcome.
- Never-resigning players: eval band is a control; prefer contested positions for headline findings.
- ~36k peer rows / ~10% errors → per-concept rating slopes are rough; mine more band data if stage 1 is
  under-powered for a concept.

## Build order
0. Motif tagger (missed/allowed) on existing data; check concept frequencies per game.
1. Stage 1 peer model + calibration report.
2. Stage 2 + temporal holdout on real data, side by side with the current 08/16/17 findings.
3. Wire into `coach/findings.py`; retire the slice findings it supersedes (remove dead paths, don't keep both).
