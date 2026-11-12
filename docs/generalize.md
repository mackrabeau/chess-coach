# Where the code is tailored to one player (notes for generalizing)

Kept up to date as we go. Each item: what is specific, where, and what a general version needs.

## Data source and identity
- **chess.com only** for the player (`coach/fetch_chesscom.py`, `00_fetch_player.py`, `15` terminations,
  `coach/openings.player_tree`). A Lichess player would need a Lichess fetcher producing the same
  position schema (`coach/mine.SCHEMA`) and the same archive → tree path.
- **One username** in `player.username`; one player per data folder. Multi-player = one config/data dir each.
- **Time class rapid** (`player.time_class`) and the peer filter `filters.rapid_*`. 10|0 is assumed in
  some caveat texts (11c, 16 notes), not in code.

## Ratings and levels
- **Peer band 2000–2200** (`population.rating_center: 2100`) was chosen for chess.com ~1920. A general
  version should pick it from the player's rating via the Maia offset (09b) or a chess.com→Lichess table.
- **`player.min_player_rating: 1750`**: the cut-off for "vs peers" comparisons, picked because this
  player climbed from 960. General: compare within a rolling window of the player's recent level, or
  map every game to its own level (as 16 does with the Maia offset).
- **Maia offset / slope** (`maia.player_offset: 160`, `player_offset_slope: -0.10`) are fitted values
  for this player (09b `--fit-only`). They must be refit per player; nothing else should hard-code them.
- **Opponents' level** in 18 comes from his games (median opponent rating mapped to Maia's scale): general.

## Behaviour-specific caveats baked into interpretation (not code)
- **Never resigns**: error rates in lost positions and endgames look better than they are. Texts in
  11/16/17 tell the reader to prefer contested rows. For a player who resigns normally this caveat
  is moot; a general report should detect it (share of games played to mate/flag from a losing eval)
  and only then show the warning.
- **Rating lag** (fast climber): handled by the rating-dependent Maia offset; status notes mention it.

## Report wording
- Reports say "you" and refer to his findings in a few hard-coded notes (e.g. opening caveats in
  14's docstring, the moves 10–15 remark). The planned report generator should take all claims from
  computed results, never from text written for one player.

## Environment
- `configs/v0.yaml` holds his paths (Windows, `D:\chess\...`) and fitted values. A new user should copy
  it, set `player.username`, `paths.*`, and refit the Maia offset. A `configs/example.yaml` with
  placeholders would make this explicit.
- Hardware assumptions (workers 10, GTX 1660) live only in the config.
