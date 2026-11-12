# Working on this repo in parallel (several agents / branches)

The codebase is split so that independent work touches different files.

## The contract: findings
- An analysis module is one script (`scripts/NN_name.py`, `NNb_` for a follow-up of NN) that reads
  data, writes its own outputs under `data/<module>/` or `data/player/<module>*`, and publishes its
  conclusions with `coach.findings.write(cfg, "<module>", [...])` → `data/findings/<module>.json`.
- The report (`20_report.py`) only reads findings. **To add an analysis, don't edit the report**:
  write findings (section, tag, elo, examples, drill) and they appear, ordered by expected Elo gain.
- Schema and the Elo convention live in `coach/findings.py`. Change the schema only additively
  (new optional keys), so old modules keep working.

## Where to put code
- Shared logic goes in `coach/` (one concern per module: `runner.py` engine passes, `results.py`
  loading, `categories.py`, `openings.py`, `maia.py`, `study.py` PGN studies, `webreport.py` HTML
  components, `why.py`, `game_rating.py`). Prefer adding a new module over growing an existing one.
- Pipeline order: one `Stage(...)` line per script in `main.py` (inputs/outputs for staleness).
  Adding a stage = adding one line in the right place; that's the only shared edit.
- Long compute: make it resumable with `coach/parts.py` (atomic part files keyed by pos_id) or the
  opening cache (`coach/openings.evaluate`, deepest result wins), so `refine.py` can deepen it.

## Branches
- One branch per module/feature; keep commits inside that module's files plus its `main.py` line.
- `CLAUDE.md` is compact on purpose: add one line to "Pipeline" / "Open / next" per feature, put
  details in the script docstring or `docs/status_log.md` (append-only, newest first).
- Test on fake data (`tests/make_fake_data.py`, see CLAUDE.md) before pushing; `pyflakes coach scripts`
  should be clean.

## Things that are shared state (coordinate before changing)
- `data/` layout and `pos_id` format, `coach/mine.SCHEMA`, `coach/engine.RESULT_SCHEMA`.
- `configs/*.yaml` keys (add new blocks; don't rename existing keys).
- `coach/findings.py` schema, `coach/webreport.py` CSS/JS (report look).
