# Memory Index

## Always Read For Project Task
- `AGENTS.md` if present
- `docs/PROJECT_BRIEF.md`
- `docs/MEMORY_INDEX.md`

## Read By Task Type

### Upstream Sync / Fork Maintenance
- `docs/DECISIONS_INDEX.md` when resolving conflicts that affect contracts or long-lived fork behavior.
- `docs/CHANGELOG.md` for short summaries of completed sync work.

### Python Runtime or Training Change
- Read the relevant files under `toolkit/`.
- Check root entry points such as `run.py`, `run_modal.py`, and `flux_train_ui.py` when behavior changes at startup or orchestration boundaries.

### Extension Change
- `extensions_built_in/`
- `extensions/`
- Relevant config examples under `config/` or `jobs/`.

### Frontend or UI Change
- `ui/`
- Create or refresh `docs/UI_SYSTEM.md` before broad UI work.

### Infra, Deployment, or Environment Config
- `requirements.txt`
- `docker-compose.yml`
- `docker/Dockerfile`
- `ui/package.json`

## Module Map
- `toolkit/**` -> core Python runtime/training logic.
- `extensions_built_in/**`, `extensions/**` -> extension logic.
- `ui/**` -> web UI.
- `docker/**`, `docker-compose.yml`, `requirements.txt` -> runtime/deployment dependencies.
- `scripts/**`, `testing/**` -> maintenance and test/support tooling.

## Decision Lookup
- Read `docs/DECISIONS_INDEX.md` before changing APIs, config/job schemas, dependency strategy, model/runtime contracts, or cross-module boundaries.
- Add a decision only when it remains useful after the current task.

## Task Notes
- Use `docs/tasks/active/<task-id>.md` only for multi-turn or high-risk tasks that need durable handoff notes.

## Archives
- Do not read `docs/archive/*` by default.
- Open archive files only to recover specific historical context.
