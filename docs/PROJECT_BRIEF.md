# Project Brief

## Purpose
- AI Toolkit is a Python-based training toolkit for diffusion/model workflows, with a Next.js UI for local or containerized operation.
- Primary users are model trainers/operators running jobs from configs, scripts, CLI entry points, or the web UI.
- Keep training/runtime behavior stable when syncing fork changes with upstream.

## System Shape
- Python runtime entry points live at the repo root, including `run.py`, `run_modal.py`, and `flux_train_ui.py`.
- Core training, model, dataset, optimizer, sampler, and utility code lives under `toolkit/`.
- Built-in extension behavior lives under `extensions_built_in/`; user/example extension space lives under `extensions/`.
- Web UI lives under `ui/` and uses Next.js/TypeScript.
- Container/dev deployment is described by `docker-compose.yml`, `docker/Dockerfile`, and root dependency files.

## Main Modules
- `toolkit/`: core Python training/runtime library.
- `extensions_built_in/`: bundled extension integrations.
- `extensions/`: external or example extension area.
- `ui/`: Next.js UI and related worker/config files.
- `config/`, `jobs/`, `notebooks/`, `scripts/`, `testing/`: examples, jobs, notebooks, scripts, and test/support assets.
- `docker/`, `docker-compose.yml`: container packaging and runtime wiring.

## Global Invariants
- Do not change public config/job semantics without checking examples and migration impact.
- Keep fork-specific changes visible during upstream sync; prefer merge/rebase analysis before overwriting files.
- Do not treat generated outputs, datasets, or local caches as source changes.

## Build / Test / Lint
- Python install: `pip install -r requirements.txt`
- UI install: `cd ui; npm install`
- UI build/checks: inspect `ui/package.json` scripts before running.
- Docker run path: `docker-compose.yml`

## Module Boundaries
- Root scripts should orchestrate `toolkit/` behavior instead of duplicating core logic.
- `ui/` should communicate through the existing app/runtime interfaces rather than reaching into unrelated Python internals without a clear contract.
- Built-in extensions should remain self-contained unless a shared utility belongs in `toolkit/`.

## Safety Constraints
- Upstream sync work should preserve local fork commits unless an intentional replacement is documented.
- Resolve merge conflicts by understanding both upstream changes and fork intent, not by blindly taking one side.
- Avoid broad dependency upgrades beyond what upstream already introduced during a sync.

## Key References
- `docs/MEMORY_INDEX.md`
- `docs/DECISIONS_INDEX.md`
- `docs/CHANGELOG.md`
