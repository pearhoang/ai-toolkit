# Decisions Index

| ID | Decision | Status | Scope | Impact |
|----|----------|--------|-------|--------|
| DEC-001 | Scheduler phase params default to optimizer-update units; configs may opt into displayed training-step units with `phase_step_unit: training_steps`. | Active | `toolkit/scheduler.py`, `jobs/process/BaseSDTrainProcess.py` | Medium |

## Notes
- Keep only active decisions here.
- Add a decision when a repo-wide contract, architecture boundary, dependency policy, or long-lived fork behavior changes.
- Prefer concise entries and link deeper context from a separate decision note only when needed.
