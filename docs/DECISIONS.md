# Decisions

## DEC-001: Scheduler Step Units

Status: Active

Scheduler stepping in `BaseSDTrainProcess` happens only when the optimizer updates, not on every displayed training step. Scheduler parameters such as `warmup_steps`, `total_iters`, `T_0`, `T_max`, and `phases[].steps` therefore default to optimizer-update units.

Configs authored in displayed training-step units can set `phase_step_unit: training_steps` inside `lr_scheduler_params`. The trainer will convert those counts to optimizer updates using `gradient_accumulation_steps`.

Use `lr_scheduler: warmup_then_phased_cosine` when `phases` are present. The older `warmup_then_cosine_restarts` plus `phases` path remains supported for compatibility, but emits a warning.
