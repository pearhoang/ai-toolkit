# Decisions

## DEC-001: Scheduler Step Units

Status: Active

Scheduler stepping in `BaseSDTrainProcess` happens only when the optimizer updates, not on every displayed training step. Scheduler parameters such as `warmup_steps`, `total_iters`, `T_0`, `T_max`, and `phases[].steps` therefore default to optimizer-update units.

Configs authored in displayed training-step units can set `phase_step_unit: training_steps` inside `lr_scheduler_params`. The trainer will convert those counts to optimizer updates using `gradient_accumulation_steps`.

Use `lr_scheduler: warmup_then_phased_cosine` when `phases` are present. The older `warmup_then_cosine_restarts` plus `phases` path remains supported for compatibility, but emits a warning.

## DEC-002 — FLUX.3 paired edit sau upstream sync
- Giữ scheduler theo optimizer updates và phase_step_unit của fork khi merge upstream main82e6311.
- FLUX.3 bật raw/multiple-control capabilities; train/preview không bỏ hoặc gộp reference roles. Optional reference_megapixels/reference_resize_method dùng để khớp workflow inference; mặc định sizing upstream giữ nguyên.
- Canonical: docs/modules/flux3-edit-training.md. Kiểm CPU actual data/model methods, chưa full GPU training; không đổi notebook legacy pins.
