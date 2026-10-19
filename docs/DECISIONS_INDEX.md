# Decisions Index

| ID | Decision | Status | Scope | Impact |
|----|----------|--------|-------|--------|
| DEC-001 | Scheduler phase params default to optimizer-update units; configs may opt into displayed training-step units with `phase_step_unit: training_steps`. | Active | `toolkit/scheduler.py`, `jobs/process/BaseSDTrainProcess.py` | Medium |

## Notes
- Keep only active decisions here.
- Add a decision when a repo-wide contract, architecture boundary, dependency policy, or long-lived fork behavior changes.
- Prefer concise entries and link deeper context from a separate decision note only when needed.

| DEC-002 | FLUX.3 edit giữ raw ordered controls; budget reference độc lập là opt-in, scheduler fork và legacy pins không đổi | Active | FLUX.3 model/paired training | High |

| DEC-003 | Colab dùng requirements.txt của main theo luồng notebook V7/V8; dependency baseline nằm trong repo, không ép manager Torch/CUDA hoặc global pip checks từ notebook | Active | Colab/install | Medium |

| DEC-004 | FLUX.3 dùng BFL pad position1 và cache CFG pair ở window chung; text cache revision mới, latent/TAR không đổi | Active | FLUX.3 text/cache/preview | High |
