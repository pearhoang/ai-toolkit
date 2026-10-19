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


## DEC-003 — Colab legacy installation flow

- User yêu cầu nhập cập nhật vào main và tải/cài như V7/V8. Cell0 chỉ chuẩn bị NumPy rồi restart; cell1 clone/fetch main và pip install requirements.txt. NumPy/SciPy/MarkupSafe/Jedi nằm trong root requirements. Không cài bộ GPU stack theo manager/detect trong notebook.
- Record commit thực tế và phiên bản thư viện vào run/cache identity; kiểm import model thật sau cài với traceback đầy đủ. Không coi CPU/AST checks là Colab GPU training.

## DEC-004 — FLUX.3 text conditioning và cached CFG

- Match BFL: right-pad token inputs, đặt padding position_ids=1 rõ ràng, không phụ thuộc mặc định Transformers.
- Cache positive/negative ở window chung khi TE còn loaded; sau unload không zero-pad embeddings vì DiT không có text mask. Các kiến trúc khác không có hook giữ đường cache cũ.
- New text cache identity gồm encoding revision/max_length/pad_multiple. Latent space không đổi. Không buộc sửa TAR ảnh/caption vì lỗi encoding này.
- Physical training batch1 với caption windows khác nhau; tăng effective batch qua accumulation. Không âm thầm đổi encoding khi batch cache khác chiều dài.
- Kiểm CPU không thay pretrained Base GPU A/B; không coi checkpoint đã học đạt hoặc buộc train lại chỉ từ source discrepancy.
