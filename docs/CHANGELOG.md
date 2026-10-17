# Changelog

## 2026-06-23
- Bootstrapped lean project memory files for future project tasks.
- Merged `upstream/main` into fork sync branch and preserved fork scheduler changes.
- Fixed upstream Ideogram 4 prompt literal so Python compilation preserves backslash escapes.
- Added explicit scheduler step-unit normalization for phased schedules.

## 2026-10-17 — Đồng bộ upstream cho FLUX.3 Base
- Merge upstream82e6311, giữ phased scheduler/optimizer-step/resume của fork; sửa raw/list/stacked references và preview role order FLUX.3, thêm opt-in reference budget/Lanczos.7scheduler+11reference CPU tests PASS; không tải weights hoặc trainGPU.

## 2026-10-18 — Giữ gradient qua microbatch
- Sửa SDTrainer xóa gradient ở đầu mỗi call khi accumulation>1; chỉ zero đầu cycle và sau update.6actual-hook CPU regression tests đối chiếu gradient trung bình, nhiều cycle, accumulation1/2/3 và batch-list; không đổi schema hoặc legacy notebook pins.
