# Changelog

## 2026-10-19
- Probe Base A/B theo user dùng flowmatch25steps/CFG4 mặc định; thêm CLI overrides và ghi recipe thực vào REPORT. Core training, alpha6.93 và conditioning patch giữ nguyên.
- Sửa FLUX.3 Qwen padding position_ids và cache CFG window trước unload TE; đổi text-cache identity, giữ latent/TAR/scheduler/install contracts.14text/reduced-transformer +11reference +6accumulation +7scheduler CPU checks đạt; thêm6probe-runner checks và script Base A/B không training. Pretrained Base GPU quality test chưa chạy.

## 2026-06-23
- Bootstrapped lean project memory files for future project tasks.
- Merged `upstream/main` into fork sync branch and preserved fork scheduler changes.
- Fixed upstream Ideogram 4 prompt literal so Python compilation preserves backslash escapes.
- Added explicit scheduler step-unit normalization for phased schedules.

## 2026-10-17 — Đồng bộ upstream cho FLUX.3 Base
- Merge upstream82e6311, giữ phased scheduler/optimizer-step/resume của fork; sửa raw/list/stacked references và preview role order FLUX.3, thêm opt-in reference budget/Lanczos.7scheduler+11reference CPU tests PASS; không tải weights hoặc trainGPU.

## 2026-10-18 — Giữ gradient qua microbatch
- Sửa SDTrainer xóa gradient ở đầu mỗi call khi accumulation>1; chỉ zero đầu cycle và sau update.6actual-hook CPU regression tests đối chiếu gradient trung bình, nhiều cycle, accumulation1/2/3 và batch-list; không đổi schema hoặc legacy notebook pins.

- 2026-10-18: Merge upstream8cf43dd (6commits): raw/stacked FLUX3 reference fix reconciled with fork helper, park TE after caching under low_vram, Triton libdevice globals and MOSS updates. Root requirements now declare NumPy1.26.4/MarkupSafe floor/Jedi for legacy Colab install flow; no manager GPU-stack override.11reference+6accumulation+7scheduler CPUtests PASS.
