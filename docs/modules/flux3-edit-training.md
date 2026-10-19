# FLUX.3 paired edit training trong fork

## Phạm vi
- Hỗ trợ Base upstream, giữ scheduler tùy chỉnh của fork và sửa đường reference cho paired edit training.
- Không tự sinh prompt bằng upsampler: dataset phải chứa caption đã chuẩn bị đúng recipe. Không dùng notebook FLUX.2 bằng cách chỉ đổi model name.

## Reference contract
- `Flux3Model.use_raw_control_images=True`, `has_multiple_control_images=True`: loader giữ kích thước/aspect độc lập, trả list đúng thứ tự cho nhiều control.
- Train nhận cả raw per-sample lists và legacy BCHW/BNCHW; không bỏ ảnh thứ hai khi đọc stacked controls.
- Các slot preview `ctrl_img_1/2/3` giữ thứ tự và cả repeated paths có chủ đích. `ctrl_img` chỉ là alias ảnh đầu, không là ảnh thứ tư.
- Batch vật lý1 là lựa chọn an toàn cho paired controls khác shape; tăng effective batch qua accumulation. Mọi sample trong một batch phải có cùng số reference và grid tương thích.
- Raw controls không tự nhận target crop/flip. Với destination aligned, chuẩn bị paired target/reference views cùng geometry trước loader; mặc định không flip riêng target. Donor/global có aspect riêng phải được giữ riêng.

## Budget reference độc lập (opt-in)

```yaml
model:
  arch: flux3
  name_or_path: black-forest-labs/flux-3-image-klein-base
  model_kwargs:
    reference_megapixels: 1.0
    reference_resize_method: lanczos
```

- `reference_megapixels` tính theo1024²pixels, mỗi ảnh; giữ aspect, snap16. Có override này thì budget độc lập target area và ưu tiên hơn `match_target_res`.
- `lanczos` dùng RGB8 PIL quantize/resize giống đường ComfyUI workflowV1. `bilinear`/`bicubic` vẫn có sẵn.
- Nếu không khai báo override, giữ sizing upstream: `match_target_res=True` mặc định area-match target; False chỉ cap ảnh lớn hơn target, không tự cap1MP. Interpolation mặc định vẫn bilinear.
- Thay budget/interpolation sau training là đổi recipe; ghi rõ khi A/B. VAE latent space không đổi, không bump version target-latent cache vì sửa reference path.

## Scheduler fork
- `warmup_then_phased_cosine`, warmup/restarts và aliases của fork được giữ.
- Scheduler chỉ step khi optimizer update. Notebook hiện tại dùng `phase_step_unit: training_steps` để quy warmup/phase theo displayed steps; params được deep-copy trước conversion theo accumulation.
- Không thay default của configs cũ hoặc pin commit trong notebook legacy. Resume index tiếp tục dùng optimizer-update units.

## Gradient accumulation
- Khi `gradient_accumulation_steps>1`, SDTrainer giữ gradient qua các lần gọi microbatch và chỉ xóa ở đầu cycle hoặc sau optimizer update. Bản trước đã xóa ở mỗi lần gọi, làm mất gradient của các microbatch trước dù loss được chia đúng accumulation.
- Counter của Base bắt đầu1, chuyển0 tại lượt update rồi trở về1. Accumulation1 giữ hành vi cũ; batch-list accumulation và repeated-call accumulation chỉ normalize loss một lần theo số microbatch thực.
- `testing/test_sd_trainer_accumulation.py` chạy actual hook bằng tensor/SGD CPU với các target khác nhau, đối chiếu mean-gradient, không chỉ đếm call.

## Kiểm chứng
- Text-conditioning revision `flux3_qwen3vl_padpos1_v2`: right-padding rõ ràng; Qwen token đệm dùng position_ids=1 như BFL. Text cache namespace chứa revision/max_length/pad_multiple, tách latent cache vốn không đổi.
- Preview CFG encode prompt ngắn lại ở window dài trước khi unload TE qua hook `prepare_sample_prompt_pair`; cached/live đều giữ real pad-token states. Không zero-pad context để ghép batch. Với cached training prompts khác window, dùng physical batch1 + accumulation; batch nhiều window khác nhau báo lỗi rõ.
- `testing/test_flux3_text_conditioning.py`:14CPU tests gồm exact SDTrainer cache→FakeTE→generation, cache identity, mismatched cache rejection, tiny real Qwen3VLTextModel (đã chạy Transformers5.5.3 cách ly), real reduced FLUX.3 transformer + actual LoRAModule optimizer/checkpointing parity. Đây chưa là pretrained Base BF16 GPU/quality validation.
- `python testing/test_scheduler_step_units.py`:7CPU tests cho phase conversion, curve, param-group ratio, resume và merged trainer call site.
- `python testing/test_flux3_reference_contract.py`:11CPU tests dùng actual loader/model methods và packers; kiểm independent shapes,2/3refs, legacy stacked input, role IDs, t0 conditioning, gradient qua recording transformer, budget/Lanczos và prefix roundtrip.
- Tests reference dùng AST loading để tách dependency/model weights không liên quan; không phải end-to-end training. Chưa tải weights hoặc kiểm GPU optimizer step/VRAM/quality. Prefix roundtrip cũng chưa chứng minh mọi exporter/checkpoint load vào ComfyUI.

## Môi trường
- Colab theo luồng V7/V8: cell0 chuẩn bị NumPy1.26.4 rồi manualrestart; cell1 clone/fetch `pearhoang/ai-toolkit` main và `pip install -r requirements.txt`. Requirements root giữ NumPy/SciPy và MarkupSafe/Jedi; FLUX.3 Transformers/Diffusers đã nằm trong requirements_base. Không ép bộ Torch/CUDA của manager vào runtime Colab, không chặn vì dependency của gói ngoài training. Trainer thực vẫn báo lỗi import/CUDA nếu không tương thích; log phải hiện stdout/stderr đầy đủ.
- Sync upstream8cf43dd (2026-10-18): giữ helper raw/stacked controls của fork (bao phủ upstream9d408eb), thêm park text encoder sau cache khi low_vram và libdevice global cho Triton cũ. Scheduler/gradient accumulation/reference budget giữ contract cũ.
- Requirements theo upstream mới, gồm Transformers5.5.3 và Diffusers git pin. Dùng môi trường training riêng/Colab; không nâng cấp embedded Python của ComfyUI/PhotoCraft để phục vụ trainer.
- Trước run lớn: validate paired geometry/mask, caption recipe, số/order refs thực sự tới transformer, cache disk, sample/checkpoint/ZIP marker và exporter LoRA.
