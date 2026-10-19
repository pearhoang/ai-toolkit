"""CPU regression tests for the real FLUX.3 encode/cache/preview functions.

AST isolation avoids installing the full trainer. Small fixtures do not replace
a pretrained Base BF16 quality/optimizer test.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
import importlib.util
import json
import math
from pathlib import Path
import types
import typing
import unittest
import weakref

import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "extensions_built_in/diffusion_models/flux3"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_functions(path, names, env, owner=None):
    parsed = ast.parse(path.read_text(encoding="utf-8-sig"))
    body = parsed.body
    if owner:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    selected = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), str(path), "exec"), env)
    return {name: env[name] for name in names}


class FakeTextEncoder:
    pass


class Tokenizer:
    # Deliberately left by default: encode must explicitly request right padding.
    padding_side = "left"

    def __call__(self, text, **kwargs):
        real = min(len(text) + 3, kwargs["max_length"])
        length = kwargs["max_length"] if kwargs.get("padding") == "max_length" else real
        ids = torch.zeros((1, length), dtype=torch.long)
        mask = torch.zeros_like(ids)
        start = 0 if kwargs.get("padding_side", self.padding_side) == "right" else length - real
        ids[:, start:start + real] = 3
        mask[:, start:start + real] = 1
        return {"input_ids": ids, "attention_mask": mask}


class RecordingEncoder:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def to(self, *args, **kwargs):
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        # Nonzero pad states deliberately distinguish Qwen encoding from F.pad.
        values = kwargs["position_ids"][0].float()[..., None].repeat(1, 1, 2) + 1
        return types.SimpleNamespace(hidden_states=tuple(values + i for i in range(33)))


def paths():
    embeds = load_module(ROOT / "toolkit/advanced_prompt_embeds.py", "test_flux3_embeds").AdvancedPromptEmbeds
    positions = load_module(SRC / "src/positional.py", "test_flux3_positions")
    parsed = ast.parse((SRC / "src/text_encoder.py").read_text(encoding="utf-8"))
    version = next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "TEXT_EMBEDDING_VERSION" for t in n.targets))
    env = dict(torch=torch, math=math, List=typing.List, Optional=typing.Optional, rearrange=rearrange,
               SELECT_LAYERS=[4, 8, 12, 16, 20, 24, 28, 32], DEFAULT_MAX_TOKENS=8192,
               DEFAULT_PAD_MULTIPLE=80, TEXT_EMBEDDING_VERSION=version, AdvancedPromptEmbeds=embeds,
               FakeTextEncoder=FakeTextEncoder, prc_txt=positions.prc_txt)
    load_functions(SRC / "src/text_encoder.py", ["padded_token_length", "encode_flux3_prompt"], env)
    load_functions(SRC / "src/pipeline.py", ["build_text_conditioning"], env)
    names = ["_encode_at_length", "get_prompt_embeds", "prepare_sample_prompt_pair", "generate_single_image",
             "get_text_embedding_space_version", "get_bucket_divisibility", "_preview_control_paths"]
    methods = load_functions(SRC / "flux3.py", names, env, "Flux3Model")
    cls = type("Model", (), methods)
    cache_env = dict(torch=torch, print_acc=lambda *a: None,
                     GenerateImageConfig=lambda **kw: types.SimpleNamespace(**kw),
                     os=__import__("os"))
    cache_method = load_functions(ROOT / "extensions_built_in/sd_trainer/SDTrainer.py",
                                 ["cache_sample_prompts"], cache_env, "SDTrainer")["cache_sample_prompts"]
    return cls, env, cache_method


class TextConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Model, cls.env, cache = paths()
        cls.cache = staticmethod(cache)

    def model(self):
        m = self.Model()
        m.processor = types.SimpleNamespace(tokenizer=Tokenizer(), apply_chat_template=lambda messages, **kw: messages[0]["content"])
        m.text_encoder = RecordingEncoder()
        m.torch_dtype = torch.float32
        m.device_torch = torch.device("cpu")
        m.max_text_length = 8192
        m.text_pad_multiple = 80
        m.model = types.SimpleNamespace(device=torch.device("cpu"), to=lambda *a: None)
        m.model_config = types.SimpleNamespace(model_kwargs={})
        return m

    def config(self, prompt="p" * 100, negative=""):
        return types.SimpleNamespace(prompt=prompt, negative_prompt=negative, width=64, height=64,
            guidance_scale=3.5, num_inference_steps=2, latents=None,
            ctrl_img=None, ctrl_img_1=None, ctrl_img_2=None, ctrl_img_3=None)

    def test_real_encoder_function_forces_right_padding_and_pad_position_one(self):
        m = self.model()
        out = m.get_prompt_embeds("hi")
        call = m.text_encoder.calls[-1]
        self.assertEqual(tuple(out.text_embeds[0].shape), (80, 16))
        self.assertEqual(call["position_ids"].shape, (4, 1, 80))
        self.assertTrue(torch.equal(call["attention_mask"][0, :5], torch.ones(5, dtype=torch.long)))
        self.assertTrue(torch.all(call["position_ids"][:, :, 5:] == 1))
        self.assertTrue(torch.equal(call["position_ids"][0, 0, :5], torch.arange(5)))

    def test_prompt_batch_common_window_preserves_real_encoded_pad_states(self):
        m = self.model()
        out = m.get_prompt_embeds(["hi", "p" * 100])
        self.assertEqual([len(t) for t in out.text_embeds], [160, 160])
        self.assertGreater(torch.count_nonzero(out.text_embeds[0][80:]).item(), 0)

    def test_cache_key_invalidates_old_text_only_and_tracks_window_config(self):
        m = self.model()
        a = m.get_text_embedding_space_version()
        self.assertNotEqual(a, "flux3")
        self.assertIn("padpos1_v2", a)
        m.max_text_length = 4096
        self.assertNotEqual(a, m.get_text_embedding_space_version())
        m.max_text_length = 8192
        m.text_pad_multiple = 160
        self.assertNotEqual(a, m.get_text_embedding_space_version())
        # Actual DTO key builder must retain the new model version.
        env = dict(OrderedDict=OrderedDict)
        info = load_functions(ROOT / "toolkit/dataloader_mixins.py", ["get_text_embedding_info_dict"], env,
            "TextEmbeddingFileItemDTOMixin")["get_text_embedding_info_dict"]
        item = types.SimpleNamespace(caption="p", text_embedding_space_version=a, text_embedding_version=1,
                                     encode_control_in_text_embeddings=False, encode_first_frame_in_text_embeddings=False)
        key = info(item)
        self.assertEqual(key["text_embedding_space_version"], a)
        old = dict(key, text_embedding_space_version="flux3")
        self.assertNotEqual(json.dumps(old, sort_keys=True), json.dumps(key, sort_keys=True))

    def test_positive_longer_reencodes_actual_empty_negative(self):
        m = self.model(); c = self.config()
        pos = m.get_prompt_embeds(c.prompt); neg = m.get_prompt_embeds("")
        pos, neg = m.prepare_sample_prompt_pair(c, pos, neg)
        self.assertEqual([len(pos.text_embeds[0]), len(neg.text_embeds[0])], [160, 160])
        self.assertGreater(torch.count_nonzero(neg.text_embeds[0][80:]).item(), 0)

    def test_negative_longer_keeps_its_text_and_reencodes_positive(self):
        m = self.model(); c = self.config("p", "n" * 170)
        pos = m.get_prompt_embeds(c.prompt); neg = m.get_prompt_embeds(c.negative_prompt)
        original = neg.text_embeds[0].clone()
        pos, neg = m.prepare_sample_prompt_pair(c, pos, neg)
        self.assertEqual(len(pos.text_embeds[0]), 240)
        self.assertTrue(torch.equal(neg.text_embeds[0], original))

    def test_unloaded_equal_pair_works_without_encoder(self):
        m = self.model(); c = self.config()
        pos, neg = m.prepare_sample_prompt_pair(c, m.get_prompt_embeds(c.prompt), m.get_prompt_embeds(""))
        m.text_encoder = FakeTextEncoder()
        p, n = m.prepare_sample_prompt_pair(c, pos, neg)
        self.assertIs(p, pos); self.assertIs(n, neg)

    def test_old_unloaded_mismatched_pair_fails_with_actionable_error(self):
        m = self.model(); c = self.config()
        pos, neg = m.get_prompt_embeds(c.prompt), m.get_prompt_embeds("")
        m.text_encoder = FakeTextEncoder()
        with self.assertRaisesRegex(RuntimeError, "Rebuild sample"):
            m.prepare_sample_prompt_pair(c, pos, neg)

    def test_packer_refuses_silent_zero_padding(self):
        with self.assertRaisesRegex(ValueError, "physical batch size 1"):
            self.env["build_text_conditioning"]([torch.ones(80, 4), torch.ones(160, 4)], "cpu", torch.float32)

    def test_packer_equal_windows_preserves_pad_data(self):
        tensors = [torch.ones(160, 4), torch.ones(160, 4) * 2]
        ctx, ids, times = self.env["build_text_conditioning"](tensors, "cpu", torch.float32)
        self.assertTrue(torch.equal(ctx[0], tensors[0]))
        self.assertEqual(tuple(ids.shape), (2, 160, 4))
        self.assertEqual(torch.count_nonzero(times).item(), 0)

    def test_real_trainer_cache_hook_then_unloaded_real_generation_hook(self):
        m = self.model(); m.encode_prompt = m.get_prompt_embeds
        m.encode_control_in_text_embeddings = False
        sample = types.SimpleNamespace(neg="", ctrl_img=None, ctrl_img_1=None, ctrl_img_2=None, ctrl_img_3=None)
        trainer = types.SimpleNamespace(train_config=types.SimpleNamespace(disable_sampling=False),
            sample_config=types.SimpleNamespace(samples=[sample], prompts=["p" * 100]),
            sd=m, trigger_word=None, save_root="unused")
        self.cache(trainer)
        entry = m.sample_prompts_cache[0]
        self.assertEqual([len(entry[k].text_embeds[0]) for k in ("conditional", "unconditional")], [160, 160])
        self.assertTrue(all(entry[k].text_embeds[0].device.type == "cpu" for k in entry))
        m.text_encoder = FakeTextEncoder()
        captured = {}
        def pipeline(**kwargs):
            captured.update(kwargs)
            return ["sample sentinel"]
        result = m.generate_single_image(pipeline, self.config(), entry["conditional"], entry["unconditional"], None, {})
        self.assertEqual(result, "sample sentinel")
        self.assertGreater(torch.count_nonzero(captured["unconditional_embeds"].text_embeds[0][80:]).item(), 0)

    def test_none_negative_keeps_non_cfg_path(self):
        m = self.model()
        pos = m.get_prompt_embeds("hi")
        self.assertEqual(m.prepare_sample_prompt_pair(self.config(), pos, None), (pos, None))

    def test_actual_tiny_qwen_accepts_positions_and_pad_states_change(self):
        # Current local Transformers, not a claim of pretrained Qwen/Base QA.
        from transformers import Qwen3VLTextConfig, Qwen3VLTextModel
        torch.manual_seed(711)
        config = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]})
        tiny = Qwen3VLTextModel(config).eval()
        ids = torch.tensor([[3, 4, 5, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
        with torch.no_grad():
            fixed = tiny(input_ids=ids, attention_mask=mask, position_ids=positions[None].expand(4, -1, -1), use_cache=False).last_hidden_state
            default = tiny(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        self.assertTrue(torch.isfinite(fixed).all())
        self.assertTrue(torch.allclose(fixed[:, :3], default[:, :3], atol=1e-6))
        self.assertGreater((fixed[:, 3:] - default[:, 3:]).abs().max().item(), 1e-6)

    def test_actual_small_transformer_lora_optimizer_checkpointing_parity(self):
        # Real transformer and real LoRAModule/mixin arithmetic, reduced sizes.
        # Only loader/quantizer dependencies are isolated, not attention/LoRA.
        import sys
        module = types.ModuleType("test_flux3_small_transformer")
        sys.modules[module.__name__] = module
        module.OstrisModelMixin = type("OstrisModelMixin", (), {})
        module.COMFY_REPO = "unused"
        module.COMFY_TRANSFORMER_FILES = []
        tree = ast.parse((SRC / "src/transformer.py").read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom)
                     and (n.module == "toolkit.models.v2._mixin" or n.level))]
        exec(compile(tree, "actual_flux3_transformer", "exec"), module.__dict__)
        env = dict(torch=torch, nn=torch.nn, weakref=weakref, math=math, Union=typing.Union,
                   List=typing.List, Optional=typing.Optional, CONV_MODULES=["Conv2d"],
                   QTensor=type("QTensor", (), {}), ExtractableModuleMixin=type("ExtractableModuleMixin", (), {}))
        load_functions(ROOT / "toolkit/network_mixins.py", ["broadcast_and_multiply"], env)
        methods = load_functions(ROOT / "toolkit/network_mixins.py",
            ["__init__", "_set_runtime_scale", "_call_forward", "forward"], env, "ToolkitModuleMixin")
        env["ToolkitModuleMixin"] = type("ToolkitModuleMixin", (), methods)
        lora_tree = ast.parse((ROOT / "toolkit/lora_special.py").read_text(encoding="utf-8"))
        lora_class = next(n for n in lora_tree.body if isinstance(n, ast.ClassDef) and n.name == "LoRAModule")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__",
            names=[ast.alias(name="annotations")], level=0), lora_class], type_ignores=[])), "actual_lora", "exec"), env)
        Model, _env, _cache = paths()
        hook_env = dict(self.env)
        pos = load_module(SRC / "src/positional.py", "test_flux3_small_positions")
        hook_env.update(batched_prc_img=pos.batched_prc_img, IMAGE_ROPE_SCALE=0.5)
        load_functions(SRC / "src/pipeline.py", ["pack_image_latents"], hook_env)
        hook = load_functions(SRC / "flux3.py", ["get_noise_prediction"], hook_env, "Flux3Model")["get_noise_prediction"]

        def run(checkpointing):
            torch.manual_seed(914)
            params = module.Flux3TransformerParams(hidden_size=32, num_heads=2,
                axes_dim=[4, 4, 4, 4], context_in_dim=16, depth=1, depth_single_blocks=1)
            transformer = module.Flux3Transformer(params).requires_grad_(False)
            transformer.enable_gradient_checkpointing(checkpointing)
            class Network:
                pass
            net = Network()
            net.network_type = "lora"; net.is_lorm = False; net.is_active = True
            net.is_merged_in = False; net._multiplier = 1.0; net.torch_multiplier = torch.ones(1)
            adapters = []
            for name, layer in list(transformer.named_modules()):
                if isinstance(layer, torch.nn.Linear) and name.startswith(("content_mode_blocks.", "txt_mode_blocks.", "single_blocks.")):
                    adapter = env["LoRAModule"](name, layer, lora_dim=2, alpha=2, network=net)
                    adapter.apply_to(); adapters.append(adapter)
            trainable = [p for a in adapters for p in a.parameters()]
            optimizer = torch.optim.SGD(trainable, lr=0.02)
            before_base = {k: v.clone() for k, v in transformer.state_dict().items()}
            m = types.SimpleNamespace(device_torch=torch.device("cpu"), torch_dtype=torch.float32,
                model=transformer, transformer=transformer,
                _control_images_for_batch=lambda *a: [[torch.ones(1, 3, 32, 32)]],
                _fit_reference=lambda ctrl, budget: ctrl,
                encode_images=lambda ctrl, **kw: torch.ones(1, 128, 2, 2) * 0.4)
            # Reuse actual reference packing too.
            hook_env.update(prc_img=pos.prc_img, I2I_COND_L=-1)
            load_functions(SRC / "src/pipeline.py", ["pack_i2i_image_condition"], hook_env)
            latent = torch.randn(1, 128, 2, 3)
            embeds = types.SimpleNamespace(text_embeds=[torch.randn(80, 16)])
            losses = []
            first_up_grads = None
            for step in range(2):
                optimizer.zero_grad()
                prediction = hook(m, latent, torch.tensor([500.]), embeds, object())
                loss = prediction.square().mean()
                loss.backward(); losses.append(loss.item())
                grads = [a.lora_up.weight.grad for a in adapters]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                self.assertTrue(all(g.abs().sum() > 0 for g in grads))
                if first_up_grads is None:
                    first_up_grads = [g.clone() for g in grads]
                optimizer.step()
            self.assertTrue(all(torch.equal(v, transformer.state_dict()[k]) for k, v in before_base.items()))
            self.assertTrue(all(a.lora_up.weight.abs().sum() > 0 for a in adapters))
            self.assertTrue(all(a.lora_down.weight.grad.abs().sum() > 0 for a in adapters))
            return losses, first_up_grads

        plain, plain_grads = run(False)
        checkpointed, checkpointed_grads = run(True)
        self.assertTrue(torch.allclose(torch.tensor(plain), torch.tensor(checkpointed), atol=1e-6))
        self.assertTrue(all(torch.allclose(a, b, atol=1e-6) for a, b in zip(plain_grads, checkpointed_grads)))

    def test_non_cfg_generation_does_not_require_matching_negative_cache(self):
        m = self.model(); c = self.config(); c.guidance_scale = 1.0
        pos, neg = m.get_prompt_embeds(c.prompt), m.get_prompt_embeds("")
        m.text_encoder = FakeTextEncoder()
        self.assertEqual(m.generate_single_image(lambda **kw: ["ok"], c, pos, neg, None, {}), "ok")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
