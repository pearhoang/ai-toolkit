"""Controlled FLUX.3 Base preview A/B; no optimizer, training or base merging.

Uses a verified local probe bundle and one native checkpoint. Four images per
case: old cached conditioning / fixed conditioning, each LoRA OFF / ON.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import List, Optional

import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file

# Running as `python scripts/...py` must resolve toolkit from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError("Probe bundle file failed SHA/path validation: " + name)
    return manifest


def load_legacy(root):
    tree = ast.parse((root / "legacy/text_encoder.py").read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("padded_token_length", "encode_flux3_prompt")]
    env = dict(torch=torch, math=math, rearrange=rearrange, List=List, Optional=Optional,
               DEFAULT_MAX_TOKENS=8192, DEFAULT_PAD_MULTIPLE=80, SELECT_LAYERS=[4, 8, 12, 16, 20, 24, 28, 32])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "frozen_legacy_text_encoder", "exec"), env)
    name = "extensions_built_in.diffusion_models.flux3.src._legacy_ab_pipeline"
    spec = importlib.util.spec_from_file_location(name, root / "legacy/pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return env["encode_flux3_prompt"], module.Flux3Pipeline


def attach_native_lora(transformer, checkpoint, alpha):
    """Attach actual trainer LoRAModule forwards without rounding base weights."""
    from toolkit.lora_special import LoRAModule
    weights = load_file(str(checkpoint), device="cpu")
    prefixes = {key[:-len(".lora_A.weight")] for key in weights if key.endswith(".lora_A.weight")}
    expected = {p + suffix for p in prefixes for suffix in (".lora_A.weight", ".lora_B.weight")}
    if not prefixes or set(weights) != expected:
        raise ValueError("Expected native FLUX.3 lora_A/lora_B pairs only; no keys may be ignored")
    network = types.SimpleNamespace(network_type="lora", is_lorm=False, is_active=True,
        is_merged_in=False, _multiplier=0.0, torch_multiplier=torch.zeros(1, device=transformer.device))
    # LoRAModule holds a weak reference; SimpleNamespace cannot be weak-ref'd.
    class Network:
        pass
    net = Network(); net.__dict__.update(network.__dict__)
    modules = dict(transformer.named_modules())
    plans = []
    for prefix in sorted(prefixes):
        name = prefix.removeprefix("diffusion_model.").removeprefix("transformer.")
        layer = modules.get(name)
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError("Checkpoint layer not found as Linear: " + prefix)
        down, up = weights[prefix + ".lora_A.weight"], weights[prefix + ".lora_B.weight"]
        if down.ndim != 2 or up.ndim != 2 or down.shape[1] != layer.in_features or up.shape != (layer.out_features, down.shape[0]):
            raise ValueError("LoRA shape mismatch: " + prefix)
        if not torch.isfinite(down).all() or not torch.isfinite(up).all():
            raise ValueError("Nonfinite LoRA: " + prefix)
        plans.append((name, layer, down, up))
    adapters = []
    for name, layer, down, up in plans:
        adapter = LoRAModule(name, layer, lora_dim=down.shape[0], alpha=alpha, network=net)
        adapter.to(device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            adapter.lora_down.weight.copy_(down); adapter.lora_up.weight.copy_(up)
        adapter.eval().requires_grad_(False)
        adapter.apply_to(); adapters.append(adapter)
    return net, adapters


def run(args):
    if not 1 <= args.steps <= 200 or not math.isfinite(args.guidance) or args.guidance <= 1:
        raise ValueError("Use steps in [1, 200] and finite guidance > 1 for this CFG comparison")
    from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
    from toolkit.config_modules import ModelConfig
    from toolkit.unloader import unload_text_encoder
    from extensions_built_in.diffusion_models.flux3.flux3 import Flux3Model
    from extensions_built_in.diffusion_models.flux3.src.text_encoder import TEXT_EMBEDDING_VERSION

    root = Path(args.bundle).resolve(); manifest = verified_bundle(root)
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint Drive chưa tìm thấy: " + str(checkpoint))
    if args.case < 0 or args.case >= len(manifest["cases"]):
        raise ValueError("Invalid case index")
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 60 * 1024**3:
        raise RuntimeError("Use A10080GB / RTX600096GB for the full Base BF16 comparison")
    case = manifest["cases"][args.case]
    recipe = dict(manifest["model"])
    # BaseSDTrainProcess normally injects this from train.dtype before creating
    # ModelConfig. Reproduce that here so TE/VAE do not default to float16.
    recipe["dtype"] = "bf16"
    cfg = ModelConfig(**recipe)
    if cfg.arch != "flux3" or cfg.quantize or cfg.quantize_te:
        raise ValueError("This experiment requires unquantized FLUX.3 Base + text encoder")
    legacy_encode, LegacyPipeline = load_legacy(root)
    model = Flux3Model("cuda:0", cfg, dtype="bf16")
    model.load_model(); model.model.eval().requires_grad_(False)
    print("Hashing loaded Base/VAE files for the A/B record...", flush=True)
    weight_proof = {name: sha(model._package_file(name)) for name in ("model.safetensors", "ae.safetensors")}
    if model.model.dtype != torch.bfloat16:
        raise RuntimeError("Base transformer is not BF16")
    if any("ref_time_in" in name for name in model.model.state_dict()):
        raise RuntimeError("Step-distilled transformer rejected; use Base")
    positive, negative = case["prompt"], case["negative_prompt"]
    def old_embed(text):
        values = legacy_encode(model.text_encoder, model.processor, [text],
            max_length=model.max_text_length, dtype=model.torch_dtype, pad_multiple=model.text_pad_multiple)
        return AdvancedPromptEmbeds(text_embeds=[values[0]]).to("cpu")
    variants = {"legacy_cached": (old_embed(positive), old_embed(negative))}
    pos, neg = model.get_prompt_embeds(positive), model.get_prompt_embeds(negative)
    gen_config = types.SimpleNamespace(prompt=positive, negative_prompt=negative)
    pos, neg = model.prepare_sample_prompt_pair(gen_config, pos, neg)
    variants["fixed_cached"] = (pos.to("cpu"), neg.to("cpu"))
    parity_pos = model._encode_at_length(positive, len(pos.text_embeds[0])).to("cpu")
    parity_neg = model._encode_at_length(negative, len(neg.text_embeds[0])).to("cpu")
    if not torch.equal(pos.to("cpu").text_embeds[0], parity_pos.text_embeds[0]) or not torch.equal(neg.to("cpu").text_embeds[0], parity_neg.text_embeds[0]):
        raise RuntimeError("Live/cached pair parity failed")
    from torchvision.transforms.functional import to_tensor
    reference = to_tensor(Image.open(root / case["reference"]).convert("RGB")).unsqueeze(0)
    reference = model._fit_reference(reference, case["width"] * case["height"])
    initial_windows = {key: [len(p.text_embeds[0]), len(n.text_embeds[0])] for key, (p, n) in variants.items()}
    del pos, neg, parity_pos, parity_neg
    unload_text_encoder(model); gc.collect(); torch.cuda.empty_cache()
    net, adapters = attach_native_lora(model.model, checkpoint, manifest["lora_alpha"])
    report = dict(status="running", no_training=True, no_base_merge=True,
        checkpoint_sha256=sha(checkpoint), checkpoint=str(checkpoint), case=case,
        text_encoding_version=TEXT_EMBEDDING_VERSION, windows=initial_windows,
        adapters=len(adapters), sampler="flowmatch_euler", steps=args.steps, guidance=args.guidance, alpha=6.93,
        model=recipe, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        base_vae_sha256=weight_proof,
        toolkit_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), results=[])
    for variant, (p, n) in variants.items():
        pipeline = LegacyPipeline(model) if variant == "legacy_cached" else model.get_generation_pipeline()
        for multiplier in (0.0, 1.0):
            net._multiplier = multiplier; net.torch_multiplier.fill_(multiplier)
            generator = torch.Generator(device="cpu").manual_seed(case["seed"])
            start = time.time()
            images = pipeline(conditional_embeds=p, unconditional_embeds=n,
                height=case["height"], width=case["width"], num_inference_steps=args.steps,
                guidance_scale=args.guidance, generator=generator, ctrl_imgs=[reference], alpha=6.93)
            path = out / f"{variant}_lora_{'ON' if multiplier else 'OFF'}.png"
            images[0].save(path)
            report["results"].append(dict(file=path.name, sha256=sha(path), elapsed_seconds=time.time()-start))
            (out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(path, "seconds", round(time.time()-start, 1), flush=True)
    report["status"] = "completed_images_pending_visual_review"
    (out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("No training performed. Outputs:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", type=int, default=2)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=4.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
