"""Same-Base caption-only comparison; no training or live prompt upsampler."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.flux3_conditioning_ab import attach_native_lora, sha, verified_bundle

REPO = Path(__file__).resolve().parents[1]


def validate_probe(root, manifest, config, checkpoint):
    case = next((c for c in manifest["cases"] if c["sample_id"] == config["sample_id"]), None)
    if case is None:
        raise ValueError("Frozen PUP belongs to a different source case")
    if sha(root / case["reference"]) != config["reference_sha256"]:
        raise ValueError("Reference does not match frozen PUP source")
    for key, text in [("notebook_prompt", case["prompt"]), ("pup_prompt", config["pup_prompt"])]:
        if hashlib.sha256(text.encode()).hexdigest() != config[key + "_sha256"]:
            raise ValueError("Prompt changed: " + key)
    if sha(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Use the same General step2000 checkpoint as the original Base A/B")
    return case


def encode_caption_pairs(model, case, pup_prompt):
    """One common real Qwen pad window for both captions, before TE unload."""
    texts = dict(notebook=case["prompt"], pup=pup_prompt)
    first = {key: model.get_prompt_embeds(text) for key, text in texts.items()}
    negative = model.get_prompt_embeds(case["negative_prompt"])
    window = max([len(x.text_embeds[0]) for x in first.values()] + [len(negative.text_embeds[0])])
    # Never trim or pad hidden vectors. Re-encode actual tokens where needed.
    negative = model._encode_at_length(case["negative_prompt"], window).to("cpu")
    pairs = {}
    for key, text in texts.items():
        positive = model._encode_at_length(text, window).to("cpu")
        pairs[key] = (positive, negative)
    return texts, pairs, window


def generate_caption_matrix(pipeline, net, pairs, case, seeds, steps, guidance, reference, out, report):
    """All image-generation inputs except caption/LoRA multiplier are shared."""
    for seed in seeds:
        for caption, (positive, negative) in pairs.items():
            for multiplier in (0.0, 1.0):
                net._multiplier = multiplier
                net.torch_multiplier.fill_(multiplier)
                start = time.monotonic()
                generator = torch.Generator(device="cpu").manual_seed(seed)
                images = pipeline(conditional_embeds=positive, unconditional_embeds=negative,
                    height=case["height"], width=case["width"], num_inference_steps=steps,
                    guidance_scale=guidance, generator=generator, ctrl_imgs=[reference], alpha=6.93)
                path = out / f"seed{seed}_{caption}_lora_{'ON' if multiplier else 'OFF'}.png"
                images[0].save(path)
                row = dict(file=path.name, caption=caption, seed=seed, lora_multiplier=multiplier,
                    sha256=sha(path), elapsed_seconds=time.monotonic()-start)
                report["results"].append(row)
                (out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(row["file"], round(row["elapsed_seconds"], 1), "seconds", flush=True)


def annotate_results(root, case, out, report):
    ref = np.asarray(Image.open(root / case["reference"]).convert("RGB")).astype(np.int16)
    unknown = np.ones(ref.shape[:2], dtype=bool)
    left, top, right, bottom = case["known_bbox"]
    unknown[top:bottom, left:right] = False
    colors, counts = np.unique(ref[unknown], axis=0, return_counts=True)
    prefill = colors[counts.argmax()]
    for row in report["results"]:
        image = np.asarray(Image.open(out / row["file"]).convert("RGB")).astype(np.int16)
        if image.shape != ref.shape:
            raise ValueError("Unexpected output image shape")
        row["unknown_fraction_within8_of_prefill"] = float((np.abs(image[unknown]-prefill).max(-1) <= 8).mean())
        row["unknown_MAE_vs_reference"] = float(np.abs(image-ref)[unknown].mean())
    report["prefill_RGB"] = prefill.tolist()
    report["unknown_fraction"] = float(unknown.mean())
    for seed in report["seeds"]:
        board = Image.new("RGB", (1600, 1140), "white"); draw = ImageDraw.Draw(board)
        for i, caption in enumerate(("notebook", "pup")):
            for j, multiplier in enumerate((0.0, 1.0)):
                row = next(r for r in report["results"] if r["seed"] == seed and r["caption"] == caption and r["lora_multiplier"] == multiplier)
                image = Image.open(out / row["file"]).convert("RGB"); image.thumbnail((800, 533))
                x, y = j*800, i*570
                draw.text((x+10, y+8), row["file"], fill="black")
                board.paste(image, (x, y+32))
        board.save(out / f"seed{seed}_CAPTION_AB.jpg", quality=95)


def run(args):
    if not 1 <= args.steps <= 200 or not math.isfinite(args.guidance) or args.guidance <= 1:
        raise ValueError("Use steps1-200 and finite CFG>1")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**63 for s in args.seeds):
        raise ValueError("Use distinct nonnegative seeds below2**63")
    root = Path(args.bundle).resolve(); manifest = verified_bundle(root)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint).resolve()
    case = validate_probe(root, manifest, config, checkpoint)
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 60*1024**3:
        raise RuntimeError("Use the same A10080GB / RTX600096GB Base BF16 runtime")
    from toolkit.config_modules import ModelConfig
    from toolkit.unloader import unload_text_encoder
    from torchvision.transforms.functional import to_tensor
    from extensions_built_in.diffusion_models.flux3.flux3 import Flux3Model
    from extensions_built_in.diffusion_models.flux3.src.text_encoder import TEXT_EMBEDDING_VERSION
    recipe = dict(manifest["model"], dtype="bf16")
    cfg = ModelConfig(**recipe)
    if cfg.arch != "flux3" or cfg.quantize or cfg.quantize_te:
        raise ValueError("Unquantized Base and text encoder required")
    model = Flux3Model("cuda:0", cfg, dtype="bf16")
    model.load_model(); model.model.eval().requires_grad_(False)
    weights = {name: sha(model._package_file(name)) for name in ("model.safetensors", "ae.safetensors")}
    if weights["model.safetensors"] != config["base_sha256"] or weights["ae.safetensors"] != config["vae_sha256"]:
        raise ValueError("Base/VAE changed from previous cloud test")
    if model.model.dtype != torch.bfloat16 or any("ref_time_in" in n for n in model.model.state_dict()):
        raise ValueError("Expected BF16 Base, not step-distilled checkpoint")
    texts, pairs, window = encode_caption_pairs(model, case, config["pup_prompt"])
    reference = to_tensor(Image.open(root/case["reference"]).convert("RGB")).unsqueeze(0)
    reference = model._fit_reference(reference, case["width"]*case["height"])
    unload_text_encoder(model); gc.collect(); torch.cuda.empty_cache()
    net, adapters = attach_native_lora(model.model, checkpoint, config["lora_alpha"])
    if len(adapters) != 258:
        raise ValueError("General2000 adapter count changed")
    report = dict(status="running", comparison="notebook_caption_vs_frozen_PUP_same_Base",
        no_training=True, no_base_merge=True, no_new_upsampler_call=True,
        case=case, captions=texts, prompt_provenance=config, checkpoint_sha256=sha(checkpoint),
        base_vae_sha256=weights, model=recipe, seeds=args.seeds, steps=args.steps, guidance=args.guidance,
        alpha=6.93, sampler="flowmatch_euler", shared_text_window=window,
        windows={key:[len(p.text_embeds[0]),len(n.text_embeds[0])] for key,(p,n) in pairs.items()},
        text_encoding_version=TEXT_EMBEDDING_VERSION, adapters=len(adapters),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        toolkit_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),results=[],
        limits=["Two seeds of one difficult four-border case are not a general success rate",
                "Common encoded window may differ from previous caption's native window; compare within this experiment",
                "Prefill pixel fraction does not assess subject duplication, perspective or retained-source identity",
                "Existing checkpoint was trained before the text-encoding fix; this is inference only"])
    pipeline = model.get_generation_pipeline()
    generate_caption_matrix(pipeline,net,pairs,case,args.seeds,args.steps,args.guidance,reference,out,report)
    annotate_results(root,case,out,report)
    report["status"]="completed_images_pending_visual_review"
    (out/"REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    archive=shutil.make_archive(str(out),"zip",out)
    print("Caption A/B ZIP:",archive,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config",default=str(REPO/"config/flux3_caption_ab_boat_v1.json"))
    parser.add_argument("--steps",type=int,default=25)
    parser.add_argument("--guidance",type=float,default=4.0)
    parser.add_argument("--seeds",type=int,nargs="+",default=[20261020,20261021])
    run(parser.parse_args())


if __name__=="__main__":
    main()
