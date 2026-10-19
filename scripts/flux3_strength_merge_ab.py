"""Pilot checkpoint strengths and FP32-forward vs BF16-merge, no training."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import torch
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.flux3_conditioning_ab import attach_native_lora, sha

REPO=Path(__file__).resolve().parents[1]
VARIANTS=[('forward_s0',0.,False),('forward_s1',1.,False),('forward_s2',2.,False),('forward_s4',4.,False),('merge_bf16_s1',1.,True)]


def verify_inputs(root,checkpoint,spec):
    if sha(root/'metadata/manifest.jsonl')!=spec['manifest_sha256']:
        raise ValueError('Use the exact released caption PILOT TAR dataset, not full V2')
    if sha(checkpoint)!=spec['checkpoint_sha256']:
        raise ValueError('Use the uploaded pilot final step200 checkpoint')
    rows=[json.loads(s) for s in (root/'metadata/manifest.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(rows)!=15:raise ValueError('Pilot must contain15pairs')
    byid={r['sample_id']:r for r in rows}
    cases=[]
    for planned in spec['cases']:
        row=byid[planned['sample_id']]
        if row['split']!=('train' if planned['role']=='train' else 'validation'):
            raise ValueError('Case split changed')
        for key in ['control_path_1','target','caption_file']:
            path=(root/row[key]).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path)!=row['payload_integrity'][row[key]]['sha256']:
                raise ValueError('Payload changed: '+row[key])
        if (root/row['caption_file']).read_text(encoding='utf-8').strip()!=row['caption']:
            raise ValueError('Caption mismatch')
        cases.append(dict(row,probe_role=planned['role'],probe_seed=planned['seed']))
    return cases


@contextmanager
def exact_merge_trial(adapters,net,stats):
    """Actual trainer merge method; CPU original snapshots restore bit-for-bit.

    Keep only one expanded BA on GPU at a time. No subtractive merge_out.
    Snapshots take roughly the adapted Base weight size in system RAM.
    """
    originals=[]
    old_merged=net.is_merged_in
    try:
        net.is_merged_in=True
        totals=dict(elements=0,unchanged=0,base_square=0.,expected_delta_square=0.,realized_delta_square=0.,delta_rounding_error_square=0.)
        for index,adapter in enumerate(adapters):
            layer=adapter.org_module[0]
            original=layer.weight.detach().cpu().clone()
            originals.append((layer,original))
            adapter.merge_in(merge_weight=1.0)
            base=original.to(device=layer.weight.device,dtype=torch.float32)
            delta=(adapter.lora_up.weight.float()@adapter.lora_down.weight.float())*float(adapter.scale)
            realized=layer.weight.detach().float()-base
            totals['elements']+=base.numel()
            totals['unchanged']+=int((layer.weight.detach()==base).sum().item())
            for name,value in [('base_square',base),('expected_delta_square',delta),('realized_delta_square',realized),('delta_rounding_error_square',realized-delta)]:
                totals[name]+=float(value.double().square().sum().item())
            del base,delta,realized
            if index%40==0:print('BF16 merge measurement',index+1,'/',len(adapters),flush=True)
        stats.update(totals)
        stats['unchanged_weight_fraction']=totals['unchanged']/totals['elements']
        stats['relative_delta_norm_to_base']=(totals['expected_delta_square']/max(totals['base_square'],1e-30))**.5
        stats['relative_rounding_error_norm_to_delta']=(totals['delta_rounding_error_square']/max(totals['expected_delta_square'],1e-30))**.5
        yield
    finally:
        restored=True
        with torch.no_grad():
            for layer,original in originals:
                layer.weight.copy_(original)
                restored=restored and torch.equal(layer.weight.detach().cpu(),original)
        net.is_merged_in=old_merged
        stats['restored_base_weights_bit_exact']=restored
        if not restored:raise RuntimeError('Base snapshot restore failed; stop the experiment')


def prediction_stats(prediction,baseline):
    delta=prediction.double()-baseline.double()
    return dict(MAE=float(delta.abs().mean()),RMS=float(delta.square().mean().sqrt()),
        relative_L2=float(delta.square().sum().sqrt()/baseline.double().square().sum().sqrt().clamp_min(1e-30)))


def run_case(model,pipeline,net,pair,reference,case,steps,guidance,out,results,first_predictions,variants=VARIANTS):
    for label,strength,merged in variants:
        if bool(net.is_merged_in)!=merged:continue
        net._multiplier=strength;net.torch_multiplier.fill_(strength)
        first=[]
        def capture(module,inputs,output):
            if not first:first.append(output['x_image'].detach().float().cpu())
        handle=model.transformer.register_forward_hook(capture)
        try:
            generator=torch.Generator(device='cpu').manual_seed(case['probe_seed'])
            start=time.monotonic()
            images=pipeline(conditional_embeds=pair[0],unconditional_embeds=pair[1],
                height=case['height'],width=case['width'],num_inference_steps=steps,guidance_scale=guidance,
                generator=generator,ctrl_imgs=[reference],alpha=6.93)
        finally:handle.remove()
        filename=f"{case['probe_role']}_{case['sample_id']}_{label}.png";images[0].save(out/filename)
        if not first:raise RuntimeError('No transformer prediction captured')
        first_predictions[(case['sample_id'],label)]=first[0]
        row=dict(file=filename,case_id=case['sample_id'],role=case['probe_role'],seed=case['probe_seed'],variant=label,
            strength=strength,merged=merged,sha256=sha(out/filename),seconds=time.monotonic()-start)
        results.append(row);print(filename,round(row['seconds'],1),'seconds',flush=True)


def run(args):
    config=json.loads(Path(args.config).read_text(encoding='utf-8'));spec=config['tasks'][args.task]
    root=Path(args.dataset).resolve();checkpoint=Path(args.checkpoint).resolve()
    cases=verify_inputs(root,checkpoint,spec)
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<60*1024**3:
        raise RuntimeError('Use A10080GB / RTX600096GB with Base BF16')
    from toolkit.config_modules import ModelConfig
    from toolkit.unloader import unload_text_encoder
    from torchvision.transforms.functional import to_tensor
    from extensions_built_in.diffusion_models.flux3.flux3 import Flux3Model
    recipe=dict(name_or_path=args.base_path or 'black-forest-labs/flux-3-image-klein-base',arch='flux3',dtype='bf16',
        quantize=False,quantize_te=False,low_vram=False,layer_offloading=False,
        model_kwargs=dict(reference_megapixels=1.0,reference_resize_method='lanczos',match_target_res=True,max_text_length=8192,text_pad_multiple=80,sample_alpha=6.93))
    if args.te_path:recipe['te_name_or_path']=args.te_path
    model=Flux3Model('cuda:0',ModelConfig(**recipe),dtype='bf16');model.load_model()
    model.model.eval().requires_grad_(False)
    weight_proof={name:sha(model._package_file(name)) for name in ['model.safetensors','ae.safetensors']}
    if weight_proof!={'model.safetensors':config['base_sha256'],'ae.safetensors':config['vae_sha256']}:
        raise ValueError('Base/VAE mismatch')
    if model.model.dtype!=torch.bfloat16:raise ValueError('Expected BF16 Base')
    pairs={};refs={}
    for case in cases:
        pos=model.get_prompt_embeds(case['caption']);neg=model.get_prompt_embeds('')
        pos,neg=model.prepare_sample_prompt_pair(types.SimpleNamespace(prompt=case['caption'],negative_prompt=''),pos,neg)
        pairs[case['sample_id']]=(pos.to('cpu'),neg.to('cpu'))
        refs[case['sample_id']]=model._fit_reference(to_tensor(Image.open(root/case['control_path_1']).convert('RGB')).unsqueeze(0),case['width']*case['height'])
    del pos,neg
    unload_text_encoder(model);gc.collect();torch.cuda.empty_cache()
    net,adapters=attach_native_lora(model.model,checkpoint,32)
    # Match training: adapters FP32, Base BF16. Strength is the only scale knob.
    for adapter in adapters:adapter.to(device=model.device_torch,dtype=torch.float32)
    if len(adapters)!=258:raise ValueError('Expected258adapters')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',task=args.task,no_training=True,checkpoint_sha256=sha(checkpoint),
        base_vae_sha256=weight_proof,base_dtype='bf16',adapter_dtype='float32',alpha=32,rank=32,flow_shift=6.93,
        steps=args.steps,cfg=args.guidance,cases=cases,results=[],merge_stats={},first_prediction_metrics={},
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        toolkit_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        limits=['Strength>1 is diagnostic, not quality approval','First-timestep prediction difference is not full image quality',
            'Two cases/one seed; train vs heldout not a generalization estimate','FP32 adapter forward may differ from earlier BF16-adapter standalone probes'])
    first={};pipeline=model.get_generation_pipeline()
    for case in cases:
        run_case(model,pipeline,net,pairs[case['sample_id']],refs[case['sample_id']],case,args.steps,args.guidance,out,report['results'],first)
        (out/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    net._multiplier=1.;net.torch_multiplier.fill_(1.)
    with exact_merge_trial(adapters,net,report['merge_stats']):
        for case in cases:
            run_case(model,pipeline,net,pairs[case['sample_id']],refs[case['sample_id']],case,args.steps,args.guidance,out,report['results'],first)
            (out/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for case in cases:
        sid=case['sample_id'];baseline=first[(sid,'forward_s0')]
        report['first_prediction_metrics'][sid]={label:prediction_stats(first[(sid,label)],baseline) for label,_,_ in VARIANTS}
        report['first_prediction_metrics'][sid]['merge_vs_forward_s1']=prediction_stats(first[(sid,'merge_bf16_s1')],first[(sid,'forward_s1')])
        board=Image.new('RGB',(2000,380),'white');draw=ImageDraw.Draw(board)
        for i,(label,_,_) in enumerate(VARIANTS):
            row=next(r for r in report['results'] if r['case_id']==sid and r['variant']==label)
            image=Image.open(out/row['file']).convert('RGB');image.thumbnail((400,340))
            draw.text((400*i+5,8),label,fill='black');board.paste(image,(400*i,35))
        board.save(out/(case['probe_role']+'_STRENGTH_MERGE.jpg'),quality=95)
    report['status']='completed_images_pending_visual_review'
    (out/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Result ZIP:',shutil.make_archive(str(out),'zip',out),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['general','structural'],required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(REPO/'config/flux3_strength_merge_pilot_v1.json'))
    p.add_argument('--base-path');p.add_argument('--te-path');p.add_argument('--steps',type=int,default=25);p.add_argument('--guidance',type=float,default=4.)
    args=p.parse_args()
    if not 1<=args.steps<=200 or not 1<args.guidance<=30:p.error('Use steps1-200 and finite CFG>1<=30')
    run(args)


if __name__=='__main__':main()
