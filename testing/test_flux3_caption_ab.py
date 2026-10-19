"""Caption-only experiment controls, real file I/O, fixture inference only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("caption_probe",ROOT/"scripts/flux3_caption_ab.py")
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class Embeds:
    def __init__(self, value):self.text_embeds=[value]
    def to(self,device):return Embeds(self.text_embeds[0].to(device))


class CaptionABTests(unittest.TestCase):
    def test_common_window_uses_real_encoded_states_for_every_pair(self):
        calls=[]
        def encode(text,length):
            calls.append((text,length))
            return Embeds(torch.ones(length,4)*(1+len(text)))
        model=types.SimpleNamespace(get_prompt_embeds=lambda text:encode(text,160 if len(text)>10 else 80),
                                    _encode_at_length=encode)
        case=dict(prompt="short",negative_prompt="")
        texts,pairs,window=probe.encode_caption_pairs(model,case,"long PUP caption")
        self.assertEqual(window,160)
        self.assertEqual(texts,{"notebook":"short","pup":"long PUP caption"})
        for positive,negative in pairs.values():
            self.assertEqual(len(positive.text_embeds[0]),160)
            self.assertEqual(len(negative.text_embeds[0]),160)
            self.assertTrue(torch.all(negative.text_embeds[0]!=0))
        self.assertIn(("short",160),calls)

    def test_eight_jobs_hold_reference_seed_cfg_and_sampler_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);net=types.SimpleNamespace(_multiplier=0.,torch_multiplier=torch.zeros(1))
            ref=torch.ones(1,3,16,32)
            negative=Embeds(torch.ones(80,4))
            pairs={"notebook":(Embeds(torch.ones(80,4)*2),negative),"pup":(Embeds(torch.ones(80,4)*3),negative)}
            calls=[]
            def pipeline(**kw):
                calls.append(dict(seed=kw['generator'].initial_seed(),mode=net._multiplier,**{k:v for k,v in kw.items() if k!='generator'}))
                return [Image.new('RGB',(32,16),(128,128,128))]
            case=dict(width=32,height=16);report=dict(results=[])
            probe.generate_caption_matrix(pipeline,net,pairs,case,[11,12],25,4.,ref,out,report)
            self.assertEqual(len(report['results']),8)
            self.assertEqual(len(set(r['file'] for r in report['results'])),8)
            self.assertEqual([r['seed'] for r in calls],[11]*4+[12]*4)
            self.assertEqual([r['mode'] for r in calls],[0.,1.,0.,1.]*2)
            for row in calls:
                self.assertEqual((row['height'],row['width'],row['num_inference_steps'],row['guidance_scale'],row['alpha']),(16,32,25,4.,6.93))
                self.assertIs(row['ctrl_imgs'][0],ref)
                self.assertIs(row['unconditional_embeds'],negative)
            self.assertTrue(all((out/r['file']).is_file() for r in report['results']))

    def test_provenance_guards_source_prompt_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=root/'ref.png';Image.new('RGB',(32,16)).save(ref)
            checkpoint=root/'checkpoint';checkpoint.write_bytes(b'fixture')
            case=dict(sample_id='boat',reference='ref.png',prompt='notebook')
            config=dict(sample_id='boat',reference_sha256=probe.sha(ref),
                notebook_prompt_sha256=hashlib.sha256(b'notebook').hexdigest(),pup_prompt='PUP',
                pup_prompt_sha256=hashlib.sha256(b'PUP').hexdigest(),checkpoint_sha256=probe.sha(checkpoint))
            self.assertIs(probe.validate_probe(root,dict(cases=[case]),config,checkpoint),case)
            for change in [dict(sample_id='wrong'),dict(reference_sha256='bad'),dict(notebook_prompt_sha256='bad'),dict(pup_prompt_sha256='bad'),dict(checkpoint_sha256='bad')]:
                with self.assertRaises(ValueError):probe.validate_probe(root,dict(cases=[case]),dict(config,**change),checkpoint)

    def test_frozen_caption_config_is_consistent(self):
        config=json.loads((ROOT/'config/flux3_caption_ab_boat_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(config['pup_prompt'].encode()).hexdigest(),config['pup_prompt_sha256'])
        self.assertEqual(config['sample_id'],'f3_expand_1072_3')
        self.assertEqual(config['seeds'],[20261020,20261021])


if __name__=='__main__':unittest.main(verbosity=2)
