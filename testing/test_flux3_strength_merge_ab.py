"""CPU controls/restore tests; not pretrained Base precision/quality validation."""
import importlib.util
import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('strength_probe',ROOT/'scripts/flux3_strength_merge_ab.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class Adapter:
    def __init__(self):
        torch.manual_seed(24)
        layer=torch.nn.Linear(8,8,bias=False).bfloat16().requires_grad_(False)
        self.org_module=[layer];self.scale=1.
        self.lora_down=torch.nn.Linear(8,2,bias=False).requires_grad_(False)
        self.lora_up=torch.nn.Linear(2,8,bias=False).requires_grad_(False)
        with torch.no_grad():self.lora_down.weight.mul_(.01);self.lora_up.weight.mul_(.01)
    def merge_in(self,merge_weight):
        layer=self.org_module[0]
        with torch.no_grad():layer.weight.copy_((layer.weight.float()+merge_weight*(self.lora_up.weight@self.lora_down.weight)*self.scale).bfloat16())


class StrengthTests(unittest.TestCase):
    def test_actual_preview_merge_branch_honors_model_capability(self):
        base=ast.parse((ROOT/'toolkit/models/base_model.py').read_text(encoding='utf-8'))
        cls=next(n for n in base.body if isinstance(n,ast.ClassDef) and n.name=='BaseModel')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate_images')
        branch=next(n for n in ast.walk(method) if isinstance(n,ast.If) and 'unique_network_weights' in ast.unparse(n.test))
        # Execute the actual merge decision and body with a recording network.
        merged=[]
        for enabled,expected in [(True,True),(False,False)]:
            env=dict(self=types.SimpleNamespace(merge_lora_for_sampling=enabled),
                unique_network_weights={1.},network=types.SimpleNamespace(can_merge_in=True,merge_in=lambda **kw:merged.append(kw)))
            before=len(merged)
            exec(compile(ast.Module(body=[branch],type_ignores=[]),'actual_Base_preview_merge_branch','exec'),env)
            self.assertEqual(len(merged)>before,expected)
        for path,name,expected in [(ROOT/'toolkit/models/base_model.py','BaseModel',True),
            (ROOT/'extensions_built_in/diffusion_models/flux3/flux3.py','Flux3Model',False)]:
            tree=ast.parse(path.read_text(encoding='utf-8'))
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
            value=next(ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='merge_lora_for_sampling' for t in n.targets))
            self.assertEqual(value,expected)

    def test_restore_and_measure_with_actual_trainer_merge_method(self):
        tree=ast.parse((ROOT/'toolkit/network_mixins.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ToolkitModuleMixin')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='merge_in')
        env=dict(torch=torch)
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,fn],type_ignores=[])),'actual_trainer_merge','exec'),env)
        adapter=Adapter();adapter.can_merge_in=True;adapter.full_rank=False
        adapter.merge_in=types.MethodType(env['merge_in'],adapter)
        original=adapter.org_module[0].weight.clone();net=types.SimpleNamespace(is_merged_in=False)
        fake=types.ModuleType('toolkit.util.quantize');fake.is_quantized_tensor=lambda weight:False
        stats={}
        with patch.dict(sys.modules,{'toolkit.util.quantize':fake}):
            with probe.exact_merge_trial([adapter],net,stats):
                self.assertEqual(stats['elements'],64)
        self.assertTrue(torch.equal(adapter.org_module[0].weight,original))
        self.assertTrue(stats['restored_base_weights_bit_exact'])

    def test_actual_base_snapshot_restore_is_exact_and_measured(self):
        adapter=Adapter();original=adapter.org_module[0].weight.clone()
        net=types.SimpleNamespace(is_merged_in=False);stats={}
        with probe.exact_merge_trial([adapter],net,stats):
            self.assertTrue(net.is_merged_in)
        self.assertFalse(net.is_merged_in)
        self.assertTrue(torch.equal(adapter.org_module[0].weight,original))
        self.assertTrue(stats['restored_base_weights_bit_exact'])
        self.assertEqual(stats['elements'],64)
        self.assertGreater(stats['expected_delta_square'],0)
        self.assertGreaterEqual(stats['unchanged_weight_fraction'],0)

    def test_restore_runs_even_if_generation_raises(self):
        adapter=Adapter();original=adapter.org_module[0].weight.clone();net=types.SimpleNamespace(is_merged_in=False);stats={}
        with self.assertRaisesRegex(RuntimeError,'generation failed'):
            with probe.exact_merge_trial([adapter],net,stats):raise RuntimeError('generation failed')
        self.assertTrue(torch.equal(adapter.org_module[0].weight,original))
        self.assertFalse(net.is_merged_in)
        self.assertTrue(stats['restored_base_weights_bit_exact'])

    def test_variant_matrix_same_seed_ref_and_cfg_with_prediction_capture(self):
        net=types.SimpleNamespace(is_merged_in=False,_multiplier=0.,torch_multiplier=torch.zeros(1))
        class Transformer(torch.nn.Module):
            def forward(self):return {'x_image':torch.ones(1,2,2)*(1+net._multiplier)}
        transformer=Transformer();model=types.SimpleNamespace(transformer=transformer);calls=[]
        reference=object();pair=(object(),object());case=dict(probe_seed=21,height=16,width=32,probe_role='heldout',sample_id='fixture')
        def pipeline(**kwargs):
            calls.append((net._multiplier,kwargs['generator'].initial_seed(),kwargs['ctrl_imgs'][0],kwargs['num_inference_steps'],kwargs['guidance_scale']))
            transformer();return [Image.new('RGB',(32,16))]
        results=[];predictions={}
        with tempfile.TemporaryDirectory() as tmp:
            probe.run_case(model,pipeline,net,pair,reference,case,25,4.,Path(tmp),results,predictions)
            self.assertEqual([c[0] for c in calls],[0.,1.,2.,4.])
            self.assertTrue(all(c[1:]==(21,reference,25,4.) for c in calls))
            self.assertEqual(len(results),4)
            self.assertTrue(torch.equal(predictions[('fixture','forward_s2')],torch.ones(1,2,2)*3))
            net.is_merged_in=True
            probe.run_case(model,pipeline,net,pair,reference,case,25,4.,Path(tmp),results,predictions)
            self.assertEqual(len(results),5)
            self.assertEqual(results[-1]['variant'],'merge_bf16_s1')
            self.assertEqual(probe.prediction_stats(predictions[('fixture','forward_s1')],predictions[('fixture','forward_s0')])['MAE'],1.)

    def test_frozen_manifest_and_payload_guards_on_real_pilot(self):
        # Verify the data fixture produced in this workspace when available.
        workspace=ROOT.parents[1]
        config=json.loads((ROOT/'config/flux3_strength_merge_pilot_v1.json').read_text(encoding='utf-8'))
        for task,spec in config['tasks'].items():
            dataset=workspace/'outputs/flux3_training_reaudit_20261019/caption_pilot_v1/qa_extracted'/task
            ckpt=workspace/'outputs/flux3_training_reaudit_20261019/pilot_actual_0918'/task/spec['checkpoint_name']
            if not dataset.exists() or not ckpt.exists():continue
            cases=probe.verify_inputs(dataset,ckpt,spec)
            self.assertEqual([c['probe_role'] for c in cases],['train','heldout'])
            self.assertEqual([c['split'] for c in cases],['train','validation'])
            with self.assertRaisesRegex(ValueError,'exact released'):
                probe.verify_inputs(dataset,ckpt,dict(spec,manifest_sha256='bad'))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
