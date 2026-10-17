"""CPU contract checks using actual model/loader methods; no weights or trainer install.

AST loading isolates these tensor/data paths from unrelated model dependencies.
It does not replace a real GPU optimizer-step smoke test.
"""
import ast
import importlib.util
from pathlib import Path
import tempfile
import types
import typing
import unittest

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'extensions_built_in/diffusion_models/flux3/flux3.py'


def load_paths(model_path=MODEL_PATH):
    positional_path = ROOT / 'extensions_built_in/diffusion_models/flux3/src/positional.py'
    spec = importlib.util.spec_from_file_location('flux3_contract_positions', positional_path)
    positional = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(positional)
    env = dict(torch=torch, F=F, np=np, math=__import__('math'), Image=Image,
               List=typing.List, Optional=typing.Optional, IMAGE_ROPE_SCALE=.5, I2I_COND_L=-1,
               prc_img=positional.prc_img, batched_prc_img=positional.batched_prc_img, prc_txt=positional.prc_txt,
               DEFAULT_MAX_TOKENS=8192, DEFAULT_PAD_MULTIPLE=80, ModelConfig=object,
               AdvancedPromptEmbeds=object)
    pipeline = ast.parse((ROOT / 'extensions_built_in/diffusion_models/flux3/src/pipeline.py').read_text())
    selected = [n for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name in (
        'build_text_conditioning', 'pack_image_latents', 'pack_i2i_image_condition')]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(model_path), 'exec'), env)

    class Base:
        def __init__(self, device, config, dtype, *args, **kwargs):
            self.model_config = config
            self.device_torch = torch.device('cpu')
            self.torch_dtype = torch.float32
            self.has_multiple_control_images = False
            self.use_raw_control_images = False

        @property
        def transformer(self):
            return self.model

    env['BaseModel'] = Base
    parsed = ast.parse(Path(model_path).read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Flux3Model')
    keep = {'__init__', 'get_bucket_divisibility', '_fit_reference', '_control_images_for_batch',
            '_preview_control_paths', 'get_noise_prediction', 'convert_lora_weights_before_save',
            'convert_lora_weights_before_load'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(model_path), 'exec'), env)
    loader_path = ROOT / 'toolkit/dataloader_mixins.py'
    loader = ast.parse(loader_path.read_text())
    method = next(n for c in loader.body if isinstance(c, ast.ClassDef)
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'load_control_image')
    loader_env = dict(torch=torch, Image=Image, exif_transpose=exif_transpose, transforms=transforms,
                      ImageOps=__import__('PIL.ImageOps', fromlist=['ImageOps']), print_acc=lambda *a: None)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(loader_path), 'exec'), loader_env)
    return env['Flux3Model'], loader_env['load_control_image']


class RecordingTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.))
        self.device = torch.device('cpu')
        self.last = None

    def forward(self, **kwargs):
        self.last = kwargs
        return {'x_image': kwargs['x_image'] * self.gain}


class Flux3ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Model, loader = load_paths()
        cls.loader = staticmethod(loader)

    def model(self, **kwargs):
        model = self.Model('cpu', types.SimpleNamespace(model_kwargs=kwargs))
        model.model = RecordingTransformer()
        model.encode_images = lambda image, **kw: F.avg_pool2d(image.mean(1, keepdim=True), 16).repeat(1, 128, 1, 1)
        return model

    def predict(self, model, batch, batch_size=1):
        latent = torch.ones(batch_size, 128, 4, 6)
        embeds = types.SimpleNamespace(text_embeds=[torch.ones(80, 4)] * batch_size)
        return model.get_noise_prediction(latent, torch.tensor([500.] * batch_size), embeds, batch)

    def test_model_declares_raw_multi_reference_capability(self):
        model = self.model()
        self.assertTrue(model.use_raw_control_images)
        self.assertTrue(model.has_multiple_control_images)

    def test_actual_loader_keeps_independent_shapes_and_two_references(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, size in enumerate(((80, 32), (32, 80))):
                p = Path(tmp) / f'{i}.png'
                Image.new('RGB', size, (40 + i*120,)*3).save(p)
                paths.append(str(p))
            item = types.SimpleNamespace(get_new_control_paths=lambda: paths, load_rgba=False,
                full_size_control_images=True, use_raw_control_images=model.use_raw_control_images,
                flip_x=False, flip_y=False, dataset_config=types.SimpleNamespace(buckets=True, control_transparent_color=[255]*3),
                scale_to_width=64, scale_to_height=64, crop_x=0, crop_y=0, crop_width=64, crop_height=64,
                aug_replay_spatial_transforms=False, control_tensor=None, control_tensor_list=None)
            self.loader(item)
            self.assertIsNotNone(item.control_tensor_list)
            self.assertEqual([tuple(x.shape[-2:]) for x in item.control_tensor_list], [(32, 80), (80, 32)])
            batch = types.SimpleNamespace(control_tensor_list=[item.control_tensor_list], control_tensor=None)
            self.predict(model, batch)
            ids = model.model.last['x_image_cond_ids'][0, :, 3]
            self.assertEqual(set(ids.tolist()), {-1., -2.})

    def test_legacy_stacked_controls_do_not_drop_second_reference(self):
        model = self.model()
        batch = types.SimpleNamespace(control_tensor_list=None,
            control_tensor=torch.stack([torch.full((3, 32, 32), .2), torch.full((3, 32, 32), .8)])[None])
        self.predict(model, batch)
        ids = model.model.last['x_image_cond_ids'][0, :, 3]
        self.assertEqual(set(ids.tolist()), {-1., -2.})
        cond = model.model.last['x_image_cond'][0]
        self.assertLess(cond[:len(cond)//2].mean(), cond[len(cond)//2:].mean())

    def test_three_reference_role_order_timestep_and_gradient(self):
        model = self.model()
        batch = types.SimpleNamespace(control_tensor=None,
            control_tensor_list=[[torch.full((3, 32, 64), v) for v in (.2, .5, .8)]])
        pred = self.predict(model, batch)
        kw = model.model.last
        self.assertEqual(set(kw['x_image_cond_ids'][0, :, 3].tolist()), {-1., -2., -3.})
        self.assertTrue(torch.all(kw['x_image_cond_timesteps'] == 0))
        means = [kw['x_image_cond'][0][kw['x_image_cond_ids'][0, :, 3] == -1-i].mean().item() for i in range(3)]
        self.assertLess(means[0], means[1]); self.assertLess(means[1], means[2])
        self.assertEqual(pred.shape, (1, 128, 4, 6))
        pred.sum().backward()
        self.assertIsNotNone(model.model.gain.grad)

    def test_fixed_budget_is_independent_of_target_and_matches_workflow_grid(self):
        model = self.model(reference_megapixels=1., reference_resize_method='lanczos')
        image = torch.linspace(0, 1, 3*64*96).reshape(1, 3, 64, 96)
        a = model._fit_reference(image, 1024**2)
        b = model._fit_reference(image, 2048**2)
        self.assertEqual(a.shape, b.shape)
        scale = (1024**2 / (96*64))**.5
        width, height = round(96*scale/16)*16, round(64*scale/16)*16
        expected = Image.fromarray(np.clip(255*image[0].movedim(0, -1).numpy(), 0, 255).astype(np.uint8)).resize((width, height), Image.Resampling.LANCZOS)
        expected = torch.from_numpy(np.array(expected).astype(np.float32)/255).movedim(-1, 0)[None]
        self.assertTrue(torch.equal(a, expected))

    def test_default_reference_sizing_remains_target_matched(self):
        model = self.model()
        image = torch.rand(1, 3, 64, 96)
        self.assertNotEqual(model._fit_reference(image, 1024**2).shape,
                            model._fit_reference(image, 2048**2).shape)

    def test_invalid_budget_and_resize_method_fail_early(self):
        for value in (0., -1., float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.model(reference_megapixels=value)
        with self.assertRaises(ValueError): self.model(reference_resize_method='unknown')

    def test_lora_prefix_roundtrip_preserves_tensor_keys(self):
        model = self.model()
        state = {'transformer.content_mode_blocks.image.0.attn.to_q.lora_down.weight': torch.ones(1, 2)}
        saved = model.convert_lora_weights_before_save(state)
        restored = model.convert_lora_weights_before_load(saved)
        self.assertEqual(list(restored), list(state))
        self.assertIs(next(iter(restored.values())), next(iter(state.values())))

    def test_preview_alias_is_removed_without_deduplicating_distinct_slots(self):
        model = self.model()
        cfg = types.SimpleNamespace(ctrl_img='same.png', ctrl_img_1='same.png', ctrl_img_2='same.png', ctrl_img_3='third.png')
        self.assertEqual(model._preview_control_paths(cfg), ['same.png', 'same.png', 'third.png'])

    def test_single_reference_no_reference_and_invalid_mixed_batch(self):
        model = self.model()
        self.predict(model, types.SimpleNamespace(control_tensor=None, control_tensor_list=None))
        self.assertNotIn('x_image_cond', model.model.last)
        self.predict(model, types.SimpleNamespace(control_tensor=torch.ones(1, 3, 32, 48), control_tensor_list=None))
        self.assertEqual(set(model.model.last['x_image_cond_ids'][0, :, 3].tolist()), {-1.})
        bad = types.SimpleNamespace(control_tensor=None, control_tensor_list=[[torch.ones(3, 32, 48)], []])
        with self.assertRaisesRegex(ValueError, 'same 1..10'):
            self.predict(model, bad, 2)

    def test_two_batch_items_keep_each_reference_stream(self):
        model = self.model()
        batch = types.SimpleNamespace(control_tensor_list=None, control_tensor=torch.stack([
            torch.stack([torch.full((3, 32, 48), a), torch.full((3, 32, 48), b)])
            for a, b in ((.1, .4), (.6, .9))]))
        pred = self.predict(model, batch, 2)
        self.assertEqual(pred.shape[0], 2)
        for ids in model.model.last['x_image_cond_ids']:
            self.assertEqual(set(ids[:, 3].tolist()), {-1., -2.})


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
