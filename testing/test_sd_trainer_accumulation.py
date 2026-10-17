"""Run the actual SDTrainer hook with CPU gradients, no model downloads."""
import ast
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Union, List
import unittest

import torch


def load_hook():
    path = Path(__file__).resolve().parents[1] / 'extensions_built_in/sd_trainer/SDTrainer.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SDTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'hook_train_loop')
    namespace = dict(torch=torch, Union=Union, List=List, DataLoaderBatchDTO=object,
                     OrderedDict=OrderedDict, sync_grad_transfers=lambda: None,
                     CustomAdapter=type('CustomAdapter', (), {}),
                     ClipVisionAdapter=type('ClipVisionAdapter', (), {}))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['hook_train_loop']


def make_trainer(accumulation_steps):
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    scheduler_updates = []
    trainer = SimpleNamespace(
        optimizer=torch.optim.SGD([parameter], lr=.1),
        train_config=SimpleNamespace(gradient_accumulation_steps=accumulation_steps,
                                     optimizer='adamw8bit', max_grad_norm=100.),
        sd=SimpleNamespace(is_multistage=False), model_config=SimpleNamespace(low_vram=False),
        params=[parameter], adapter=None, embedding=None, ema=None,
        steps_this_boundary=0, grad_accumulation_step=1,
        accelerator=SimpleNamespace(clip_grad_norm_=lambda *args: None),
        timer=lambda *args: nullcontext(),
        lr_scheduler=SimpleNamespace(step=lambda: scheduler_updates.append(True)),
        end_of_training_loop=lambda: None)

    def microbatch(target, accum_scale):
        loss = (parameter - float(target)).square().mean() * accum_scale
        loss.backward()
        return loss.detach()

    trainer.train_single_accumulation = microbatch
    trainer.parameter = parameter
    trainer.scheduler_updates = scheduler_updates
    return trainer


def call_like_base_loop(trainer, targets):
    # Same counter setup as BaseSDTrainProcess; optimizer lives inside the real hook.
    trainer.is_grad_accumulation_step = trainer.grad_accumulation_step < trainer.train_config.gradient_accumulation_steps
    if not trainer.is_grad_accumulation_step:
        trainer.grad_accumulation_step = 0
    result = load_hook()(trainer, targets)
    trainer.grad_accumulation_step += 1
    return result


def reference_updates(groups):
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    for targets in groups:
        optimizer.zero_grad()
        torch.stack([(parameter - t).square().mean() for t in targets]).mean().backward()
        optimizer.step()
    return parameter


class AccumulationTests(unittest.TestCase):
    def test_two_different_microbatches_match_true_mean_gradient(self):
        trainer = make_trainer(2)
        call_like_base_loop(trainer, [1.])
        call_like_base_loop(trainer, [3.])
        self.assertTrue(torch.allclose(trainer.parameter, reference_updates([[1., 3.]]), atol=1e-7))
        self.assertAlmostEqual(trainer.parameter.item(), .4)

    def test_first_microbatch_gradient_survives_until_update(self):
        trainer = make_trainer(2)
        call_like_base_loop(trainer, [1.])
        self.assertAlmostEqual(trainer.parameter.grad.item(), -1.)
        self.assertEqual(len(trainer.scheduler_updates), 0)
        call_like_base_loop(trainer, [3.])
        self.assertIsNone(trainer.parameter.grad)
        self.assertEqual(len(trainer.scheduler_updates), 1)

    def test_two_cycles_do_not_reuse_previous_gradients(self):
        trainer = make_trainer(2)
        for target in [1., 3., 0., 2.]:
            call_like_base_loop(trainer, [target])
        self.assertTrue(torch.allclose(trainer.parameter, reference_updates([[1., 3.], [0., 2.]]), atol=1e-7))
        self.assertEqual(len(trainer.scheduler_updates), 2)

    def test_single_step_accumulation_is_unchanged(self):
        trainer = make_trainer(1)
        for target in [1., 3.]:
            call_like_base_loop(trainer, [target])
        self.assertTrue(torch.allclose(trainer.parameter, reference_updates([[1.], [3.]]), atol=1e-7))
        self.assertEqual(len(trainer.scheduler_updates), 2)

    def test_batch_list_and_repeated_calls_normalize_once(self):
        trainer = make_trainer(2)
        call_like_base_loop(trainer, [1., 2.])
        call_like_base_loop(trainer, [3., 4.])
        self.assertTrue(torch.allclose(trainer.parameter, reference_updates([[1., 2., 3., 4.]]), atol=1e-7))

    def test_three_microbatches_match_mean(self):
        trainer = make_trainer(3)
        for target in [1., 2., 6.]:
            call_like_base_loop(trainer, [target])
        self.assertTrue(torch.allclose(trainer.parameter, reference_updates([[1., 2., 6.]]), atol=1e-7))
        self.assertEqual(len(trainer.scheduler_updates), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
