import importlib.util
import sys
import types
import unittest

import torch


def load_scheduler_module():
    optimization = types.ModuleType("diffusers.optimization")
    optimization.SchedulerType = lambda name: name
    optimization.TYPE_TO_SCHEDULER_FUNCTION = {}
    optimization.get_constant_schedule_with_warmup = (
        lambda optimizer, num_warmup_steps: torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, float(step) / float(max(1, num_warmup_steps))),
        )
    )
    sys.modules["diffusers"] = types.ModuleType("diffusers")
    sys.modules["diffusers.optimization"] = optimization
    spec = importlib.util.spec_from_file_location("scheduler_under_test", "toolkit/scheduler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SchedulerStepUnitTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = load_scheduler_module()

    def test_converts_training_steps_to_optimizer_updates_for_phased_schedule(self):
        params = {
            "phase_step_unit": "training_steps",
            "warmup_steps": 300,
            "total_iters": 7300,
            "phases": [
                {"steps": 1000, "start_lr": 1e-4, "end_lr": 5e-5},
                {"steps": 2000, "start_lr": 5e-5, "end_lr": 3e-5},
            ],
        }

        normalized = self.scheduler.normalize_scheduler_params_for_optimizer_updates(
            params,
            gradient_accumulation_steps=2,
        )

        self.assertNotIn("phase_step_unit", normalized)
        self.assertEqual(normalized["warmup_steps"], 150)
        self.assertEqual(normalized["total_iters"], 3650)
        self.assertEqual(normalized["phases"][0]["steps"], 500)
        self.assertEqual(normalized["phases"][1]["steps"], 1000)

    def test_uses_explicit_phased_scheduler_name_with_phases(self):
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([param], lr=1.0)

        scheduler = self.scheduler.get_lr_scheduler(
            "warmup_then_phased_cosine",
            optimizer,
            warmup_steps=2,
            reference_lr=1.0,
            phases=[
                {"steps": 4, "start_lr": 1.0, "end_lr": 0.0},
            ],
        )

        scheduler.step(0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5)


if __name__ == "__main__":
    unittest.main()
