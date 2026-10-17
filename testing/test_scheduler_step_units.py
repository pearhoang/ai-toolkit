import importlib.util
import ast
from pathlib import Path
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

    def test_optimizer_units_default_and_input_config_not_mutated(self):
        original = {"phase_step_unit": "training_steps", "warmup_steps": 101,
                    "phases": [{"steps": 999, "start_lr": 1., "end_lr": .1}]}
        normalized = self.scheduler.normalize_scheduler_params_for_optimizer_updates(original, 4)
        self.assertEqual(normalized["warmup_steps"], 26)
        self.assertEqual(normalized["phases"][0]["steps"], 250)
        self.assertEqual(original["warmup_steps"], 101)
        self.assertEqual(original["phases"][0]["steps"], 999)
        self.assertEqual(self.scheduler.normalize_scheduler_params_for_optimizer_updates({"warmup_steps": 101}, 4), {"warmup_steps": 101})

    def test_training_step_alias_shortcuts_and_epoch_accumulation_guard(self):
        normalized = self.scheduler.normalize_scheduler_params_for_optimizer_updates(
            {"step_unit": "displayed_steps", "phase1_steps": 9, "T_max": 20}, 2)
        self.assertEqual(normalized, {"phase1_steps": 5, "T_max": 10})
        with self.assertRaises(ValueError):
            self.scheduler.normalize_scheduler_params_for_optimizer_updates({"phase_step_unit": "training_steps"}, -1)
        with self.assertRaises(ValueError):
            self.scheduler.normalize_scheduler_params_for_optimizer_updates({"phase_step_unit": "unknown"}, 2)

    def test_phased_curve_and_resume_preserve_parameter_group_ratio(self):
        def make():
            optimizer = torch.optim.SGD([{"params": [torch.nn.Parameter(torch.ones(1))], "lr": 1.},
                                         {"params": [torch.nn.Parameter(torch.ones(1))], "lr": .5}])
            scheduler = self.scheduler.get_lr_scheduler("warmup_then_phased_cosine", optimizer,
                warmup_steps=2, reference_lr=1., phases=[{"steps": 3, "start_lr": 1., "end_lr": .2},
                                                      {"steps": 2, "start_lr": .2, "end_lr": .1}])
            return optimizer, scheduler
        optimizer, scheduler = make()
        observed = []
        for i in range(7):
            scheduler.step(i)
            a, b = [group["lr"] for group in optimizer.param_groups]
            observed.append(a)
            self.assertAlmostEqual(b, a*.5)
        self.assertEqual(observed[:3], [.5, 1., 1.])
        self.assertAlmostEqual(observed[4], .2)
        self.assertAlmostEqual(observed[-1], .1)
        scheduler.step(3)
        _, resumed = make()
        resumed.load_state_dict(scheduler.state_dict())
        scheduler.step(); resumed.step()
        self.assertEqual(scheduler.get_last_lr(), resumed.get_last_lr())

    def test_actual_merged_trainer_steps_scheduler_only_with_optimizer(self):
        path = Path("extensions_built_in/sd_trainer/SDTrainer.py")
        tree = ast.parse(path.read_text())
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp):
                if ast.unparse(node.test) == "not self.is_grad_accumulation_step":
                    body = ast.Module(body=node.body, type_ignores=[])
                    methods = [ast.unparse(call.func) for call in ast.walk(body) if isinstance(call, ast.Call)]
                    if "self.optimizer.step" in methods:
                        calls.append(methods)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].count("self.lr_scheduler.step"), 1)
        scheduler_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and ast.unparse(node.func) == "self.lr_scheduler.step"]
        self.assertEqual(len(scheduler_calls), 1)

    def test_actual_resume_index_uses_optimizer_updates(self):
        tree = ast.parse(Path("jobs/process/BaseSDTrainProcess.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseSDTrainProcess")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_scheduler_step_index")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "resume_index", "exec"), namespace)
        fake = types.SimpleNamespace(step_num=1234, train_config=types.SimpleNamespace(gradient_accumulation_steps=4))
        self.assertEqual(namespace[method.name](fake), 308)


if __name__ == "__main__":
    unittest.main()
