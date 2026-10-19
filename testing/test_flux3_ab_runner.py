"""Probe runner integrity and native-checkpoint binding checks; no GPU run."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("flux3_ab_runner", ROOT / "scripts/flux3_conditioning_ab.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class Adapter(torch.nn.Module):
    def __init__(self, name, layer, lora_dim, alpha, network):
        super().__init__()
        self.lora_down = torch.nn.Linear(layer.in_features, lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(lora_dim, layer.out_features, bias=False)
        self.layer = [layer]; self.network = network; self.scale = alpha / lora_dim
    def apply_to(self):
        original = self.layer[0].forward
        self.layer[0].forward = lambda x: original(x) + self.network._multiplier * self.scale * self.lora_up(self.lora_down(x))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.block = torch.nn.Linear(3, 4, bias=False)
    @property
    def device(self):
        return self.block.weight.device


class ABRunnerTests(unittest.TestCase):
    def attach(self, weights, model=None):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "checkpoint.safetensors"
        save_file(weights, str(path))
        fake = types.ModuleType("toolkit.lora_special"); fake.LoRAModule = Adapter
        with patch.dict(sys.modules, {"toolkit.lora_special": fake}):
            return runner.attach_native_lora(model or Model(), path, 2)

    def weights(self):
        return {"diffusion_model.block.lora_A.weight": torch.ones(2, 3),
                "diffusion_model.block.lora_B.weight": torch.ones(4, 2)}

    def test_native_pair_mapping_and_off_on_do_not_merge_base(self):
        model = Model(); before = model.block.weight.clone()
        net, adapters = self.attach(self.weights(), model)
        x = torch.ones(1, 3)
        off = model.block(x)
        net._multiplier = 1
        on = model.block(x)
        self.assertEqual(len(adapters), 1)
        self.assertTrue(torch.allclose(on-off, torch.ones(1, 4)*6))
        self.assertTrue(torch.equal(model.block.weight, before))

    def test_unmatched_extra_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "no keys may be ignored"):
            self.attach(dict(self.weights(), unknown=torch.ones(1)))

    def test_wrong_shape_rejected_before_binding(self):
        w = self.weights(); w["diffusion_model.block.lora_A.weight"] = torch.ones(2, 5)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.attach(w)

    def test_nonfinite_weights_rejected(self):
        w = self.weights(); w["diffusion_model.block.lora_A.weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.attach(w)

    def test_unknown_layer_rejected(self):
        w = {k.replace("block", "unknown"): v for k, v in self.weights().items()}
        with self.assertRaisesRegex(ValueError, "layer not found"):
            self.attach(w)

    def test_bundle_integrity_checks_payload_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); payload = root / "ref.png"; payload.write_bytes(b"fixture")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": {"ref.png": runner.sha(payload)}}))
            runner.verified_bundle(root)
            payload.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "validation"):
                runner.verified_bundle(root)
            manifest.write_text(json.dumps({"files": {"../escape": "bad"}}))
            with self.assertRaisesRegex(ValueError, "validation"):
                runner.verified_bundle(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
