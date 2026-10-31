"""Preserve the complete published v1.1.3 NN, including legacy state keys."""

import hashlib
import importlib.util
import os
from pathlib import Path
import unittest

import torch

from torcheeg.datasets.constants import DEAP_GENERAL_REGION_LIST
from torcheeg.models import LGGNet


@unittest.skipUnless(
    os.environ.get('TORCHEEG_LGGNET_REFERENCE'),
    'Complete published v1.1.3 source supplied separately.',
)
class TestLGGNetNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ['TORCHEEG_LGGNET_REFERENCE'])
        cls.digest = 'f520b427e2ae543e306cd88edc676d0cf0459e095bf63388500276e113d6575e'
        if hashlib.sha256(cls.path.read_bytes()).hexdigest() != cls.digest:
            raise AssertionError('Complete unchanged v1.1.3 source required.')
        spec = importlib.util.spec_from_file_location('published_lggnet_v113', cls.path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        if hashlib.sha256(cls.path.read_bytes()).hexdigest() != cls.digest:
            raise AssertionError('Published reference file changed.')

    def test_complete_initialization_outputs_gradients_and_adam_update(self):
        for chunk_size in (128, 768):
            with self.subTest(chunk_size=chunk_size):
                torch.manual_seed(23)
                native = self.module.LGGNet(
                    DEAP_GENERAL_REGION_LIST, chunk_size=chunk_size
                )
                rng = torch.get_rng_state().clone()
                torch.manual_seed(23)
                model = LGGNet(DEAP_GENERAL_REGION_LIST, chunk_size=chunk_size)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(set(model.state_dict()), set(native.state_dict()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value, native.state_dict()[name], rtol=0, atol=0
                    )
                model.load_state_dict(native.state_dict(), strict=True)
                eeg = torch.randn(2, 1, 32, chunk_size)
                target = torch.tensor([0, 1])
                rng = torch.get_rng_state().clone()
                results = []
                gradients = []
                for current in (native, model):
                    torch.set_rng_state(rng)
                    if current is model:
                        self.assertEqual(
                            current.feature_dim, current.local_filter_weight.shape[-1]
                        )
                    value = eeg.clone().requires_grad_()
                    optimizer = torch.optim.Adam(current.parameters(), lr=1e-3)
                    logits = current(value)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    loss.backward()
                    gradients.append(
                        {
                            name: None if p.grad is None else p.grad.clone()
                            for name, p in current.named_parameters()
                        }
                    )
                    results.append((logits.detach(), loss.detach(), value.grad.clone()))
                    optimizer.step()
                for left, right in zip(*results):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for name, gradient in gradients[0].items():
                    if gradient is None:
                        self.assertIsNone(gradients[1][name], name)
                    else:
                        torch.testing.assert_close(
                            gradient, gradients[1][name], rtol=0, atol=0
                        )
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value, native.state_dict()[name], rtol=0, atol=0
                    )
                native.eval()
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(eeg), native(eeg), rtol=0, atol=0)
