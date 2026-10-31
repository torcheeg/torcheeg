"""LGGNet metadata queries must preserve learned state and random streams."""

import copy
import unittest

import torch

from torcheeg.datasets.constants import DEAP_GENERAL_REGION_LIST
from torcheeg.models import LGGNet


class TestLGGNet(unittest.TestCase):
    def test_feature_dim_preserves_training_and_eval_state(self):
        for training in (True, False):
            with self.subTest(training=training):
                model = LGGNet(DEAP_GENERAL_REGION_LIST).train(training)
                before = copy.deepcopy(model.state_dict())
                rng = torch.get_rng_state().clone()
                for _ in range(3):
                    self.assertEqual(model.feature_dim, 1856)
                self.assertEqual(model.training, training)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_feature_dim_after_dtype_conversion_and_checkpoint_restore(self):
        model = LGGNet(DEAP_GENERAL_REGION_LIST).double().eval()
        state = copy.deepcopy(model.state_dict())
        restored = LGGNet(DEAP_GENERAL_REGION_LIST).double().eval()
        restored.load_state_dict(state, strict=True)
        rng = torch.get_rng_state().clone()
        self.assertEqual(model.feature_dim, restored.feature_dim)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        eeg = torch.randn(2, 1, 32, 128, dtype=torch.float64)
        with torch.no_grad():
            torch.testing.assert_close(model(eeg), restored(eeg), rtol=0, atol=0)
        self.assertEqual(set(state), set(restored.state_dict()))

    def test_query_does_not_change_dropout_training_update(self):
        torch.manual_seed(73)
        model = LGGNet(DEAP_GENERAL_REGION_LIST)
        untouched = copy.deepcopy(model)
        eeg = torch.randn(3, 1, 32, 128)
        target = torch.tensor([0, 1, 0])
        rng = torch.get_rng_state().clone()
        optimizers = [torch.optim.Adam(m.parameters()) for m in (model, untouched)]
        for current, optimizer in zip((model, untouched), optimizers):
            torch.set_rng_state(rng)
            if current is model:
                self.assertEqual(current.feature_dim, 1856)
            loss = torch.nn.functional.cross_entropy(current(eeg), target)
            loss.backward()
            optimizer.step()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, untouched.state_dict()[name], rtol=0, atol=0
            )


if __name__ == '__main__':
    unittest.main()
