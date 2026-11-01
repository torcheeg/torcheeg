import unittest

import torch
import torch.nn.functional as F

from torcheeg.models import BCGlow


class TestBCGlow(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        model = BCGlow(num_classes=2)
        # forward to calculate loss function
        mock_eeg = torch.randn(2, 4, 32, 32)
        y = torch.randint(0, 2, (2, ))

        z, nll_loss, y_logits = model(mock_eeg, y)
        loss = nll_loss.mean() + F.cross_entropy(y_logits, y)
        self.assertTrue(torch.isfinite(loss))

        # sample a generated result
        fake_X = model.sample(y, temperature=1.0)
        for output in fake_X if isinstance(fake_X, (tuple, list)) else (fake_X,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(fake_X.shape), (2, 4, 32, 32))


if __name__ == '__main__':
    unittest.main()
