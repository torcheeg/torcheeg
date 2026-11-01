import unittest

import torch

from torcheeg.models import BGlow


class TestBGlow(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        model = BGlow()
        # forward to calculate loss function
        mock_eeg = torch.randn(2, 4, 32, 32)
        z, nll_loss = model(mock_eeg)
        loss = nll_loss.mean()
        self.assertTrue(torch.isfinite(loss))

        # sample a generated result
        fake_X = model.sample(32, temperature=1.0)
        for output in fake_X if isinstance(fake_X, (tuple, list)) else (fake_X,):
            self.assertEqual(output.device.type, 'cpu')
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(fake_X.shape), (32, 4, 32, 32))


if __name__ == '__main__':
    unittest.main()
