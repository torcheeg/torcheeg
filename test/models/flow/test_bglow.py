import unittest

import torch

from torcheeg.models import BGlow


class TestBglow(unittest.TestCase):
    def test_bglow(self):
        model = BGlow()
        # forward to calculate loss function
        mock_eeg = torch.randn(2, 4, 32, 32)
        z, nll_loss = model(mock_eeg)
        loss = nll_loss.mean()

        # sample a generated result
        fake_X = model.sample(32, temperature=1.0)
        self.assertEqual(tuple(fake_X.shape), (32, 4, 32, 32))


if __name__ == '__main__':
    unittest.main()
