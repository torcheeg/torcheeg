import unittest

import torch
import torch.nn.functional as F

from torcheeg.models import BCGlow, BGlow


class TestFlow(unittest.TestCase):
    def test_bglow(self):
        model = BGlow()
        # forward to calculate loss function
        mock_eeg = torch.randn(2, 4, 32, 32)
        z, nll_loss = model(mock_eeg)
        loss = nll_loss.mean()

        # sample a generated result
        fake_X = model.sample(32, temperature=1.0)
        self.assertEqual(tuple(fake_X.shape), (32, 4, 32, 32))

    def test_bcglow(self):
        model = BCGlow(num_classes=2)
        # forward to calculate loss function
        mock_eeg = torch.randn(2, 4, 32, 32)
        y = torch.randint(0, 2, (2, ))

        z, nll_loss, y_logits = model(mock_eeg, y)
        loss = nll_loss.mean() + F.cross_entropy(y_logits, y)

        # sample a generated result
        fake_X = model.sample(y, temperature=1.0)
        self.assertEqual(tuple(fake_X.shape), (2, 4, 32, 32))


if __name__ == '__main__':
    unittest.main()
