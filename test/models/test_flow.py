import unittest

import torch.nn.functional as F

import torch
from torcheeg.models import BGlow


class TestFlow(unittest.TestCase):
    def test_bglow(self):
        model = BGlow()
        # forward to calculate loss function
        mock_eeg = torch.randn(2, 4, 32, 32)
        z, nll_loss, y_logits = model(mock_eeg)
        loss = nll_loss.mean()

        # sample a generated result
        fake_X = model(temperature=1.0, reverse=True)
        self.assertEqual(tuple(fake_X.shape), (32, 4, 32, 32))

        mock_eeg = mock_eeg.cuda()
        model = model.cuda()

        # forward to calculate loss function
        z, nll_loss, y_logits = model(mock_eeg)

        # sample a generated result
        fake_X = model(temperature=1.0, reverse=True)
        self.assertEqual(tuple(fake_X.shape), (32, 4, 32, 32))

    def test_bcglow(self):
        model = BGlow(num_classes=2)
        # forward to calculate loss function
        mock_eeg = torch.randn(2, 4, 32, 32)
        y = torch.randint(0, 2, (2, ))
        z, nll_loss, y_logits = model(mock_eeg, y)
        loss = F.cross_entropy(y_logits, y) + F.cross_entropy(y_logits, y)

        # sample a generated result
        fake_X = model(y=y, temperature=1.0, reverse=True)
        self.assertEqual(tuple(fake_X.shape), (2, 4, 32, 32))

        model = model.cuda()
        mock_eeg = mock_eeg.cuda()
        y = y.cuda()

        # forward to calculate loss function
        z, nll_loss, y_logits = model(mock_eeg, y)

        # sample a generated result
        fake_X = model(y=y, temperature=1.0, reverse=True)
        self.assertEqual(tuple(fake_X.shape), (2, 4, 32, 32))


if __name__ == '__main__':
    unittest.main()
