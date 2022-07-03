import unittest

import torch
import torch.nn as nn

from torcheeg.losses import FocalLoss, VirtualAdversarialTrainingLoss


class TestLosses(unittest.TestCase):
    def test_focal_loss(self):
        inputs = torch.randn(3, 5, requires_grad=True)
        targets = torch.empty(3, dtype=torch.long).random_(5)
        loss = FocalLoss(num_classes=5)
        output = loss(inputs, targets)
        output.backward()
        self.assertEqual(tuple(output.shape), ())

    def test_virtual_adversarial_training_loss(self):
        inputs = torch.randn(3, 5, requires_grad=True)
        model = nn.Linear(5, 5)
        loss = VirtualAdversarialTrainingLoss()
        output = loss(model, inputs)
        output.backward()
        self.assertEqual(tuple(output.shape), ())


if __name__ == '__main__':
    unittest.main()
