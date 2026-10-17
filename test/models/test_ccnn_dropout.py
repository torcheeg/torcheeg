import io
import unittest
import warnings

import torch
import torch.nn.functional as F

from torcheeg.models import CCNN


class TestCCNNDropout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.eeg = torch.randn(4, 4, 3, 5)

    def model(self, dropout=0.5):
        return CCNN(in_channels=4, grid_size=(3, 5), num_classes=3,
                    dropout=dropout)

    def test_training_without_spatial_dropout_warning(self):
        model = self.model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = model.conv1[1].weight.detach().clone()
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='dropout2d:')
            loss = F.cross_entropy(model(self.eeg), torch.tensor([0, 1, 2, 0]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertFalse(torch.equal(before, model.conv1[1].weight))

    def test_zero_probability_preserves_features(self):
        model = self.model(dropout=0.0)
        training = model(self.eeg)
        model.eval()
        torch.testing.assert_close(training, model(self.eeg), rtol=0, atol=0)

    def test_full_dropout_returns_classifier_bias(self):
        model = self.model(dropout=1.0)
        expected = model.lin2.bias.unsqueeze(0).expand(4, -1)
        torch.testing.assert_close(model(self.eeg), expected)

    def test_training_is_stochastic_and_evaluation_is_repeatable(self):
        model = self.model()
        first = model(self.eeg)
        second = model(self.eeg)
        self.assertFalse(torch.equal(first, second))
        model.eval()
        torch.testing.assert_close(model(self.eeg), model(self.eeg),
                                   rtol=0, atol=0)

    def test_checkpoint_round_trip(self):
        model = self.model().eval()
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = self.model().eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True),
                                 strict=True)
        torch.testing.assert_close(model(self.eeg), restored(self.eeg),
                                   rtol=0, atol=0)
        self.assertFalse(any('lin1.2' in key for key in model.state_dict()))

    def test_output_is_unbounded_logits(self):
        model = self.model().eval()
        with torch.no_grad():
            model.lin2.weight.zero_()
            model.lin2.bias.copy_(torch.tensor([-2.0, 0.0, 3.0]))
        expected = torch.tensor([-2.0, 0.0, 3.0]).expand(4, -1)
        torch.testing.assert_close(model(self.eeg), expected)


if __name__ == '__main__':
    unittest.main()
