import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LOCATION_DICT
from torcheeg.models import BDecoder, BEncoder


class TestBVAEExample(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / 'examples' / 'examples_eegmat_bvae.py'
        spec = importlib.util.spec_from_file_location('eegmat_bvae_example', path)
        cls.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.example)

    def test_feature_grid_contains_standardized_electrode_bands(self):
        rng = np.random.default_rng(42)
        time = np.arange(800) / 200.
        signal = np.array([(index + 1) * np.sin(2 * np.pi * 10 * time)
                           + rng.normal(size=800) for index in range(19)])
        features = self.example.feature_transform()(eeg=signal)['eeg']
        self.assertEqual(features.shape, (4, 5, 5))
        self.assertTrue(np.isfinite(features).all())
        values = []
        occupied = np.zeros((5, 5), dtype=bool)
        for row, column in MENTAL_ARITHMETIC_CHANNEL_LOCATION_DICT.values():
            occupied[row, column] = True
            values.append(features[:, row, column])
        values = np.stack(values)
        np.testing.assert_allclose(values.mean(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(values.std(axis=0), 1., atol=1e-12)
        np.testing.assert_array_equal(features[:, ~occupied], 0.)

    def test_training_testing_and_prior_generation_complete(self):
        torch.manual_seed(42)
        train = TensorDataset(torch.randn(4, 4, 5, 5), torch.tensor([0, 1, 0, 1]))
        held_out = TensorDataset(torch.randn(2, 4, 5, 5), torch.tensor([0, 1]))
        torch.manual_seed(70)
        initial_encoder = BEncoder(in_channels=4, grid_size=(5, 5), hid_channels=32)
        initial_decoder = BDecoder(in_channels=32, out_channels=4, grid_size=(5, 5))
        torch.manual_seed(70)
        trainer, generated, metrics = self.example.train_and_evaluate(
            train, held_out, epochs=1, batch_size=2)
        self.assertEqual(generated.shape, (4, 4, 5, 5))
        self.assertTrue(torch.isfinite(generated).all())
        self.assertIn('test_rec_loss', metrics)
        self.assertIn('test_kld_loss', metrics)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertGreater(trainer.encoder.conv1[1].num_batches_tracked.item(), 0)
        for model, initial in ((trainer.encoder, initial_encoder),
                               (trainer.decoder, initial_decoder)):
            self.assertTrue(any(not torch.equal(parameter, original)
                                for parameter, original in zip(model.parameters(), initial.parameters())))
            self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        restored = BDecoder(in_channels=32, out_channels=4, grid_size=(5, 5)).eval()
        restored.load_state_dict(trainer.decoder.state_dict(), strict=True)
        latent = torch.randn(2, 32)
        with torch.no_grad():
            torch.testing.assert_close(restored(latent), trainer.decoder(latent))


if __name__ == '__main__':
    unittest.main()
