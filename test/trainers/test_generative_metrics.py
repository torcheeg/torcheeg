import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.linalg import sqrtm
from torch import nn

from torcheeg.trainers.generative.utils import FrechetInceptionDistance


class EEGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, eeg):
        self.calls += 1
        return eeg.mean(dim=-1)


class TestGenerativeMetrics(unittest.TestCase):
    def setUp(self):
        self.real = torch.tensor([[0., 0.], [2., 0.], [0., 2.], [2., 2.]], dtype=torch.float64)
        self.fake = torch.tensor([[1., 0.], [4., 0.], [1., 1.], [4., 1.]], dtype=torch.float64)

    def make_metric(self, **kwargs):
        return FrechetInceptionDistance(nn.Identity(), 2, **kwargs)

    def test_initialization_uses_only_the_supplied_eeg_extractor(self):
        extractor = EEGFeatureExtractor()
        with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('Image weights requested')):
            metric = FrechetInceptionDistance(extractor, 2)
        self.assertIs(metric.inception, extractor)
        self.assertEqual(extractor.calls, 0)
        metric.update(self.real.unsqueeze(-1), real=True)
        metric.update(self.fake.unsqueeze(-1), real=False)
        self.assertEqual(extractor.calls, 2)
        self.assertTrue(torch.isfinite(metric.compute()))

    def test_distance_matches_independent_numpy_scipy_reference(self):
        metric = self.make_metric()
        metric.update(self.real, real=True)
        metric.update(self.fake, real=False)
        real, fake = self.real.numpy(), self.fake.numpy()
        cov_real, cov_fake = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
        expected = np.square(real.mean(axis=0) - fake.mean(axis=0)).sum()
        expected += np.trace(cov_real + cov_fake - 2 * sqrtm(cov_real @ cov_fake).real)
        self.assertAlmostEqual(metric.compute().item(), expected, places=10)

    def test_equal_distributions_have_zero_distance(self):
        metric = self.make_metric()
        metric.update(self.real, real=True)
        metric.update(self.real, real=False)
        self.assertAlmostEqual(metric.compute().item(), 0., places=10)

    def test_batch_accumulation_matches_single_update(self):
        full, batched = self.make_metric(), self.make_metric()
        for samples, real in ((self.real, True), (self.fake, False)):
            full.update(samples, real=real)
            for batch in samples.split(2):
                batched.update(batch, real=real)
        torch.testing.assert_close(full.compute(), batched.compute())

    def test_reset_clears_both_distributions(self):
        metric = self.make_metric()
        metric.update(self.real, real=True)
        metric.update(self.fake, real=False)
        metric.reset()
        self.assertEqual(metric.real_features_num_samples.item(), 0)
        self.assertEqual(metric.fake_features_num_samples.item(), 0)
        self.assertEqual(metric.real_features_sum.count_nonzero().item(), 0)
        self.assertEqual(metric.fake_features_cov_sum.count_nonzero().item(), 0)

    def test_reset_can_preserve_real_distribution(self):
        metric = self.make_metric(reset_real_features=False)
        metric.update(self.real, real=True)
        metric.update(self.fake, real=False)
        expected = metric.compute()
        metric.reset()
        self.assertEqual(metric.real_features_num_samples.item(), 4)
        self.assertEqual(metric.fake_features_num_samples.item(), 0)
        metric.update(self.fake, real=False)
        torch.testing.assert_close(metric.compute(), expected)

    def test_each_distribution_requires_two_samples(self):
        for real_count, fake_count in ((1, 4), (4, 1)):
            metric = self.make_metric()
            metric.update(self.real[:real_count], real=True)
            metric.update(self.fake[:fake_count], real=False)
            with self.assertRaisesRegex(RuntimeError, 'More than one sample'):
                metric.compute()

    def test_output_dtype_and_metric_flags_are_preserved(self):
        for dtype in (torch.float32, torch.float64):
            metric = self.make_metric()
            metric.update(self.real.to(dtype), real=True)
            metric.update(self.fake.to(dtype), real=False)
            self.assertEqual(metric.compute().dtype, dtype)
            self.assertFalse(metric.higher_is_better)
            self.assertFalse(metric.is_differentiable)
            self.assertFalse(metric.full_state_update)

    def test_reset_option_must_be_boolean(self):
        with self.assertRaises(ValueError):
            self.make_metric(reset_real_features='no')


if __name__ == '__main__':
    unittest.main()
