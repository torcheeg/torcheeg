"""Interface and standalone-workflow guards, with no author code or weights."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from torcheeg.models import NeuroRVQTokenizer
from test.models.test_neurorvq_tokenizer_native import load_example


class TestNeuroRVQTokenizer(unittest.TestCase):
    def test_requires_explicit_complete_backend_and_physical_metadata(self):
        for backend, names, vocab, rate, usage in [
            (None, ['fp1'], ['fp1'], 200, False),
            (nn.Identity(), ['fp1'], ['fp1'], 200, 'yes'),
            (nn.Identity(), ['fp1'], ['fp1'], 256, False),
            (nn.Identity(), ['fp1', 'FP1'], ['fp1'], 200, False),
            (nn.Identity(), ['unknown'], ['fp1'], 200, False),
            (nn.Identity(), ['fp1'], ['fp1'], 200, False),
        ]:
            with self.subTest(rate=rate, usage=usage, names=names):
                with self.assertRaises(ValueError):
                    NeuroRVQTokenizer(backend, names, vocab, rate, usage)

    def test_cache_manifest_checks_source_and_preprocessing(self):
        example = load_example()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'cache'
            marker, config = example.cache_protocol(cache, {'Subject00_1.edf': 'abc'})
            cache.mkdir()
            marker.write_text(json.dumps(config))
            self.assertEqual(
                example.cache_protocol(cache, {'Subject00_1.edf': 'abc'}),
                (marker, config),
            )
            with self.assertRaisesRegex(ValueError, 'differs'):
                example.cache_protocol(cache, {'Subject00_1.edf': 'changed'})
            config['sampling_rate'] = 100
            marker.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'differs'):
                example.cache_protocol(cache, {'Subject00_1.edf': 'abc'})
            marker.unlink()
            (cache / 'existing.csv').write_text('keep this cache')
            with self.assertRaisesRegex(ValueError, 'no verified manifest'):
                example.cache_protocol(cache, {'Subject00_1.edf': 'abc'})
            self.assertEqual((cache / 'existing.csv').read_text(), 'keep this cache')

    def test_window_normalization_and_atomic_safe_tensor_checkpoint(self):
        example = load_example()
        values = np.arange(19 * 800, dtype=np.float64).reshape(19, 800)
        expected = (values - values.mean(-1, keepdims=True)) / values.std(
            -1, keepdims=True
        )
        np.testing.assert_array_equal(
            example.prepare_window(values), expected.astype(np.float32)
        )
        np.testing.assert_array_equal(
            example.prepare_window(np.ones_like(values)),
            np.zeros_like(values, dtype=np.float32),
        )
        for bad in [np.ones((18, 800)), np.full((19, 800), np.nan)]:
            with self.assertRaises(ValueError):
                example.prepare_window(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.pt'
            example.save_checkpoint(path, {'values': torch.arange(4)})
            actual = torch.load(path, weights_only=True, map_location='cpu')
            torch.testing.assert_close(actual['values'], torch.arange(4))
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
