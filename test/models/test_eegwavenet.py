"""EEGWaveNet interface checks without substituting a small model."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from torcheeg.models import EEGWaveNet


class TestEEGWaveNetContract(unittest.TestCase):
    def test_invalid_configuration_precedes_source_access(self):
        for config in (
            dict(num_electrodes=0),
            dict(num_classes=-1),
            dict(chunk_size=True),
            dict(chunk_size=447),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    EEGWaveNet.from_source('absent.py', **config)

    def test_missing_and_modified_source_is_not_executed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'architecture.py'
            with self.assertRaises(FileNotFoundError):
                EEGWaveNet.from_source(path)
            path.write_text('raise RuntimeError("untrusted code executed")')
            rng = torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                EEGWaveNet.from_source(path)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_incomplete_backend_is_rejected(self):
        with self.assertRaises(TypeError):
            EEGWaveNet(object())
        with self.assertRaises(ValueError):
            EEGWaveNet(nn.Linear(2, 4))


if __name__ == '__main__':
    unittest.main()
