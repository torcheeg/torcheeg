"""ADFCNN interface checks without substituting a small model."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from torcheeg.models import ADFCNN


class TestADFCNNContract(unittest.TestCase):
    def test_invalid_configuration_precedes_source_access(self):
        for config in (
            dict(num_electrodes=0),
            dict(num_classes=-1),
            dict(chunk_size=True),
            dict(chunk_size=749),
            dict(chunk_size=768),
            dict(chunk_size=1000),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    ADFCNN.from_source('absent.py', **config)

    def test_missing_and_modified_source_is_not_executed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                ADFCNN.from_source(path)
            (path / 'ADFCNN.py').write_text(
                'raise RuntimeError("untrusted code executed")'
            )
            (path / 'layers.py').write_text(
                'raise RuntimeError("untrusted code executed")'
            )
            rng = torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                ADFCNN.from_source(path)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_incomplete_backend_is_rejected(self):
        with self.assertRaises(TypeError):
            ADFCNN(object())
        with self.assertRaises(ValueError):
            ADFCNN(nn.Linear(2, 4))


if __name__ == '__main__':
    unittest.main()
