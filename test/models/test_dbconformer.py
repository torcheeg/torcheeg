"""DBConformer rejects invalid configurations before executing external code."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from torcheeg.models import DBConformer


class TestDBConformerContract(unittest.TestCase):
    def test_invalid_configuration_precedes_source_access(self):
        for config in (
            dict(num_electrodes=0),
            dict(chunk_size=True),
            dict(num_classes=-1),
            dict(hid_channels=32),
            dict(temporal_depth=0),
            dict(spatial_depth=-1),
            dict(spatial_channels=0),
            dict(patch_size=0),
            dict(fusion='sum'),
            dict(chunk_size=1000),
            dict(chunk_size=1),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    DBConformer.from_source('absent.py', **config)
        for config in (dict(positional_embedding=1), dict(drop_last_sample=0)):
            with self.assertRaises(TypeError):
                DBConformer.from_source('absent.py', **config)

    def test_missing_or_modified_source_is_not_executed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'DBConformer.py'
            with self.assertRaises(FileNotFoundError):
                DBConformer.from_source(path)
            path.write_text('raise RuntimeError("untrusted source executed")')
            rng = torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                DBConformer.from_source(path)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_incomplete_backend_is_rejected(self):
        with self.assertRaises(TypeError):
            DBConformer(object())
        with self.assertRaisesRegex(ValueError, 'complete author'):
            DBConformer(nn.Linear(2, 4))

    def test_failed_execution_restores_flags_and_does_not_cache_module(self):
        # Only test loader cleanup; this fixture does not implement a model.
        source = b'import torch\ntorch.backends.cudnn.benchmark = False\ntorch.backends.cudnn.deterministic = True\nraise RuntimeError("loader cleanup fixture")\n'
        digest = hashlib.sha256(source).hexdigest()
        old = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fixture.py'
            path.write_bytes(source)
            try:
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (
                    True,
                    False,
                )
                with mock.patch.object(DBConformer, 'SOURCE_SHA256', digest):
                    for _ in range(2):
                        with self.assertRaisesRegex(
                            RuntimeError, 'loader cleanup fixture'
                        ):
                            DBConformer.from_source(path)
                self.assertTrue(torch.backends.cudnn.benchmark)
                self.assertFalse(torch.backends.cudnn.deterministic)
                self.assertNotIn('_torcheeg_dbconformer_author_' + digest, sys.modules)
            finally:
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old


if __name__ == '__main__':
    unittest.main()
