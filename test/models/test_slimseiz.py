"""Portable contracts for the complete external SlimSeiz network."""

from pathlib import Path
import tempfile
import unittest
import torch
from torcheeg.models import SlimSeiz


class TestSlimSeiz(unittest.TestCase):
    def test_validate_before_source_access(self):
        for config in [
            dict(num_electrodes=0),
            dict(num_electrodes=True),
            dict(chunk_size=31),
            dict(chunk_size=0),
            dict(chunk_size=1.5),
            dict(num_classes=3),
        ]:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    SlimSeiz.from_source('missing.ipynb', **config)
        with self.assertRaises(TypeError):
            SlimSeiz.from_source('missing.ipynb', return_features=1)

    def test_missing_or_modified_source_never_executes(self):
        with self.assertRaises(FileNotFoundError):
            SlimSeiz.from_source('missing.ipynb')
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'changed.ipynb'
            marker = Path(directory) / 'executed'
            p.write_text('open(' + repr(str(marker)) + ',"w").write("bad")')
            rng = torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                SlimSeiz.from_source(p)
            self.assertFalse(marker.exists())
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_reject_incomplete_backend(self):
        with self.assertRaises(TypeError):
            SlimSeiz(None)
        with self.assertRaises(ValueError):
            SlimSeiz(torch.nn.Linear(8, 2))
