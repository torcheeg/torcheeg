"""Independent PSD, physical boundaries, source-only statistics and full training."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import torch

from torcheeg.models import SleepTransformer

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / 'examples/examples_sleepedf_sleeptransformer.py'
)
spec = importlib.util.spec_from_file_location(
    'sleepedf_sleeptransformer_example', EXAMPLE
)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class _ScoredEpochs:
    """Small in-memory dataset-reader fixture; the complete NN is never replaced."""

    def __init__(self):
        self.info = pd.DataFrame(
            [
                dict(
                    subject_id=subject,
                    record_id=subject + '1E0',
                    start_at=i * 30.0,
                    end_at=(i + 1) * 30.0,
                    label=list(workflow.STAGES)[i % 5],
                    _record_id=subject,
                    clip_id=str(i),
                )
                for subject in ['SC400', 'SC401', 'SC402']
                for i in range(40)
            ]
        )

    def read_eeg(self, record, clip):
        t = np.arange(3000) / 100
        phase = int(clip) / 7
        return (
            1e-6
            * (np.sin(2 * np.pi * 10 * t + phase) + 0.2 * np.sin(2 * np.pi * 3 * t))
        )[None, :]


class TestSleepEDFSleepTransformer(unittest.TestCase):
    def test_independent_scipy_spectrum_and_physical_units(self):
        eeg = np.random.default_rng(17).normal(size=(1, 3000)) * 1e-6
        frequencies, times, psd = spectrogram(
            eeg[0] * 1e6,
            fs=100,
            window=np.hamming(200),
            nperseg=200,
            noverlap=100,
            nfft=256,
            detrend=False,
            scaling='density',
            mode='psd',
        )
        expected = np.log(np.maximum(psd[1:], 1e-12)).T[..., None].astype(np.float32)
        np.testing.assert_allclose(
            workflow.spectral_features(eeg), expected, rtol=1e-6, atol=1e-6
        )
        self.assertEqual((len(times), len(frequencies) - 1), (29, 128))
        for value in [
            np.zeros((2, 3000)),
            np.zeros((1, 2999)),
            np.full((1, 3000), np.nan),
        ]:
            with self.assertRaises(ValueError):
                workflow.spectral_features(value)

    def test_contexts_preserve_subject_record_and_unscored_gaps(self):
        fixture = _ScoredEpochs()
        fixture.info.loc[fixture.info.subject_id == 'SC400', 'start_at'] += np.where(
            fixture.info[fixture.info.subject_id == 'SC400'].clip_id.astype(int) >= 10,
            30,
            0,
        )
        features, targets, identities = workflow.select_contexts(fixture, 2, 20)
        self.assertEqual(tuple(features.shape), (5, 20, 29, 128, 1))
        self.assertEqual(tuple(targets.shape), (5, 20))
        self.assertEqual(sum(row['subject'] == 'SC400' for row in identities), 1)
        for row in identities:
            np.testing.assert_array_equal(np.diff(row['onsets']), np.full(19, 30))
            self.assertTrue(row['record'].startswith(row['subject']))
        with self.assertRaises(ValueError):
            workflow.select_contexts(fixture, 1, 19)

    def test_training_statistics_exclude_held_out_subject(self):
        train = torch.randn(3, 20, 29, 128, 1)
        held_out = torch.full((1, 20, 29, 128, 1), 1e5)
        scaled, test, mean, scale = workflow.standardize_from_train(train, held_out)
        torch.testing.assert_close(mean, train.mean((0, 1, 2)), rtol=0, atol=0)
        torch.testing.assert_close(
            scale, train.std((0, 1, 2), unbiased=False), rtol=0, atol=0
        )
        torch.testing.assert_close(test, (held_out - mean) / scale, rtol=0, atol=0)
        torch.testing.assert_close(
            scaled.mean((0, 1, 2)), torch.zeros_like(mean), rtol=0, atol=1e-6
        )

    def test_complete_network_training_reload_and_atomic_no_overwrite_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sleep-cassette').mkdir()
            (root / 'sleep-cassette/fixture.edf').write_bytes(
                b'synthetic reader fixture'
            )
            output = root / 'model.pt'
            argv = [
                str(EXAMPLE),
                '--root-path',
                str(root),
                '--output-path',
                str(output),
                '--epochs',
                '1',
                '--max-contexts',
                '1',
                '--batch-size',
                '2',
            ]
            with patch.object(sys, 'argv', argv), patch.object(
                workflow, 'SleepEDFxDataset', return_value=_ScoredEpochs()
            ):
                workflow.main()
            checkpoint = torch.load(output, map_location='cpu', weights_only=True)
            metadata = json.loads(checkpoint['metadata_json'])
            self.assertEqual(metadata['train_subjects'], ['SC400', 'SC401'])
            self.assertEqual(metadata['held_out_subject'], 'SC402')
            self.assertEqual(
                (
                    metadata['train_contexts'],
                    metadata['test_contexts'],
                    metadata['updates'],
                ),
                (2, 1, 1),
            )
            self.assertEqual(len(checkpoint['model']), 65)
            self.assertEqual(len(checkpoint['optimizer']['state']), 65)
            model = SleepTransformer(**json.loads(checkpoint['config_json'])).eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            x = (checkpoint['test_features'] - checkpoint['mean']) / checkpoint['scale']
            with torch.no_grad():
                torch.testing.assert_close(
                    model(x), checkpoint['test_logits'], rtol=0, atol=0
                )
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                workflow.save_checkpoint({}, output)
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
