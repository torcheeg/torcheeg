"""Independent features, train-only scaling and runnable HSLT export."""

import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.signal import butter, lfilter
import torch

from torcheeg.models import HSLT


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_deap_hslt.py'
spec = importlib.util.spec_from_file_location('deap_hslt_example', EXAMPLE)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class TestDEAPHSLT(unittest.TestCase):
    def test_independent_band_features_channel_order_and_input(self):
        eeg = np.random.default_rng(42).normal(size=(32, 1000))
        actual = workflow.prepare_features(eeg, np.zeros((32, 128)))['eeg']
        expected = []
        for low, high in workflow.BANDS.values():
            b, a = butter(5, [low / 64, high / 64], btype='band')
            filtered = lfilter(b, a, eeg[:, :768], axis=-1)
            expected.append(0.5 * np.log2(2 * np.pi * np.e * filtered.var(-1)))
        expected = np.stack(expected, -1)[workflow.CHANNEL_INDICES].astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(set(workflow.CHANNEL_ORDER)), 32)
        for x in [np.zeros((31, 768)), np.zeros((32, 767)), np.full((32, 768), np.nan)]:
            with self.assertRaises(ValueError):
                workflow.prepare_features(x, None)

    def test_only_training_features_fit_statistics(self):
        train = torch.arange(4 * 32 * 4, dtype=torch.float32).reshape(4, 32, 4)
        test = torch.full((2, 32, 4), 10000.0)
        scaled, held_out, mean, scale = workflow.standardize_from_train(train, test)
        torch.testing.assert_close(mean, train.mean(0), rtol=0, atol=0)
        torch.testing.assert_close(scale, train.std(0, unbiased=False), rtol=0, atol=0)
        torch.testing.assert_close(
            scaled.mean(0), torch.zeros_like(mean), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(held_out, (test - mean) / scale, rtol=0, atol=0)

    def test_complete_workflow_and_atomic_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            output = root / 'model.pt'
            rng = np.random.default_rng(19)
            for name in ['s01.dat', 's02.dat', 's03.dat']:
                with (data / name).open('wb') as stream:
                    pickle.dump(
                        dict(
                            data=rng.normal(size=(2, 40, 1152)),
                            labels=np.array(
                                [[4.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]]
                            ),
                        ),
                        stream,
                    )
            command = [
                sys.executable,
                str(EXAMPLE),
                '--root-path',
                str(data),
                '--output-path',
                str(output),
                '--epochs',
                '1',
                '--batch-size',
                '2',
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint = torch.load(output, map_location='cpu', weights_only=True)
            meta = json.loads(checkpoint['metadata_json'])
            config = json.loads(checkpoint['config_json'])
            self.assertEqual(meta['train_subjects'], ['s01.dat', 's02.dat'])
            self.assertEqual(meta['held_out_subject'], 's03.dat')
            self.assertEqual(
                (meta['train_windows'], meta['test_windows'], meta['updates']),
                (4, 2, 2),
            )
            self.assertEqual(meta['label_rule'], 'valence_ge_5')
            self.assertEqual(checkpoint['test_targets'].tolist(), [0, 1])
            self.assertEqual(checkpoint['feature_mean'].shape, (32, 4))
            self.assertEqual(checkpoint['feature_scale'].shape, (32, 4))
            model = HSLT(**config)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            before = output.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(before, output.read_bytes())
