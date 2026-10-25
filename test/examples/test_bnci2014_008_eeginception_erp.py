import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat
import torch

from torcheeg.models import EEGInceptionERP


class TestBNCI2014008EEGInceptionERP(unittest.TestCase):
    @staticmethod
    def example():
        root = Path(__file__).resolve().parents[2]
        path = root / 'examples/examples_bnci2014_008_eeginception_erp.py'
        spec = importlib.util.spec_from_file_location('erp_example', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return root, path, module

    @staticmethod
    def make_subject(path, offset=0.0):
        generator = np.random.default_rng(23)
        eeg = generator.normal(size=(4096, 8)) * (1 + offset / 10) + offset
        markers = np.zeros(4096, dtype=np.int64)
        for onset, label in zip([1024, 1536, 2048, 2560], [1, 2, 1, 2]):
            markers[onset : onset + 32] = label
        savemat(
            path,
            {
                'data': {
                    'X': eeg,
                    'y': markers,
                    'channels': np.array(
                        ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8'], dtype=object
                    ),
                    'classes': np.array(['NonTarget', 'Target'], dtype=object),
                }
            },
        )

    def test_marker_edges_epoch_labels_and_channel_validation(self):
        _, _, example = self.example()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'A01.mat'
            self.make_subject(path)
            epochs, labels, onsets = example.load_subject(path)
            self.assertEqual(epochs.shape, (4, 8, 128))
            np.testing.assert_array_equal(labels, [0, 1, 0, 1])
            np.testing.assert_array_equal(onsets, [1024, 1536, 2048, 2560])
            self.assertTrue(np.isfinite(epochs).all())
            self.assertLess(np.abs(epochs.mean(axis=1)).max(), 1e-6)
            savemat(
                path,
                {
                    'data': {
                        'X': np.zeros((1024, 8)),
                        'y': np.zeros(1024),
                        'channels': ['wrong'] * 8,
                        'classes': np.array(['NonTarget', 'Target'], dtype=object),
                    }
                },
            )
            with self.assertRaisesRegex(ValueError, 'eight-channel'):
                example.load_subject(path)

    def test_synthetic_training_train_only_normalization_and_safe_export(self):
        root, script, example = self.example()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for subject, offset in [('A01', 0), ('A02', 100)]:
                self.make_subject(folder / (subject + '.mat'), offset)
            output = folder / 'erp.pt'
            hashes = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in folder.glob('*.mat')
            }
            command = [
                sys.executable,
                str(script),
                '--root-path',
                str(folder),
                '--epochs',
                '1',
                '--batch-size',
                '2',
                '--output-path',
                str(output),
            ]
            env = dict(
                os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='2', MPLBACKEND='Agg'
            )
            result = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=120
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = torch.load(output, weights_only=True, map_location='cpu')
            metadata = json.loads(payload['metadata_json'])
            self.assertEqual(metadata['train_subjects'], ['A01'])
            self.assertEqual(metadata['test_subject'], 'A02')
            self.assertEqual(metadata['train_epochs'], 4)
            self.assertEqual(metadata['test_epochs'], 4)
            self.assertEqual(metadata['updates'], 2)
            self.assertEqual(metadata['source_sha256'], hashes)
            train_x, _, _ = example.load_subject(folder / 'A01.mat')
            np.testing.assert_allclose(
                payload['normalization_mean'].numpy(),
                train_x.mean(axis=(0, 2), keepdims=True, dtype=np.float64),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                payload['normalization_std'].numpy(),
                train_x.std(axis=(0, 2), keepdims=True, dtype=np.float64),
                rtol=1e-6,
                atol=1e-6,
            )
            model = EEGInceptionERP(**json.loads(payload['config_json']))
            model.load_state_dict(payload['model'], strict=True)
            self.assertEqual(
                len(payload['optimizer']['state']), len(list(model.parameters()))
            )
            test_x, labels, onsets = example.load_subject(folder / 'A02.mat')
            normalized = (
                torch.from_numpy(test_x) - payload['normalization_mean']
            ) / payload['normalization_std']
            model.eval()
            torch.set_num_threads(2)
            with torch.no_grad():
                actual = torch.cat([model(x) for x in normalized.split(2)])
            torch.testing.assert_close(actual, payload['test_logits'], rtol=0, atol=0)
            np.testing.assert_array_equal(payload['test_targets'].numpy(), labels)
            np.testing.assert_array_equal(payload['test_onsets'].numpy(), onsets)
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            again = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=120
            )
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('output-path exists', again.stderr)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)
            self.assertEqual(
                {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in folder.glob('*.mat')
                },
                hashes,
            )

    def test_cli_help_without_data(self):
        root, script, _ = self.example()
        result = subprocess.run(
            [sys.executable, str(script), '--help'],
            env=dict(os.environ, PYTHONPATH=str(root)),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--root-path', result.stdout)


if __name__ == '__main__':
    unittest.main()
