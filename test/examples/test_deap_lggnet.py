"""Raw row order, source-only statistics and complete LGGNet export."""

import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from torcheeg.models import LGGNet


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_deap_lggnet.py'
spec = importlib.util.spec_from_file_location('deap_lggnet_example', EXAMPLE)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class TestDEAPLGGNet(unittest.TestCase):
    def test_raw_features_preserve_channel_order_and_ignore_baseline(self):
        eeg = np.arange(32 * 1000, dtype=np.float64).reshape(32, 1000)
        before = eeg.copy()
        actual = workflow.prepare_features(eeg, np.ones((32, 128)) * 999)['eeg']
        np.testing.assert_array_equal(actual, eeg[:, :768].astype(np.float32))
        np.testing.assert_array_equal(eeg, before)
        self.assertEqual(len(set(workflow.CHANNEL_ORDER)), 32)
        self.assertEqual(
            sorted(i for group in workflow.DEAP_GENERAL_REGION_LIST for i in group),
            list(range(32)),
        )
        for x in [np.zeros((31, 768)), np.zeros((32, 767)), np.full((32, 768), np.nan)]:
            with self.assertRaises(ValueError):
                workflow.prepare_features(x, None)

    def test_only_training_features_fit_statistics(self):
        train = torch.arange(4 * 32 * 4, dtype=torch.float32).reshape(4, 32, 4)
        test = torch.full((2, 32, 4), 10000.0)
        scaled, held_out, mean, scale = workflow.standardize_from_train(train, test)
        torch.testing.assert_close(
            mean, train.mean((0, 2)).unsqueeze(1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            scale, train.std((0, 2), unbiased=False).unsqueeze(1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            scaled.mean((0, 2)).unsqueeze(1), torch.zeros_like(mean), atol=1e-6, rtol=0
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
            self.assertEqual(checkpoint['feature_mean'].shape, (32, 1))
            self.assertEqual(checkpoint['feature_scale'].shape, (32, 1))
            model = LGGNet(**config)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval()
            self.assertEqual(model.feature_dim, meta['feature_dim'])
            self.assertTrue(meta['parameters_without_task_gradients'])
            with (data / 's03.dat').open('rb') as stream:
                raw = pickle.load(stream)['data'][:, :32, 384:1152]
            features = torch.tensor(raw, dtype=torch.float32)
            features = (features - checkpoint['feature_mean']) / checkpoint[
                'feature_scale'
            ]
            with torch.no_grad():
                torch.testing.assert_close(
                    model(features.unsqueeze(1)),
                    checkpoint['test_logits'],
                    rtol=0,
                    atol=0,
                )
            before = output.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(before, output.read_bytes())
