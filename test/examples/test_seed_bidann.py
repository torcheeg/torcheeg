"""Check trial boundaries, source-only normalization and complete UDA export."""

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

from torcheeg.models import BiDANN
from torcheeg.models.rnn.bidann import _SEED_CHANNELS


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/examples_seed_bidann.py'
spec = importlib.util.spec_from_file_location('seed_bidann_example', EXAMPLE)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class TestSEEDBiDANN(unittest.TestCase):
    def test_normalization_fits_only_source_frames(self):
        source = torch.arange(3 * 9 * 62 * 5, dtype=torch.float32).reshape(3, 9, 62, 5)
        target = torch.full((2, 9, 62, 5), 100000.0)
        x, y, mean, scale = workflow.standardize_from_source(source, target)
        torch.testing.assert_close(mean, source.mean((0, 1)), rtol=0, atol=0)
        torch.testing.assert_close(
            scale, source.std((0, 1), unbiased=False), rtol=0, atol=0
        )
        torch.testing.assert_close(
            x.mean((0, 1)), torch.zeros_like(mean), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(y, (target - mean) / scale, rtol=0, atol=0)

    def test_synthetic_provider_features_full_training_and_safe_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            rng = np.random.default_rng(23)
            savemat(data / 'label.mat', dict(label=np.array([[-1, 0, 1]])))
            source_values = []
            for subject in [1, 2]:
                record = {}
                for trial in [1, 2, 3]:
                    values = rng.normal(size=(62, 12, 5)).astype(np.float32)
                    record['de_LDS' + str(trial)] = values
                    record['de_movingAve' + str(trial)] = values + 10
                    if subject == 1:
                        source_values.append(
                            torch.from_numpy(values[:, :9, :].transpose(1, 0, 2))
                        )
                savemat(data / (str(subject) + '_20131111.mat'), record)
            output = root / 'model.pt'
            command = [
                sys.executable,
                str(EXAMPLE),
                '--root-path',
                str(data),
                '--epochs',
                '1',
                '--batch-size',
                '3',
                '--output-path',
                str(output),
            ]
            result = subprocess.run(
                command,
                env=dict(
                    os.environ,
                    PYTHONPATH=str(ROOT),
                    OMP_NUM_THREADS='2',
                    MPLBACKEND='Agg',
                ),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = torch.load(output, map_location='cpu', weights_only=True)
            meta = json.loads(payload['metadata_json'])
            config = json.loads(payload['config_json'])
            self.assertEqual(meta['source_subjects'], [1])
            self.assertEqual(meta['target_subject'], 2)
            self.assertEqual(
                (meta['source_sequences'], meta['target_sequences'], meta['updates']),
                (3, 3, 1),
            )
            self.assertTrue(meta['unlabeled_target_features_used_in_training'])
            self.assertFalse(meta['target_emotion_labels_used_in_training'])
            self.assertEqual(meta['channel_names'], _SEED_CHANNELS)
            self.assertEqual(meta['feature'], 'de_LDS')
            self.assertEqual(
                meta['criterion'], 'source_CE_plus_left_right_global_domain_CEs'
            )
            self.assertEqual(
                meta['source_trial_ids'],
                [[1, 20131111, 0, 1], [1, 20131111, 0, 2], [1, 20131111, 0, 3]],
            )
            self.assertEqual(payload['test_targets'].tolist(), [0, 1, 2])
            self.assertEqual(payload['test_logits'].shape, (3, 3))
            source = torch.stack(source_values)
            torch.testing.assert_close(
                payload['feature_mean'], source.mean((0, 1)), rtol=0, atol=0
            )
            torch.testing.assert_close(
                payload['feature_scale'],
                source.std((0, 1), unbiased=False).clamp_min(1e-6),
                rtol=0,
                atol=0,
            )
            model = BiDANN(**config)
            model.load_state_dict(payload['state_dict'], strict=True)
            self.assertEqual(
                len(payload['optimizer']['state']), len(list(model.parameters()))
            )
            before = output.read_bytes()
            again = subprocess.run(
                command,
                env=dict(os.environ, PYTHONPATH=str(ROOT)),
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(before, output.read_bytes())
