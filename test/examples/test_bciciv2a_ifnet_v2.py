import hashlib
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

from torcheeg.models import IFNetV2


class TestBCICIV2aIFNetV2(unittest.TestCase):
    def test_synthetic_mat_training_safe_export_and_no_overwrite(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = folder / 'data'
            data.mkdir()
            generator = np.random.default_rng(32)
            for subject in ['A01', 'A02']:
                runs = np.empty((1, 1), dtype=object)
                runs[0, 0] = dict(
                    X=generator.normal(size=(8000, 25)),
                    trial=np.array([1, 2001, 4001, 6001])[:, None],
                    y=np.array([1, 2, 3, 4])[:, None],
                    fs=250,
                    classes=np.array(['left', 'right', 'feet', 'tongue']),
                    artifacts=np.zeros((4, 1)),
                )
                savemat(data / (subject + 'T.mat'), dict(data=runs))
            output = folder / 'model.pt'
            command = [
                sys.executable,
                str(root / 'examples/examples_bciciv2a_ifnet_v2.py'),
                '--root-path',
                str(data),
                '--epochs',
                '1',
                '--batch-size',
                '2',
                '--output-path',
                str(output),
            ]
            env = dict(
                os.environ, OMP_NUM_THREADS='2', MPLBACKEND='Agg', PYTHONPATH=str(root)
            )
            result = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=120
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = torch.load(output, weights_only=True, map_location='cpu')
            metadata = json.loads(payload['metadata_json'])
            self.assertEqual(metadata['train_subjects'], ['A01'])
            self.assertEqual(metadata['held_out_subject'], 'A02')
            self.assertEqual(metadata['train_windows'], 4)
            self.assertEqual(metadata['test_windows'], 4)
            self.assertEqual(metadata['updates'], 2)
            self.assertEqual(payload['test_logits'].shape, (4, 4))
            self.assertTrue(torch.isfinite(payload['test_logits']).all())
            self.assertEqual(set(payload['test_targets'].tolist()), {0, 1, 2, 3})
            model = IFNetV2(**json.loads(payload['config_json']))
            model.load_state_dict(payload['state_dict'], strict=True)
            self.assertEqual(
                len(payload['optimizer']['state']), len(list(model.parameters()))
            )
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            again = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=120
            )
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('output-path exists', again.stderr)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)

    def test_cli_help_does_not_require_data(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                str(root / 'examples/examples_bciciv2a_ifnet_v2.py'),
                '--help',
            ],
            env=dict(os.environ, PYTHONPATH=str(root)),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--root-path', result.stdout)


if __name__ == '__main__':
    unittest.main()
