"""Functional complete workflow with independent synthetic MATLAB fixtures."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
import torch

from test.models.test_neurogpt import CHANNELS, ProtocolBackend

EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_bciciv2a_neurogpt.py'


def load_example():
    spec = importlib.util.spec_from_file_location('neurogpt_workflow_fixture', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNeuroGPTExample(unittest.TestCase):
    def test_channel_time_order_and_normalization(self):
        module = load_example()
        x = np.random.RandomState(3).normal(size=(22, 1000))
        normalized = (
            (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)
        ).astype(np.float32)
        expected = np.stack([normalized[:, :500], normalized[:, 500:]])
        np.testing.assert_array_equal(module.prepare_window(x), expected)
        self.assertEqual(module.CHANNELS, CHANNELS)
        np.testing.assert_array_equal(
            module.prepare_window(np.zeros((22, 1000))), np.zeros((2, 22, 500))
        )
        for bad in [
            np.zeros((25, 1000)),
            np.zeros((22, 999)),
            np.full((22, 1000), float('nan')),
        ]:
            with self.assertRaises(ValueError):
                module.prepare_window(bad)

    def test_complete_workflow_subject_artifact_labels_and_frozen_states(self):
        module = load_example()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            source.mkdir()
            for subject in [1, 2]:
                payload = np.empty((1, 1), dtype=object)
                payload[0, 0] = dict(
                    X=np.random.RandomState(subject).normal(size=(8750, 25)),
                    trial=np.array([1, 1751, 3501, 5251, 7001]).reshape(-1, 1),
                    y=np.array([1, 2, 3, 4, 1]).reshape(-1, 1),
                    fs=250,
                    classes=['left hand', 'right hand', 'feet', 'tongue'],
                    artifacts=np.array([0, 0, 0, 0, 1]).reshape(-1, 1),
                )
                savemat(source / f'A{subject:02d}T.mat', {'data': payload})
            backend = ProtocolBackend()
            state = copy_state(backend)
            checkpoint = root / 'weights.pt'
            torch.save(state, checkpoint)
            output = root / 'probe.pt'
            args = [
                str(EXAMPLE),
                '--root-path',
                str(source),
                '--backend-root',
                str(root / 'author'),
                '--checkpoint',
                str(checkpoint),
                '--cache-path',
                str(root / 'cache'),
                '--output-path',
                str(output),
                '--epochs',
                '1',
                '--batch-size',
                '2',
            ]
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        module,
                        'build_neurogpt_backend',
                        side_effect=lambda *args: (
                            ProtocolBackend(),
                            {'protocol_fixture': True},
                        ),
                    )
                )
                stack.enter_context(patch('sys.argv', args))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                module.main()
            saved = torch.load(output, map_location='cpu', weights_only=True)
            meta = json.loads(saved['metadata_json'])
            self.assertEqual(meta['held_out_subject'], 'A02')
            self.assertEqual(meta['train_subjects'], ['A01'])
            self.assertEqual(
                (meta['train_windows'], meta['test_windows'], meta['updates']),
                (4, 4, 2),
            )
            self.assertEqual(
                (
                    meta['sampling_rate'],
                    meta['cue_offset_samples'],
                    meta['window_samples'],
                ),
                (250, 500, 1000),
            )
            self.assertEqual(meta['channel_order'], CHANNELS)
            self.assertTrue(meta['allow_montage_transfer'])
            self.assertTrue(
                json.loads(saved['backbone_config_json'])['allow_montage_transfer']
            )
            self.assertEqual(sorted(saved['test_targets'].tolist()), [0, 1, 2, 3])
            self.assertEqual(saved['test_logits'].shape, (4, 4))
            for key, value in state.items():
                torch.testing.assert_close(
                    saved['state_dict']['backbone.backend.' + key],
                    value,
                    rtol=0,
                    atol=0,
                )
            self.assertIn('backbone.metadata_fingerprint', saved['state_dict'])
            self.assertEqual(
                meta['metrics']['loss'],
                torch.nn.functional.cross_entropy(
                    saved['test_logits'], saved['test_targets']
                ).item(),
            )


def copy_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


if __name__ == '__main__':
    unittest.main()
