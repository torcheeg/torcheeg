"""Functional workflow tests use an independent small protocol fixture."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torcheeg.datasets import MentalArithmeticDataset
from test.models.test_unidmfm import ProtocolBackend


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_unidmfm.py'


def load_example():
    spec = importlib.util.spec_from_file_location('unidmfm_workflow_fixture', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Require 200 Hz.')
    samples = 1600 if record[1] == 0 else 800
    return dict(
        eeg_data=np.random.RandomState(42 + record[1]).normal(size=(19, samples)),
        label=record[1],
        record=record,
    )


class TestUniDMFMExample(unittest.TestCase):
    def test_sensor_order_normalization_and_invalid_inputs(self):
        module = load_example()
        values = np.random.RandomState(8).normal(size=(19, 800))
        expected = values.copy()
        expected = (
            (expected - expected.mean(-1, keepdims=True))
            / expected.std(-1, keepdims=True)
        ).astype(np.float32)
        np.testing.assert_array_equal(module.prepare_window(values), expected)
        self.assertEqual(
            module.CHANNELS,
            [
                'Fp1',
                'Fp2',
                'F3',
                'F4',
                'F7',
                'F8',
                'T3',
                'T4',
                'C3',
                'C4',
                'T5',
                'T6',
                'P3',
                'P4',
                'O1',
                'O2',
                'Fz',
                'Cz',
                'Pz',
            ],
        )
        np.testing.assert_array_equal(
            module.prepare_window(np.zeros((19, 800))), np.zeros((19, 800))
        )
        for bad in [np.zeros((18, 800)), np.full((19, 800), np.inf)]:
            with self.assertRaises(ValueError):
                module.prepare_window(bad)

    def test_complete_workflow_frozen_state_and_strict_restore(self):
        module = load_example()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for subject in range(2):
                for condition in [1, 2]:
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(
                        b'fixture only'
                    )
            backend = ProtocolBackend()
            state = {k: v.clone() for k, v in backend.state_dict().items()}
            checkpoint = root / 'weights.pt'
            output = root / 'probe.pt'
            torch.save({'model': state}, checkpoint)

            def factory(backend_root, checkpoint_path, channel_names):
                return ProtocolBackend(), {'protocol_fixture': True}

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
                    patch.object(module, 'build_unidmfm_backend', side_effect=factory)
                )
                stack.enter_context(
                    patch.object(
                        MentalArithmeticDataset, 'read_record', side_effect=fake_record
                    )
                )
                stack.enter_context(patch('sys.argv', args))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                module.main()
            saved = torch.load(output, weights_only=True)
            meta = json.loads(saved['metadata_json'])
            self.assertEqual(meta['held_out_subject'], 'Subject01')
            self.assertEqual(meta['train_subjects'], ['Subject00'])
            self.assertEqual(
                (meta['train_windows'], meta['test_windows'], meta['updates']),
                (3, 3, 2),
            )
            self.assertEqual(
                (meta['sampling_rate'], meta['window_samples']), (200, 800)
            )
            self.assertEqual(meta['channel_order'], module.CHANNELS)
            self.assertEqual(saved['test_logits'].shape, (3, 2))
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


if __name__ == '__main__':
    unittest.main()
