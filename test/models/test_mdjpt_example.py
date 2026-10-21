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
from test.models.test_mdjpt import ProtocolBackend


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_mdjpt.py'


def load_example():
    spec = importlib.util.spec_from_file_location('mdjpt_workflow_fixture', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_record(record, sampling_rate, **kwargs):
    if sampling_rate != 125:
        raise ValueError('Require 125 Hz.')
    samples = 2500 if record[1] == 0 else 1250
    return dict(
        eeg_data=np.random.RandomState(42 + record[1]).normal(size=(19, samples)),
        label=record[1],
        record=record,
    )


class TestMDJPTExample(unittest.TestCase):
    def test_sensor_order_normalization_and_invalid_inputs(self):
        module = load_example()
        values = np.random.RandomState(8).normal(size=(19, 625))
        expected = values.copy()
        expected = (
            (expected - expected.mean(-1, keepdims=True))
            / expected.std(-1, keepdims=True).clip(min=1e-6)
        ).astype(np.float32)
        np.testing.assert_array_equal(module.prepare_window(values), expected)
        self.assertEqual(
            module.ALIASES, {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
        )
        recording = np.random.RandomState(2).normal(size=(19, 2500))
        filtered = module.filter_recording(recording)
        np.testing.assert_allclose(filtered.mean(0), 0, rtol=0, atol=1e-12)
        common = np.sin(2 * np.pi * 10 * np.arange(2500) / 125)
        np.testing.assert_allclose(
            module.filter_recording(recording + common), filtered, rtol=0, atol=1e-12
        )
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
            module.prepare_window(np.zeros((19, 625))), np.zeros((19, 625))
        )
        for bad in [np.zeros((18, 625)), np.full((19, 625), np.inf)]:
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
            torch.save({'state_dict': state}, checkpoint)

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
                    patch.object(module, 'build_mdjpt_backend', side_effect=factory)
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
                (6, 6, 3),
            )
            self.assertEqual(
                (meta['sampling_rate'], meta['window_samples']), (125, 625)
            )
            self.assertEqual(meta['channel_order'], module.CHANNELS)
            self.assertEqual(saved['test_logits'].shape, (6, 2))
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
