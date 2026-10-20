import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torcheeg.datasets import MentalArithmeticDataset
from test.models.test_neurorvq import InterfaceBackend


EXAMPLE = Path(__file__).resolve().parents[2]/'examples/examples_eegmat_neurorvq.py'


def load_example():
    spec = importlib.util.spec_from_file_location('neurorvq_workflow_test', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Require200Hz.')
    samples = 1600 if record[1] == 0 else 800
    return dict(eeg_data=np.random.RandomState(42+record[1]).normal(size=(19, samples)), label=record[1], record=record)


class TestNeuroRVQExample(unittest.TestCase):
    def test_window_standardization_shape_and_zero_signal(self):
        module = load_example()
        values = np.arange(19*800, dtype=np.float64).reshape(19, 800)
        actual = module.prepare_window(values)
        expected = ((values-values.mean(-1, keepdims=True))/values.std(-1, keepdims=True)).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(module.prepare_window(np.zeros((19, 800))), np.zeros((19, 800), dtype=np.float32))
        for bad in [np.zeros((18, 800)), np.full((19, 800), np.nan)]:
            with self.assertRaises(ValueError): module.prepare_window(bad)

    def test_backend_workflow_export_and_immutable_state(self):
        # Interface fixture only. The external-source CI job exercises the full
        # separately licensed native model and actual official checkpoint.
        module = load_example()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            for subject in range(2):
                for condition in [1, 2]:
                    (source/f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic placeholder')
            backend = InterfaceBackend(vocab_size=19)
            state = {key: value.clone() for key, value in backend.state_dict().items()}
            checkpoint, artifact = root/'model.pt', root/'probe.pt'; torch.save(state, checkpoint)
            def factory(backend_root, channel_names):
                return InterfaceBackend(vocab_size=len(channel_names)), {'channel_vocab': [c.lower() for c in channel_names]}
            args = [str(EXAMPLE), '--root-path', str(source), '--checkpoint', str(checkpoint),
                    '--backend-root', str(root/'explicit-local-source'), '--cache-path', str(root/'cache'),
                    '--output-path', str(artifact), '--epochs', '1', '--batch-size', '2']
            with patch.object(module, 'build_neurorvq_backend', side_effect=factory), patch.object(
                MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), patch('sys.argv', args):
                module.main()
            payload = torch.load(artifact, weights_only=True)
            metadata = json.loads(payload['metadata_json'])
            self.assertEqual((metadata['train_windows'], metadata['test_windows'], metadata['updates']), (3, 3, 2))
            self.assertEqual(metadata['train_subjects'], ['Subject00'])
            self.assertEqual(metadata['held_out_subject'], 'Subject01')
            self.assertEqual(payload['test_logits'].shape, (3, 2))
            for name, value in state.items():
                torch.testing.assert_close(payload['state_dict']['backbone.backend.'+name], value, rtol=0, atol=0)
            self.assertIn('backbone.channel_fingerprint', payload['state_dict'])
            self.assertIn('backbone.spatial_ids', payload['state_dict'])
            loss = torch.nn.functional.cross_entropy(payload['test_logits'], payload['test_targets']).item()
            self.assertEqual(metadata['metrics']['loss'], loss)


if __name__ == '__main__':
    unittest.main()
