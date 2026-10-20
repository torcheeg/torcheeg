import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.models import BIOT, BIOTClassifier


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_biot.py'


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Expected 200 Hz.')
    return dict(eeg_data=np.random.RandomState(42 + record[1]).normal(size=(19, 1600)),
                label=record[1], record=record)


class TestBIOTExample(unittest.TestCase):
    def test_bipolar_normalization_and_zero_signal(self):
        namespace = runpy.run_path(str(EXAMPLE))
        values = np.arange(19 * 800, dtype=np.float64).reshape(19, 800)
        values = np.sin(values / 17) + np.cos(values / 31)
        expected = np.stack([values[MENTAL_ARITHMETIC_CHANNEL_LIST.index(a)] -
            values[MENTAL_ARITHMETIC_CHANNEL_LIST.index(b)] for a, b in namespace['BIPOLAR_PAIRS']]) * 1e-6
        expected /= np.quantile(np.abs(expected), .95, axis=-1, keepdims=True) + 1e-8
        np.testing.assert_array_equal(namespace['prepare_window'](values), expected.astype(np.float32))
        np.testing.assert_array_equal(namespace['prepare_window'](np.zeros((19, 800))), np.zeros((16, 800)))
        for eeg in [np.zeros((18, 800)), np.full((19, 800), np.nan)]:
            with self.assertRaises(ValueError):
                namespace['prepare_window'](eeg)

    def test_fullsize_checkpoint_cli_training_and_safe_complete_restore(self):
        # Generated canonical state checks the workflow; it is not pretrained.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'; source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic placeholder')
            original = BIOT(num_electrodes=16)
            checkpoint, artifact = root / 'canonical.pt', root / 'probe.pt'
            torch.save(original.state_dict(), checkpoint)
            arguments = [str(EXAMPLE), '--root-path', str(source), '--checkpoint', str(checkpoint),
                '--variant', 'prest-16ch', '--cache-path', str(root / 'cache'),
                '--output-path', str(artifact), '--epochs', '1', '--batch-size', '3']
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), \
                    patch('sys.argv', arguments):
                runpy.run_path(str(EXAMPLE), run_name='__main__')
            payload = torch.load(artifact, weights_only=True)
            metadata = json.loads(payload['metadata_json'])
            self.assertEqual(metadata['train_subjects'], ['Subject00'])
            self.assertEqual(metadata['held_out_subject'], 'Subject01')
            self.assertEqual((metadata['train_windows'], metadata['test_windows'], metadata['updates']), (4, 4, 2))
            self.assertEqual(metadata['channel_ids'], list(range(16)))
            restored = BIOTClassifier(BIOT(**json.loads(payload['backbone_config_json'])), 2, freeze_backbone=True).eval()
            restored.load_state_dict(payload['state_dict'], strict=True)
            for key, value in original.state_dict().items():
                torch.testing.assert_close(restored.biot.state_dict()[key], value, rtol=0, atol=0)
            loss = torch.nn.functional.cross_entropy(payload['test_logits'], payload['test_targets']).item()
            accuracy = (payload['test_logits'].argmax(1) == payload['test_targets']).float().mean().item()
            self.assertEqual(metadata['metrics'], dict(loss=loss, accuracy=accuracy))
            self.assertEqual(payload['test_logits'].shape, (4, 2))


if __name__ == '__main__':
    unittest.main()
