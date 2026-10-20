import gc
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
from torcheeg.models import BENDR


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/examples_eegmat_bendr.py'


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 256:
        raise ValueError('Expected 256 Hz.')
    return dict(eeg_data=np.random.RandomState(42 + record[1]).normal(size=(19, 2048)),
                label=record[1], record=record)


class TestBENDRExample(unittest.TestCase):
    def test_native_order_global_normalization_and_unknown_scale(self):
        namespace = runpy.run_path(str(EXAMPLE))
        values = np.arange(19 * 1024, dtype=np.float64).reshape(19, 1024)
        for eeg in (values, values - 9000., np.zeros_like(values), np.ones_like(values)):
            selected = eeg[[MENTAL_ARITHMETIC_CHANNEL_LIST.index(ch) for ch in namespace['CHANNEL_ORDER']]]
            low, high = min(0., selected.min()), max(0., selected.max())
            expected = 2 * ((selected - low) / (high - low) - .5) if high > low else np.zeros_like(selected)
            expected = np.concatenate((expected, np.zeros((1, 1024)))).astype(np.float32)
            actual = namespace['prepare_window'](eeg)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual[19], np.zeros(1024))
        for eeg in (np.zeros((19, 1023)), np.full((19, 1024), np.nan)):
            with self.assertRaises(ValueError):
                namespace['prepare_window'](eeg)

    def test_fullsize_synthetic_checkpoint_workflow_and_unchanged_backbone(self):
        # Full official architecture with generated weights; not pretrained.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic placeholder')
            original = BENDR()
            encoder, context, artifact = root / 'encoder.pt', root / 'context.pt', root / 'probe.pt'
            torch.save(original.encoder.state_dict(), encoder)
            torch.save(original.contextualizer.state_dict(), context)
            del original
            gc.collect()
            arguments = [str(EXAMPLE), '--root-path', str(source), '--encoder-checkpoint', str(encoder),
                '--contextualizer-checkpoint', str(context), '--cache-path', str(root / 'cache'),
                '--output-path', str(artifact), '--epochs', '1', '--batch-size', '3']
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), \
                    patch('sys.argv', arguments):
                runpy.run_path(str(EXAMPLE), run_name='__main__')
            payload = torch.load(artifact, weights_only=True)
            meta = json.loads(payload['metadata_json'])
            config = json.loads(payload['backbone_config_json'])
            self.assertEqual((config['hid_channels'], config['depth'], config['num_electrodes']), (512, 8, 20))
            self.assertEqual(meta['train_subjects'], ['Subject00'])
            self.assertEqual(meta['held_out_subject'], 'Subject01')
            self.assertEqual((meta['train_windows'], meta['test_windows'], meta['updates']), (4, 4, 2))
            self.assertEqual((meta['sampling_rate'], meta['window_samples']), (256, 1024))
            self.assertEqual(payload['test_logits'].shape, (4, 2))
            loss = torch.nn.functional.cross_entropy(payload['test_logits'], payload['test_targets']).item()
            accuracy = (payload['test_logits'].argmax(1) == payload['test_targets']).float().mean().item()
            self.assertEqual(meta['metrics'], dict(loss=loss, accuracy=accuracy))
            for prefix, checkpoint in [('encoder', encoder), ('contextualizer', context)]:
                state = torch.load(checkpoint, weights_only=True)
                for name, value in state.items():
                    torch.testing.assert_close(payload['state_dict']['backbone.' + prefix + '.' + name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
