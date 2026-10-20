import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import MENTAL_ARITHMETIC_CHANNEL_LIST
from torcheeg.models import REVE


EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'
SMALL = dict(hid_channels=16, depth=2, num_heads=2, head_channels=8,
             num_frequencies=2, patch_size=200, patch_overlap=20)


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('Expected200Hz.')
    return dict(eeg_data=np.random.RandomState(42 + record[1]).normal(size=(19, 1600)),
                label=record[1], record=record)


class TestREVEExamples(unittest.TestCase):
    def test_preprocessing_and_strict_positions(self):
        for variant in ['base', 'large']:
            namespace = runpy.run_path(str(EXAMPLES / f'examples_eegmat_reve_{variant}.py'))
            values = np.arange(19 * 800, dtype=np.float64).reshape(19, 800)
            expected = (values - values.mean(-1, keepdims=True)) / values.std(-1, keepdims=True)
            np.testing.assert_array_equal(namespace['prepare_window'](values), expected.astype(np.float32))
            np.testing.assert_array_equal(namespace['prepare_window'](np.zeros_like(values)), np.zeros_like(values))
            self.assertEqual(namespace['VARIANT'], variant)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'positions.json'
                coordinates = {name: [index*.001, 0., .1]
                               for index, name in enumerate(MENTAL_ARITHMETIC_CHANNEL_LIST)}
                path.write_text(json.dumps(coordinates))
                positions = namespace['electrode_positions'](path)
                torch.testing.assert_close(positions, torch.tensor(list(coordinates.values())))
                for bad in [{}, dict(coordinates, Fp1=[0., 1.]), dict(coordinates, Fp1=[float('nan')]*3)]:
                    path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        namespace['electrode_positions'](path)
            for invalid in [np.zeros((18, 800)), np.full((19, 800), np.nan)]:
                with self.assertRaises(ValueError):
                    namespace['prepare_window'](invalid)

    def test_base_workflow(self):
        self.run_example('base')

    def test_large_workflow(self):
        self.run_example('large')

    def run_example(self, variant):
        # Full official variants run separately on real EEGMat. CI exercises
        # both complete workflows using small strictly loaded states.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic placeholder')
            checkpoint, artifact = root / 'model.safetensors', root / 'probe.pt'
            state = REVE(**SMALL).state_dict()
            save_file(state, str(checkpoint))
            position_file = root / 'positions.json'
            position_file.write_text(json.dumps({name: [.01, .02, .03] for name in MENTAL_ARITHMETIC_CHANNEL_LIST}))
            script = EXAMPLES / f'examples_eegmat_reve_{variant}.py'
            args = [str(script), '--root-path', str(source), '--checkpoint', str(checkpoint),
                    '--positions-json', str(position_file), '--cache-path', str(root / 'cache'),
                    '--output-path', str(artifact), '--epochs', '1', '--batch-size', '8']
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), \
                    patch.object(REVE, 'variants', {variant: SMALL}), patch('sys.argv', args):
                runpy.run_path(str(script), run_name='__main__')
            payload = torch.load(artifact, weights_only=True)
            metadata = json.loads(payload['metadata_json'])
            self.assertEqual(metadata['variant'], variant)
            self.assertEqual(metadata['train_subjects'], ['Subject00'])
            self.assertEqual(metadata['held_out_subject'], 'Subject01')
            self.assertEqual((metadata['train_windows'], metadata['test_windows'], metadata['updates']), (4, 4, 1))
            self.assertEqual(metadata['sampling_rate'], 200)
            self.assertEqual(metadata['window_samples'], 800)
            self.assertEqual(payload['test_logits'].shape, (4, 2))
            self.assertEqual(payload['positions'].shape, (19, 3))
            loss = torch.nn.functional.cross_entropy(payload['test_logits'], payload['test_targets']).item()
            accuracy = (payload['test_logits'].argmax(1) == payload['test_targets']).float().mean().item()
            self.assertEqual(metadata['metrics'], dict(loss=loss, accuracy=accuracy))
            for name, value in state.items():
                torch.testing.assert_close(payload['state_dict']['backbone.' + name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
