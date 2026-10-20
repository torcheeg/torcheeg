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
from torcheeg.models import LUNA
from test.models.test_luna import TinyLUNA


EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 256:
        raise ValueError('Expected256Hz.')
    return dict(eeg_data=np.random.RandomState(42 + record[1]).normal(size=(19, 10240)),
                label=record[1], record=record)


class TestLUNAExamples(unittest.TestCase):
    def test_window_zscore_zero_and_coordinate_frame(self):
        for variant in ['base', 'large', 'huge']:
            namespace = runpy.run_path(str(EXAMPLES / f'examples_eegmat_luna_{variant}.py'))
            values = np.arange(19 * 1280, dtype=np.float64).reshape(19, 1280)
            expected = (values - values.mean(-1, keepdims=True)) / values.std(-1, keepdims=True)
            np.testing.assert_array_equal(namespace['prepare_window'](values), expected.astype(np.float32))
            np.testing.assert_array_equal(namespace['prepare_window'](np.zeros_like(values)), np.zeros_like(values))
            self.assertTrue(torch.isfinite(namespace['electrode_positions']()).all())
            self.assertEqual(namespace['electrode_positions']().shape, (19, 3))
            self.assertEqual(namespace['VARIANT'], variant)
            with self.assertRaises(ValueError):
                namespace['prepare_window'](np.zeros((18, 1280)))

    def test_base_fullsize_generated_checkpoint_workflow(self):
        self.run_example('base', fullsize=True)

    def test_large_workflow_with_small_strict_loader_fixture(self):
        self.run_example('large', fullsize=False)

    def test_huge_workflow_with_small_strict_loader_fixture(self):
        self.run_example('huge', fullsize=False)

    def run_example(self, variant, fullsize):
        # Actual canonical weights for all3 variants are tested separately on
        # real EEG. CI uses full base architecture and small large/huge fixtures
        # to exercise every workflow without allocating multi-GB exports.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            for subject in range(2):
                for condition in (1, 2):
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic placeholder')
            model = LUNA() if fullsize else TinyLUNA()
            checkpoint, artifact = root / 'model.safetensors', root / 'probe.pt'
            save_file(model.state_dict(), str(checkpoint))
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            del model
            script = EXAMPLES / f'examples_eegmat_luna_{variant}.py'
            args = [str(script), '--root-path', str(source), '--checkpoint', str(checkpoint),
                '--cache-path', str(root / 'cache'), '--output-path', str(artifact), '--epochs', '1', '--batch-size', '8']
            fixture_loader = TinyLUNA.from_pretrained
            def load_fixture(path, variant):
                self.assertEqual(variant, script.stem.rsplit('_', 1)[-1])
                return fixture_loader(path)
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), patch('sys.argv', args):
                if fullsize:
                    runpy.run_path(str(script), run_name='__main__')
                else:
                    with patch.object(LUNA, 'from_pretrained', side_effect=load_fixture):
                        runpy.run_path(str(script), run_name='__main__')
            payload = torch.load(artifact, weights_only=True)
            meta = json.loads(payload['metadata_json']); config = json.loads(payload['backbone_config_json'])
            self.assertEqual(meta['variant'], variant)
            self.assertEqual(meta['train_subjects'], ['Subject00'])
            self.assertEqual(meta['held_out_subject'], 'Subject01')
            self.assertEqual((meta['train_windows'], meta['test_windows'], meta['updates']), (16, 16, 2))
            self.assertEqual(meta['sampling_rate'], 256)
            self.assertEqual(meta['window_samples'], 1280)
            self.assertEqual(config['hid_channels'], 64 if fullsize else 16)
            self.assertEqual(payload['test_logits'].shape, (16, 2))
            self.assertEqual(payload['positions'].shape, (19, 3))
            loss = torch.nn.functional.cross_entropy(payload['test_logits'], payload['test_targets']).item()
            accuracy = (payload['test_logits'].argmax(1) == payload['test_targets']).float().mean().item()
            self.assertEqual(meta['metrics'], dict(loss=loss, accuracy=accuracy))
            for name, value in expected.items():
                torch.testing.assert_close(payload['state_dict']['backbone.' + name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
