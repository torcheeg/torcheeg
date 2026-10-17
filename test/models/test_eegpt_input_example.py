import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.models import EEGPT, EEGPTClassifier, EEGPTInputAdapter, EEGPTAdaptedModel


def synthetic_record(record, sampling_rate, **kwargs):
    if sampling_rate != 200:
        raise ValueError('The example must request 200 Hz source windows.')
    return {'eeg_data': np.random.RandomState(42).normal(size=(19, 1600)),
            'label': record[1], 'record': record}


class TestEEGPTInputExample(unittest.TestCase):
    def test_training_source_geometry_frozen_encoder_statistics_and_saved_state(self):
        example = Path(__file__).resolve().parents[2] / 'examples' / 'examples_eegmat_eegpt_input_adapter.py'
        captured = {}
        def observe(adapter, model):
            captured['encoder'] = {name: value.clone() for name, value in model.encoder.state_dict().items()}
            captured['model'] = EEGPTAdaptedModel(adapter, model)
            return captured['model']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for subject in range(2):
                for condition in [1, 2]:
                    (source / f'Subject{subject:02d}_{condition}.edf').write_bytes(b'synthetic source placeholder')
            artifact = root / 'model.pt'
            arguments = [str(example), '--root-path', str(source), '--cache-path', str(root / 'cache'),
                         '--epochs', '1', '--batch-size', '2', '--output-path', str(artifact)]
            output = io.StringIO()
            with patch.object(MentalArithmeticDataset, 'read_record', side_effect=synthetic_record), \
                    patch('torcheeg.models.EEGPTAdaptedModel', side_effect=observe), \
                    patch('sys.argv', arguments), redirect_stdout(output):
                runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(output.getvalue().strip().splitlines()[-1])
            self.assertEqual(result['held_out_subject'], 'Subject01')
            self.assertEqual(result['train_windows'], 4)
            self.assertEqual(result['test_windows'], 4)
            self.assertEqual(result['source_samples'], 800)
            self.assertEqual(result['adapted_samples'], 1024)
            self.assertFalse(result['checkpoint_loaded'])
            self.assertTrue(0 <= result['metrics']['test_accuracy'] <= 1)
            payload = torch.load(artifact, weights_only=True)
            self.assertEqual(payload['source_sampling_rate'], 200)
            self.assertEqual(payload['target_sampling_rate'], 256)
            self.assertEqual(payload['window_seconds'], 4)
            self.assertEqual(len(payload['source_channel_order']), 19)
            self.assertEqual(payload['encoder_token_names'], ['C3', 'CZ', 'C4'])
            restored = EEGPTAdaptedModel(EEGPTInputAdapter(**payload['adapter_config']),
                EEGPTClassifier(EEGPT(**payload['encoder_config']), **payload['head_config']))
            restored.load_state_dict(payload['state_dict'], strict=True)
            eeg = torch.linspace(-1, 1, 2 * 19 * 800).reshape(2, 19, 800)
            torch.testing.assert_close(restored.eval()(eeg), captured['model'].eval()(eeg))
            for name, value in captured['encoder'].items():
                torch.testing.assert_close(captured['model'].model.encoder.state_dict()[name], value, rtol=0, atol=0)
            # Two training batches update statistics; held-out evaluation adds none.
            for module in captured['model'].adapter.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    self.assertEqual(module.num_batches_tracked.item(), 2)


if __name__ == '__main__':
    unittest.main()
