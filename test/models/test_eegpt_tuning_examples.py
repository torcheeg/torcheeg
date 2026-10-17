import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from moabb.datasets.fake import FakeDataset
from moabb.paradigms import LeftRightImagery
import torch

from torcheeg.models import (EEGPT, EEGPTClassifier, add_eegpt_adapters,
                             add_eegpt_prefix, add_eegpt_lora)


class TestEEGPTTuningExamples(unittest.TestCase):
    def run_example(self, method, helper):
        source = FakeDataset(event_list=['left_hand', 'right_hand'], n_sessions=1,
            n_runs=1, n_subjects=2, seed=42, sfreq=250, duration=40, n_events=4)
        paradigm = LeftRightImagery(channels=['C3', 'Cz', 'C4'], fmin=4, fmax=38,
                                    tmin=0, tmax=4, resample=256)
        expected = {subject: len(paradigm.get_data(source, subjects=[subject])[0]) for subject in [1, 2]}
        example = Path(__file__).resolve().parents[2] / 'examples' / f'examples_bciciv2b_eegpt_{method}.py'
        captured = {}
        def observe(model, **kwargs):
            captured['base'] = {name: value.clone() for name, value in model.encoder.state_dict().items()}
            captured['model'] = helper(model, **kwargs)
            return captured['model']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'tuning.pt'
            arguments = [str(example), '--download-path', str(root / 'raw'),
                '--cache-path', str(root / 'cache'), '--epochs', '1', '--batch-size', '2',
                '--output-path', str(artifact)]
            output = io.StringIO()
            with patch('moabb.datasets.BNCI2014_004', return_value=source), \
                    patch.object(FakeDataset, 'download', return_value=None), \
                    patch('torcheeg.models.' + helper.__name__, side_effect=observe), \
                    patch('sys.argv', arguments), redirect_stdout(output):
                runpy.run_path(str(example), run_name='__main__')
            result = ast.literal_eval(output.getvalue().strip().splitlines()[-1])
            self.assertFalse(result['checkpoint_loaded'])
            self.assertEqual(result['tuning_method'], method)
            self.assertEqual(result['train_windows'], expected[1])
            self.assertEqual(result['test_windows'], expected[2])
            self.assertEqual(result['held_out_subject'], 2)
            self.assertTrue(0 < result['trainable_parameters'] < result['total_parameters'])
            self.assertTrue(0 <= result['metrics']['test_accuracy'] <= 1)
            payload = torch.load(artifact, weights_only=True)
            self.assertEqual(payload['tuning_method'], method)
            restored = EEGPTClassifier(EEGPT(**payload['encoder_config']), **payload['head_config'])
            helper(restored, **payload['tuning_config'])
            restored.load_state_dict(payload['state_dict'], strict=True)
            eeg = torch.linspace(-1, 1, 2 * 3 * 1024).reshape(2, 3, 1024)
            torch.testing.assert_close(restored.eval()(eeg), captured['model'].eval()(eeg))
            for name, value in captured['base'].items():
                torch.testing.assert_close(captured['model'].encoder.state_dict()[name], value, rtol=0, atol=0)

    def test_adapter_cli_training_and_saved_state(self):
        self.run_example('adapters', add_eegpt_adapters)

    def test_prefix_cli_training_and_saved_state(self):
        self.run_example('prefix', add_eegpt_prefix)

    def test_lora_cli_training_and_saved_state(self):
        self.run_example('lora', add_eegpt_lora)


if __name__ == '__main__':
    unittest.main()
