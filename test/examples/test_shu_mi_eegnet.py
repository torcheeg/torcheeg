from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
import torch
from torch.nn import functional as F

from test.datasets.test_shu_mi import trials
from torcheeg.models import EEGNet


PATH = Path(__file__).resolve().parents[2] / 'examples/examples_shu_mi_eegnet.py'
SPEC = importlib.util.spec_from_file_location('shu_mi_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class TestSHUMIEEGNetExample(unittest.TestCase):
    def run_example(self, root, subjects=3):
        for subject in range(1, subjects + 1):
            for session in [1, 2]:
                name = f'sub-{subject:03d}_ses-{session:02d}_task_motorimagery_eeg.mat'
                savemat(root / name, dict(data=trials(), labels=[[1, 2, 1]]))
        args = ['example', '--root-path', str(root), '--cache-path', str(root / 'cache'),
                '--output-path', str(root / 'model.pt'), '--epochs', '1', '--batch-size', '4']
        with patch('sys.argv', args), redirect_stdout(io.StringIO()):
            EXAMPLE.main()
        return json.loads((root / 'model.pt.json').read_text())

    def test_actual_mat_training_participant_split_and_export(self):
        initial = {}
        def observe(**kwargs):
            model = EEGNet(**kwargs)
            if not initial:
                initial.update({name: value.clone() for name, value in model.state_dict().items()})
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(EXAMPLE, 'EEGNet', side_effect=observe) as constructor:
                report = self.run_example(root)
            self.assertEqual((report['train_trials'], report['test_trials'], report['optimizer_steps']), (12, 6, 3))
            self.assertEqual(report['partitions']['train_subjects'], ['sub-001', 'sub-002'])
            self.assertEqual(report['partitions']['test_subjects'], ['sub-003'])
            self.assertEqual(report['partitions']['test_records'], [f'sub-003_ses-{session:02d}_task_motorimagery_eeg' for session in [1, 2]])
            self.assertFalse(set(report['partitions']['train_records']) & set(report['partitions']['test_records']))
            payload = torch.load(root / 'model.pt', weights_only=True)
            self.assertEqual(constructor.call_args.kwargs, EXAMPLE.MODEL_CONFIG)
            self.assertEqual(set(json.loads(payload['model_config_json'])), set(inspect.signature(EEGNet).parameters))
            self.assertEqual(json.loads(payload['partitions_json']), report['partitions'])
            self.assertEqual(len(payload['state_dict']), 20)
            self.assertTrue(report['strict_reload_equal'])
            self.assertEqual(report['preprocessing']['chunk_size'], -1)
            self.assertEqual(report['preprocessing']['signal_unit'], 'uV')
            for name in ['block1.0.weight', 'lin.weight']:
                self.assertFalse(torch.equal(initial[name], payload['state_dict'][name]), name)
            model = EEGNet(**json.loads(payload['model_config_json'])).requires_grad_(False).eval()
            model.load_state_dict(payload['state_dict'], strict=True)
            spectrum = np.fft.rfft(trials().astype(np.float64), axis=-1)[..., :401].copy()
            spectrum[..., -1] *= 2
            values = np.fft.irfft(spectrum, n=800, axis=-1) * .8
            values = (values - values.mean(-1, keepdims=True)) / values.std(-1, keepdims=True)
            eeg = torch.tensor(np.concatenate([values, values])[:, None], dtype=torch.float32)
            targets = torch.tensor([0, 1, 0] * 2)
            with torch.no_grad():
                logits = model(eeg)
            self.assertAlmostEqual(report['test_metrics']['test_loss'], F.cross_entropy(logits, targets).item(), places=5)
            self.assertAlmostEqual(report['test_metrics']['test_accuracy'], (logits.argmax(1) == targets).float().mean().item(), places=6)

    def test_constant_published_channel_stays_finite_and_zero_after_normalization(self):
        values = trials()
        values[:, 3] = 11.
        seen = []
        def observe(**kwargs):
            model = EEGNet(**kwargs)
            def inspect_input(module, arguments):
                eeg = arguments[0]
                self.assertTrue(torch.isfinite(eeg).all())
                self.assertTrue(torch.equal(eeg[:, :, 3], torch.zeros_like(eeg[:, :, 3])))
                seen.append(len(eeg))
            model.register_forward_pre_hook(inspect_input)
            return model
        with tempfile.TemporaryDirectory() as directory, patch(__name__ + '.trials', return_value=values), patch.object(EXAMPLE, 'EEGNet', side_effect=observe):
            report = self.run_example(Path(directory))
        self.assertTrue(seen)
        self.assertEqual(report['test_trials'], 6)

    def test_single_subject_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            with self.assertRaisesRegex(ValueError, 'two participants'):
                self.run_example(Path(directory), subjects=1)
            trainer.assert_not_called()

    def test_existing_model_or_report_is_preserved(self):
        for filename in ['model.pt', 'model.pt.json']:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / filename).write_bytes(b'existing owned output')
                with self.assertRaisesRegex(ValueError, 'fresh output'):
                    self.run_example(root)
                self.assertEqual((root / filename).read_bytes(), b'existing owned output')

    def test_invalid_cli_fails_without_dataset_io(self):
        for argument in ['--epochs', '--batch-size']:
            with self.subTest(argument=argument), patch('sys.argv', ['example', '--root-path', 'unused', argument, '0']), patch.object(EXAMPLE, 'SHUMIDataset') as dataset:
                with self.assertRaises(SystemExit):
                    EXAMPLE.main()
                dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
