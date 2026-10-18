from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np
from scipy.io import savemat
import torch
from torch.nn import functional as F

from test.datasets.test_bciciv2b import write_record
from torcheeg.models import EEGNet


PATH = Path(__file__).resolve().parents[2] / 'examples/examples_bciciv2b_native_eegnet.py'
SPEC = importlib.util.spec_from_file_location('bciciv2b_native_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class TestBCICIV2bNativeEEGNetExample(unittest.TestCase):
    def run_example(self, root, extra=(), missing_e=False):
        sources = {}
        for subject in [1, 2]:
            for phase in ['T', 'E']:
                if missing_e and phase == 'E':
                    continue
                name = f'B{subject:02d}{phase}.mat'
                sessions = write_record(root, name)
                for session in sessions:
                    session['artifacts'] = np.array([1, 0, 0])
                savemat(root / name, dict(data=np.array(sessions, dtype=object)))
                sources[name] = sessions
        args = ['example', '--root-path', str(root), '--cache-path', str(root / 'cache'),
                '--output-path', str(root / 'model.pt'), '--epochs', '1', '--batch-size', '4'] + list(extra)
        with patch('sys.argv', args), redirect_stdout(io.StringIO()):
            EXAMPLE.main()
        return json.loads((root / 'model.pt.json').read_text()), sources

    def test_actual_mat_training_phase_partition_metrics_and_safe_export(self):
        initial = {}
        def observe(**kwargs):
            model = EEGNet(**kwargs)
            if not initial:
                initial.update({name: value.clone() for name, value in model.state_dict().items()})
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(EXAMPLE, 'EEGNet', side_effect=observe) as constructor:
                report, source = self.run_example(root, extra=['--subject', '2'])
            self.assertEqual((report['train_trials'], report['test_trials'], report['optimizer_steps']), (9, 6, 3))
            self.assertEqual((report['train_flagged_trials'], report['test_flagged_trials']), (3, 2))
            self.assertEqual(report['partitions'], dict(subject=2, train_phase='train', test_phase='test',
                 train_sessions=[1, 2, 3], test_sessions=[4, 5], train_records=['B02_train_s1', 'B02_train_s2', 'B02_train_s3'], test_records=['B02_test_s4', 'B02_test_s5']))
            payload = torch.load(root / 'model.pt', weights_only=True)
            self.assertEqual(json.loads(payload['partitions_json']), report['partitions'])
            self.assertEqual(constructor.call_args.kwargs, EXAMPLE.MODEL_CONFIG)
            self.assertEqual(set(json.loads(payload['model_config_json'])), set(inspect.signature(EEGNet).parameters))
            self.assertEqual(len(payload['state_dict']), 20)
            self.assertTrue(report['strict_reload_equal'])
            for name in ['block1.0.weight', 'lin.weight']:
                self.assertFalse(torch.equal(initial[name], payload['state_dict'][name]), name)
            model = EEGNet(**json.loads(payload['model_config_json'])).requires_grad_(False).eval()
            model.load_state_dict(payload['state_dict'], strict=True)
            values = []
            for session in source['B02E.mat']:
                raw = mne.io.RawArray(session['X'][:, :3].T * 1e-6,
                      mne.create_info(['C3','Cz','C4'], 250, 'eeg'), verbose=False)
                raw.filter(4., 38., verbose=False).resample(128, verbose=False)
                samples = raw.get_data() * 1e6
                for onset in session['trial']:
                    start = round((int(onset) - 1) * 128 / 250) + 384
                    values.append(samples[:, start:start + 512])
            values = np.stack(values)
            values = (values - values.mean(-1, keepdims=True)) / values.std(-1, keepdims=True)
            with torch.no_grad():
                logits = model(torch.tensor(values[:, None], dtype=torch.float32))
            targets = torch.tensor([0, 1, 0] * 2)
            self.assertAlmostEqual(report['test_metrics']['test_loss'], F.cross_entropy(logits, targets).item(), places=5)
            self.assertAlmostEqual(report['test_metrics']['test_accuracy'], (logits.argmax(1) == targets).float().mean().item(), places=6)

    def test_artifact_exclusion_changes_both_phases_and_keeps_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = self.run_example(Path(directory), extra=['--skip-artifacts'])
            self.assertEqual((report['train_trials'], report['test_trials'], report['optimizer_steps']), (6, 4, 2))
            self.assertEqual((report['train_flagged_trials'], report['test_flagged_trials']), (0, 0))
            self.assertTrue(report['preprocessing']['skip_trial_with_artifacts'])
            self.assertEqual(report['partitions']['train_sessions'], [1, 2, 3])
            self.assertEqual(report['partitions']['test_sessions'], [4, 5])

    def test_missing_evaluation_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            with self.assertRaisesRegex(ValueError, 'nonempty T'):
                self.run_example(Path(directory), missing_e=True)
            trainer.assert_not_called()

    def test_existing_model_or_report_is_preserved(self):
        for name in ['model.pt', 'model.pt.json']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory);(root / name).write_bytes(b'existing owned artifact')
                with self.assertRaisesRegex(ValueError, 'fresh output'):
                    self.run_example(root)
                self.assertEqual((root / name).read_bytes(), b'existing owned artifact')

    def test_invalid_cli_fails_without_reading_data(self):
        for option, value in [('--subject','0'), ('--subject','10'), ('--epochs','0'), ('--batch-size','1')]:
            with self.subTest(option=option, value=value), patch('sys.argv', ['example', '--root-path', 'unused', option, value]), patch.object(EXAMPLE, 'BCICIV2bDataset') as dataset:
                with self.assertRaises(SystemExit):
                    EXAMPLE.main()
                dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
