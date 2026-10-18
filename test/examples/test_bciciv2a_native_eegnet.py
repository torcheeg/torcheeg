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

from torcheeg.models import EEGNet


PATH = Path(__file__).resolve().parents[2] / 'examples/examples_bciciv2a_native_eegnet.py'
SPEC = importlib.util.spec_from_file_location('bciciv2a_native_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class TestBCICIV2aNativeEEGNetExample(unittest.TestCase):
    def run_example(self, root, extra=(), missing_e=False):
        sources = {}
        for subject in [1, 2]:
            for phase in ['T', 'E']:
                if missing_e and phase == 'E':
                    continue
                name = f'A{subject:02d}{phase}.mat'
                sessions = []
                for run_index in range(9):
                    time = np.arange(9000) / 250.
                    values = np.stack([np.sin(2 * np.pi * (4 + i) * time) * (i + 1) + run_index
                                       for i in range(25)], axis=1)
                    session = dict(X=values, trial=np.array([1, 2402, 4701, 7101]),
                                   y=np.array([1, 2, 3, 4]), fs=250, artifacts=np.array([1, 0, 0, 0]))
                    if run_index < 3:
                        session.update(trial=np.array([], dtype=int), y=np.array([], dtype=int),
                                       artifacts=np.array([], dtype=int))
                    sessions.append(session)
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
            self.assertEqual((report['train_trials'], report['test_trials'], report['optimizer_steps']), (24, 24, 6))
            self.assertEqual((report['train_flagged_trials'], report['test_flagged_trials']), (6, 6))
            self.assertEqual(report['partitions'], dict(subject_id='A02', train_phase='train', test_phase='test',
                 train_sessions=['T'], test_sessions=['E'], train_runs=[3, 4, 5, 6, 7, 8],
                 test_runs=[3, 4, 5, 6, 7, 8], train_records=['A02T'], test_records=['A02E']))
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
            for session in source['A02E.mat'][3:]:
                raw = mne.io.RawArray(session['X'][:, :22].T * 1e-6,
                      mne.create_info([str(i) for i in range(22)], 250, 'eeg'), verbose=False)
                raw.filter(4., 38., verbose=False).resample(128, verbose=False)
                samples = raw.get_data() * 1e6
                for onset in session['trial']:
                    start = round((int(onset) - 1) * 128 / 250) + 256
                    values.append(samples[:, start:start + 512])
            values = np.stack(values)
            values = (values - values.mean(-1, keepdims=True)) / values.std(-1, keepdims=True)
            with torch.no_grad():
                logits = model(torch.tensor(values[:, None], dtype=torch.float32))
            targets = torch.tensor([0, 1, 2, 3] * 6)
            self.assertAlmostEqual(report['test_metrics']['test_loss'], F.cross_entropy(logits, targets).item(), places=5)
            self.assertAlmostEqual(report['test_metrics']['test_accuracy'], (logits.argmax(1) == targets).float().mean().item(), places=6)

    def test_artifact_exclusion_changes_both_phases_and_keeps_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = self.run_example(Path(directory), extra=['--skip-artifacts'])
            self.assertEqual((report['train_trials'], report['test_trials'], report['optimizer_steps']), (18, 18, 5))
            self.assertEqual((report['train_flagged_trials'], report['test_flagged_trials']), (0, 0))
            self.assertTrue(report['preprocessing']['skip_trial_with_artifacts'])
            self.assertEqual(report['partitions']['train_runs'], [3, 4, 5, 6, 7, 8])
            self.assertEqual(report['partitions']['test_runs'], [3, 4, 5, 6, 7, 8])

    def test_missing_evaluation_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            with self.assertRaisesRegex(ValueError, 'nonempty T'):
                self.run_example(Path(directory), missing_e=True)
            trainer.assert_not_called()

    def test_missing_run_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            from torcheeg.datasets import BCICIV2aDataset
            reader = BCICIV2aDataset.process_record
            def missing_run(*args, **kwargs):
                for row in reader(*args, **kwargs):
                    if row['info']['run'] != 8:
                        yield row
            with patch.object(BCICIV2aDataset, 'process_record', side_effect=missing_run):
                with self.assertRaisesRegex(ValueError, 'all six T runs'):
                    self.run_example(Path(directory))
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
            with self.subTest(option=option, value=value), patch('sys.argv', ['example', '--root-path', 'unused', option, value]), patch.object(EXAMPLE, 'BCICIV2aDataset') as dataset:
                with self.assertRaises(SystemExit):
                    EXAMPLE.main()
                dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
