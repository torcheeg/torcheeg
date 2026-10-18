from contextlib import redirect_stdout
import importlib.util
import io
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np
import torch
from torch.nn import functional as F

from torcheeg.models import EEGNet


MODULE = 'torcheeg.datasets.module.sleep_stage_detection.sleep_edfx'
PATH = Path(__file__).resolve().parents[2] / 'examples/examples_sleepedf_eegnet.py'
SPEC = importlib.util.spec_from_file_location('sleepedf_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def raw_record(path, preload):
    assert preload
    subject = int(Path(path).name[3:5])
    time = np.arange(24000) / 100.
    data = np.stack([np.sin(2 * np.pi * 5 * time) + subject,
                     2 * np.cos(2 * np.pi * 2 * time) - subject]) * 1e-6
    return mne.io.RawArray(data, mne.create_info(['EEG Fpz-Cz', 'EOG horizontal'], 100, 'eeg'), verbose=False)


def annotations(path):
    return mne.Annotations(np.arange(8) * 30., [30] * 8,
                           ['Sleep stage W', 'Sleep stage 1', 'Movement time',
                            'Sleep stage 2', 'Sleep stage 3', 'Sleep stage ?',
                            'Sleep stage 4', 'Sleep stage R'])


class TestSleepEDFEEGNetExample(unittest.TestCase):
    def run_example(self, root, groups=3, extra=()):
        for study, prefix in [('sleep-cassette', 'SC4'), ('sleep-telemetry', 'ST7')]:
            records = root / study
            records.mkdir(exist_ok=True)
            for subject in range(groups):
                for night in [1, 2]:
                    for suffix in ['E0-PSG.edf', 'EC-Hypnogram.edf']:
                        (records / f'{prefix}{subject:02d}{night}{suffix}').write_bytes(b'anonymous reader fixture')
        args = ['example', '--root-path', str(root), '--cache-path', str(root / 'cache'),
                '--output-path', str(root / 'model.pt'), '--epochs', '1', '--batch-size', '4'] + list(extra)
        with patch(MODULE + '.read_raw_edf', side_effect=raw_record), \
                patch(MODULE + '.mne.read_annotations', side_effect=annotations), \
                patch('sys.argv', args), redirect_stdout(io.StringIO()):
            EXAMPLE.main()
        return json.loads((root / 'model.pt.json').read_text())

    def test_training_split_preprocessing_export_and_safe_roundtrip(self):
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
            payload = torch.load(root / 'model.pt', weights_only=True)
            self.assertEqual(report['partitions'], dict(train_groups=['SC400', 'SC401'], test_groups=['SC402'],
                  train_records=['SC4001E0', 'SC4002E0', 'SC4011E0', 'SC4012E0'],
                  test_records=['SC4021E0', 'SC4022E0']))
            self.assertEqual((report['train_epochs'], report['test_epochs'], report['optimizer_steps']), (24, 12, 6))
            self.assertEqual(report['preprocessing']['channels'], ['EEG Fpz-Cz', 'EOG horizontal'])
            self.assertEqual(report['preprocessing']['normalization'], 'per_epoch_per_channel')
            self.assertTrue(report['strict_reload_equal'])
            self.assertEqual(json.loads(payload['partitions_json']), report['partitions'])
            self.assertEqual(constructor.call_args.kwargs, EXAMPLE.MODEL_CONFIG)
            self.assertEqual(set(json.loads(payload['model_config_json'])), set(inspect.signature(EEGNet).parameters))
            restored = EEGNet(**json.loads(payload['model_config_json'])).requires_grad_(False).eval()
            restored.load_state_dict(payload['state_dict'], strict=True)
            self.assertTrue(all(torch.isfinite(value).all() for value in restored.state_dict().values()))
            for name in ('block1.0.weight', 'lin.weight'):
                self.assertFalse(torch.equal(payload['state_dict'][name], initial[name]), name)
            raw = raw_record('SC4021E0-PSG.edf', True)
            raw.filter(.5, 30, verbose=False)
            values = raw.get_data().reshape(2, 8, 3000).transpose(1, 0, 2)[[0, 1, 3, 4, 6, 7]]
            values = (values - values.mean(axis=-1, keepdims=True)) / values.std(axis=-1, keepdims=True)
            eeg = torch.tensor(np.concatenate([values, values])[:, None], dtype=torch.float32)
            targets = torch.tensor([0, 1, 2, 3, 3, 4] * 2)
            with torch.no_grad():
                logits = restored(eeg)
            self.assertAlmostEqual(report['test_metrics']['test_loss'], F.cross_entropy(logits, targets).item(), places=5)
            self.assertAlmostEqual(report['test_metrics']['test_accuracy'], (logits.argmax(1) == targets).float().mean().item(), places=6)

    def test_study_selection_and_both_nights_stay_together(self):
        for studies, train, test in [(['telemetry'], ['ST700', 'ST701'], ['ST702']),
                                     (['cassette', 'telemetry'], ['SC400', 'SC401', 'SC402', 'ST700', 'ST701'], ['ST702'])]:
            with self.subTest(studies=studies), tempfile.TemporaryDirectory() as directory:
                report = self.run_example(Path(directory), extra=['--studies'] + studies)
                self.assertEqual(report['partitions']['train_groups'], train)
                self.assertEqual(report['partitions']['test_groups'], test)
                self.assertEqual(len(report['partitions']['test_records']), 2)
                self.assertEqual(report['test_epochs'], 12)
                self.assertEqual(report['preprocessing']['studies'], studies)
                self.assertIsNone(report['preprocessing']['wake_trim'])

    def test_missing_eog_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            with patch.object(EXAMPLE, 'CHANNELS', ['EEG Fpz-Cz', 'absent EOG']):
                with self.assertRaisesRegex(ValueError, 'Missing required channels'):
                    self.run_example(Path(directory))
            trainer.assert_not_called()

    def test_single_group_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            with self.assertRaisesRegex(ValueError, 'two participants'):
                self.run_example(Path(directory), groups=1)
            trainer.assert_not_called()

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.pt').write_bytes(b'existing owned artifact')
            with self.assertRaisesRegex(ValueError, 'fresh output'):
                self.run_example(root)
            self.assertEqual((root / 'model.pt').read_bytes(), b'existing owned artifact')

    def test_duplicate_studies_fail_without_reading_data(self):
        args = ['example', '--root-path', 'unused', '--studies', 'cassette', 'cassette']
        with patch('sys.argv', args), patch.object(EXAMPLE, 'SleepEDFxDataset') as dataset:
            with self.assertRaises(SystemExit):
                EXAMPLE.main()
            dataset.assert_not_called()

    def test_invalid_cli_fails_without_dataset_work(self):
        for argument in ['--epochs', '--batch-size']:
            args = ['example', '--root-path', 'unused', argument, '0']
            with self.subTest(argument=argument), patch('sys.argv', args), \
                    patch.object(EXAMPLE, 'SleepEDFxDataset') as dataset, redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    EXAMPLE.main()
                dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
