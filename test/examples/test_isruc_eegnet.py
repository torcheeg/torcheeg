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
import pandas as pd
import torch
from torch.nn import functional as F

from torcheeg.models import EEGNet


MODULE = 'torcheeg.datasets.module.sleep_stage_detection.isruc'
PATH = Path(__file__).resolve().parents[2] / 'examples/examples_isruc_eegnet.py'
SPEC = importlib.util.spec_from_file_location('isruc_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def raw_record(path, preload):
    assert preload
    fixture = json.loads(Path(path).read_text())
    time = np.arange(18000) / 100.
    eeg = np.sin(2 * np.pi * 5 * time) + fixture['subject']
    eog = 2 * np.cos(2 * np.pi * 2 * time) - fixture['night']
    if fixture['kind'] == 'bare':
        channels = ['F4', 'A1', 'LOC', 'A2']
        data = np.stack([eeg, .2 * np.cos(2 * np.pi * 2 * time),
                         eog, .1 * np.sin(2 * np.pi * 3 * time)])
    else:
        reference = 'A' if fixture['kind'] == 'A' else 'M'
        channels = ['F4-' + reference + '1', 'LOC-' + reference + '2']
        data = np.stack([eeg, eog])
    return mne.io.RawArray(data * 1e-6, mne.create_info(channels, 100, 'eeg'), verbose=False)


class TestISRUCEEGNetExample(unittest.TestCase):
    def run_example(self, root, group=0, subjects=(1, 2, 40), scorer=1):
        records = root / 'records'
        for subject in subjects:
            nights = (1, 2) if group == 1 else (1,)
            for night in nights:
                if group == 1:
                    folder = records / 'Subgroup_2' / f'{subject}_2' / str(night)
                    stem = str(night)
                else:
                    folder = records / f'Subgroup_{group + 1}' / str(subject)
                    stem = str(subject)
                folder.mkdir(parents=True, exist_ok=True)
                fixture = dict(subject=subject, night=night,
                               kind='bare' if subject == 40 else 'A' if subject == 1 else 'M')
                (folder / (stem + '.rec')).write_text(json.dumps(fixture))
                (folder / (stem + '_1.txt')).write_text('0\n1\n2\n3\n5\n5\n')
                (folder / (stem + '_2.txt')).write_text('5\n3\n2\n1\n0\n0\n')
        args = ['example', '--root-path', str(records), '--cache-path', str(root / 'cache'),
                '--output-path', str(root / 'model.pt'), '--epochs', '1', '--batch-size', '4',
                '--groups', str(group), '--scorer', str(scorer)]
        with patch(MODULE + '.read_raw_edf', side_effect=raw_record), \
                patch('sys.argv', args), redirect_stdout(io.StringIO()):
            EXAMPLE.main()
        return json.loads((root / 'model.pt.json').read_text()), records

    def test_training_reference_provenance_export_and_independent_metrics(self):
        initial = {}
        def observe(**kwargs):
            model = EEGNet(**kwargs)
            if not initial:
                initial.update({name: value.clone() for name, value in model.state_dict().items()})
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(EXAMPLE, 'EEGNet', side_effect=observe):
                report, records = self.run_example(root)
            self.assertEqual(report['partitions'], dict(train_subjects=['Subgroup_1_1', 'Subgroup_1_2'],
                                                        test_subjects=['Subgroup_1_40']))
            self.assertEqual((report['train_epochs'], report['test_epochs'], report['optimizer_steps']), (12, 6, 3))
            provenance = report['preprocessing']['record_channels']
            self.assertEqual(provenance['Subgroup_1_2_night1']['channel_names'], ['F4-M1', 'LOC-M2'])
            target = provenance['Subgroup_1_40_night1']
            self.assertEqual(target['channel_names'], ['F4-A1', 'LOC-A2'])
            self.assertEqual(target['derived_channels'], {'F4-A1': ['F4', 'A1'], 'LOC-A2': ['LOC', 'A2']})
            self.assertEqual(target['zero_filled_channels'], [])
            payload = torch.load(root / 'model.pt', weights_only=True)
            config = json.loads(payload['model_config_json'])
            self.assertEqual(set(config), set(inspect.signature(EEGNet).parameters))
            model = EEGNet(**config).requires_grad_(False).eval()
            model.load_state_dict(payload['state_dict'], strict=True)
            for name in ('block1.0.weight', 'lin.weight'):
                self.assertFalse(torch.equal(payload['state_dict'][name], initial[name]), name)
            raw = raw_record(records / 'Subgroup_1/40/40.rec', True)
            # Independent subtraction from recorded components, before filtering.
            values = raw.get_data()[[0, 2]] - raw.get_data()[[1, 3]]
            raw = mne.io.RawArray(values, mne.create_info(['EEG', 'EOG'], 100, 'eeg'), verbose=False)
            raw.filter(.5, 30, verbose=False)
            values = raw.get_data().reshape(2, 6, 3000).transpose(1, 0, 2)
            values = (values - values.mean(axis=-1, keepdims=True)) / values.std(axis=-1, keepdims=True)
            eeg = torch.from_numpy(np.ascontiguousarray(values[:, None], dtype=np.float32))
            labels = torch.tensor([0, 1, 2, 3, 4, 4])
            with torch.no_grad():logits = model(eeg)
            self.assertAlmostEqual(report['test_metrics']['test_loss'], F.cross_entropy(logits, labels).item(), places=5)
            self.assertAlmostEqual(report['test_metrics']['test_accuracy'], (logits.argmax(1) == labels).float().mean().item(), places=6)
            self.assertTrue(report['strict_reload_equal'])

    def test_both_nights_of_heldout_participant_stay_in_test(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = self.run_example(Path(directory), group=1, subjects=(1, 2))
            self.assertEqual(report['partitions']['test_subjects'], ['Subgroup_2_2_2'])
            self.assertEqual(report['test_records'], ['Subgroup_2_2_2_night1', 'Subgroup_2_2_2_night2'])
            self.assertEqual(len(report['train_records']), 2)
            self.assertEqual((report['train_epochs'], report['test_epochs']), (12, 12))
            self.assertFalse(set(report['train_records']) & set(report['test_records']))

    def test_explicit_second_expert_is_selected_without_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, records = self.run_example(root, subjects=(1, 2), scorer=2)
            self.assertEqual(report['preprocessing']['scorer'], 2)
            payload = torch.load(root / 'model.pt', weights_only=True)
            model = EEGNet(**json.loads(payload['model_config_json'])).requires_grad_(False).eval()
            model.load_state_dict(payload['state_dict'], strict=True)
            raw = raw_record(records / 'Subgroup_1/2/2.rec', True)
            raw.filter(.5, 30, verbose=False)
            values = raw.get_data().reshape(2, 6, 3000).transpose(1, 0, 2)
            values = (values - values.mean(axis=-1, keepdims=True)) / values.std(axis=-1, keepdims=True)
            eeg = torch.from_numpy(np.ascontiguousarray(values[:, None], dtype=np.float32))
            labels = torch.tensor([4, 3, 2, 1, 0, 0])
            with torch.no_grad():logits = model(eeg)
            self.assertAlmostEqual(report['test_metrics']['test_loss'], F.cross_entropy(logits, labels).item(), places=5)

    def test_single_participant_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            with self.assertRaisesRegex(ValueError, 'two ISRUC participants'):
                self.run_example(Path(directory), subjects=(1,))
            trainer.assert_not_called()

    def test_output_preserved_and_invalid_cli_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.pt').write_bytes(b'existing state')
            with self.assertRaisesRegex(ValueError, 'fresh output'):
                self.run_example(root)
            self.assertEqual((root / 'model.pt').read_bytes(), b'existing state')
        for argument, value in [('--epochs', '0'), ('--batch-size', '0'), ('--scorer', '3'), ('--groups', '3')]:
            with self.subTest(argument=argument), patch('sys.argv', ['example', '--root-path', 'unused', argument, value]), \
                    patch.object(EXAMPLE, 'ISRUCDataset') as dataset, redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):EXAMPLE.main()
                dataset.assert_not_called()

    def test_inconsistent_or_zero_filled_provenance_fails(self):
        from types import SimpleNamespace
        frame = pd.DataFrame(dict(record_id=['one', 'one'],
                                   channel_names=['["EEG", "EOG"]', '["EEG", "other"]'],
                                   derived_channels=['{}', '{}'], zero_filled_channels=['[]', '[]']))
        with self.assertRaisesRegex(ValueError, 'constant'):
            EXAMPLE.channel_provenance(SimpleNamespace(info=frame))
        frame['channel_names'] = '["EEG", "EOG"]'
        frame['zero_filled_channels'] = '["EOG"]'
        with self.assertRaisesRegex(ValueError, 'measured'):
            EXAMPLE.channel_provenance(SimpleNamespace(info=frame))


if __name__ == '__main__':
    unittest.main()
