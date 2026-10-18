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
from scipy.io import loadmat, savemat
import torch
from torch.nn import functional as F

from test.datasets.test_adftd_rs import participants, write_record
from torcheeg.models import EEGNet


PATH = Path(__file__).resolve().parents[2] / 'examples/examples_adftd_rs_eegnet.py'
SPEC = importlib.util.spec_from_file_location('adftd_rs_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)
ENTRIES = [('sub-001', 'C'), ('sub-010', 'C'), ('sub-002', 'A'),
           ('sub-020', 'A'), ('sub-003', 'F'), ('sub-030', 'F')]


class TestADFTDRSEEGNetExample(unittest.TestCase):
    def write_source(self, root, entries=ENTRIES, constant=False):
        participants(root, entries)
        signals = {}
        for subject, diagnosis in entries:
            for cleaned in [False, True]:
                path, values = write_record(root, subject, samples=2000,
                                            preprocessed=cleaned, boundary=750.5 if cleaned else None)
                if constant:
                    values[3] = .113596
                    data = loadmat(path)
                    data['data'] = values
                    savemat(path, {name: value for name, value in data.items() if not name.startswith('__')})
                signals[subject] = values.astype(np.float64) * 1e-6
        return signals

    def run_example(self, root, cleaned=False):
        output = root / ('cleaned.pt' if cleaned else 'raw.pt')
        args = ['example', '--root-path', str(root), '--cache-path', str(root / 'cache'),
                '--output-path', str(output), '--epochs', '1', '--batch-size', '4']
        if cleaned:
            args.append('--preprocessed')
        with patch('sys.argv', args), redirect_stdout(io.StringIO()):
            EXAMPLE.main()
        return json.loads(output.with_suffix('.pt.json').read_text()), torch.load(output, weights_only=True)

    def test_actual_set_training_subject_holdout_metrics_and_safe_export(self):
        initial = {}
        def observe(**kwargs):
            model = EEGNet(**kwargs)
            if not initial:
                initial.update({name: value.clone() for name, value in model.state_dict().items()})
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_source(root)
            write_record(root, 'sub-001', run='02', samples=2000)
            with patch.object(EXAMPLE, 'EEGNet', side_effect=observe):
                report, payload = self.run_example(root)
            parts = report['partitions']
            self.assertEqual(parts['test_subject_ids_by_label'], {'0': 10, '1': 20, '2': 30})
            self.assertEqual(parts['train_subject_ids'], [1, 2, 3])
            self.assertEqual(parts['test_subject_ids'], [10, 20, 30])
            self.assertEqual(parts['test_participants'], ['sub-010', 'sub-020', 'sub-030'])
            self.assertFalse(set(parts['train_records']) & set(parts['test_records']))
            self.assertEqual(len(parts['train_records']), 4)
            self.assertEqual(len(parts['test_records']), 3)
            self.assertEqual((report['train_windows'], report['test_windows'], report['optimizer_steps']), (16, 12, 4))
            self.assertEqual(json.loads(payload['partitions_json']), parts)
            self.assertEqual(json.loads(payload['preprocessing_json']), report['preprocessing'])
            self.assertEqual(set(json.loads(payload['model_config_json'])), set(inspect.signature(EEGNet).parameters))
            self.assertEqual(len(payload['state_dict']), 20)
            for name in ['block1.0.weight', 'lin.weight']:
                self.assertFalse(torch.equal(initial[name], payload['state_dict'][name]))
            self.assertTrue(report['strict_reload_equal'])
            self.assertEqual(report['preprocessing']['signal_unit'], 'V')
            self.assertEqual(report['preprocessing']['sfreq'], 500)
            values, targets = [], []
            for subject, target in [('sub-010', 0), ('sub-020', 1), ('sub-030', 2)]:
                for start in range(0, 2000, 500):
                    values.append(source[subject][:, start:start + 500])
                    targets.append(target)
            values = np.stack(values)
            values = (values - values.mean(-1, keepdims=True)) / values.std(-1, keepdims=True)
            model = EEGNet(**json.loads(payload['model_config_json'])).requires_grad_(False).eval()
            model.load_state_dict(payload['state_dict'], strict=True)
            with torch.no_grad():
                logits = model(torch.tensor(values[:, None], dtype=torch.float32))
            truth = torch.tensor(targets)
            self.assertAlmostEqual(report['test_metrics']['test_loss'], F.cross_entropy(logits, truth).item(), places=5)
            self.assertAlmostEqual(report['test_metrics']['test_accuracy'], (logits.argmax(1) == truth).float().mean().item(), places=6)
            prediction = logits.argmax(1)
            scores = []
            for label in range(3):
                tp = ((prediction == label) & (truth == label)).sum().item()
                denominator = (prediction == label).sum().item() + (truth == label).sum().item()
                scores.append(2 * tp / denominator if denominator else 0.)
            self.assertAlmostEqual(report['test_metrics']['test_f1score'], sum(scores) / 3, places=6)

    def test_cleaned_source_has_separate_cache_and_segment_safe_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root)
            original, _ = self.run_example(root)
            cleaned, _ = self.run_example(root, True)
            self.assertEqual(cleaned['partitions']['test_participants'], original['partitions']['test_participants'])
            self.assertEqual((cleaned['train_windows'], cleaned['test_windows']), (9, 9))
            self.assertEqual(cleaned['preprocessing']['data_source'], 'preprocessed')
            self.assertTrue((root / 'cache/raw').is_dir())
            self.assertTrue((root / 'cache/preprocessed').is_dir())

    def test_constant_channel_has_finite_zero_normalized_values(self):
        seen = []
        def observe(**kwargs):
            model = EEGNet(**kwargs)
            def check(module, arguments):
                values = arguments[0]
                self.assertTrue(torch.isfinite(values).all())
                self.assertTrue(torch.equal(values[:, :, 3], torch.zeros_like(values[:, :, 3])))
                seen.append(len(values))
            model.register_forward_pre_hook(check)
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root, constant=True)
            with patch.object(EXAMPLE, 'EEGNet', side_effect=observe):
                self.run_example(root)
            self.assertTrue(seen)

    def test_missing_diagnosis_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            root = Path(directory)
            self.write_source(root, ENTRIES[:4])
            with self.assertRaisesRegex(ValueError, 'all three diagnosis'):
                self.run_example(root)
            trainer.assert_not_called()

    def test_single_participant_in_diagnosis_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            root = Path(directory)
            self.write_source(root, ENTRIES[:-1])
            with self.assertRaisesRegex(ValueError, 'two participants per diagnosis'):
                self.run_example(root)
            trainer.assert_not_called()

    def test_existing_output_is_preserved(self):
        for name in ['raw.pt', 'raw.pt.json']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / name).write_bytes(b'owned output')
                with self.assertRaisesRegex(ValueError, 'fresh output'):
                    self.run_example(root)
                self.assertEqual((root / name).read_bytes(), b'owned output')

    def test_invalid_arguments_fail_without_reading_data(self):
        for argument, value in [('--epochs', '0'), ('--batch-size', '1')]:
            with self.subTest(argument=argument), patch('sys.argv', ['example', '--root-path', 'unused', argument, value]), patch.object(EXAMPLE, 'ADFTDRSDataset') as dataset:
                with self.assertRaises(SystemExit):
                    EXAMPLE.main()
                dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
