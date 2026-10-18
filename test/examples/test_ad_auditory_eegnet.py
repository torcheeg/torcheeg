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
import torch
from torch.nn import functional as F

from test.datasets.test_ad_auditory import participants, write_record
from torcheeg.models import EEGNet


PATH = Path(__file__).resolve().parents[2] / 'examples/examples_ad_auditory_eegnet.py'
SPEC = importlib.util.spec_from_file_location('ad_auditory_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)
ENTRIES = [('sub-01','Normal'), ('sub-02','Normal'), ('sub-03','Mild AD'),
           ('sub-04','Moderate AD'), ('sub-05','MCI'), ('sub-06','MCI'), ('sub-07','-')]


class TestADAuditoryEEGNetExample(unittest.TestCase):
    def run_example(self, root, entries=ENTRIES, constant=False):
        participants(root, entries)
        sources = {}
        for subject, group in entries:
            for run in [None, '02'] if subject == 'sub-06' else [None]:
                path, signal, events = write_record(root, subject=subject, run=run)
                if constant:
                    signal[3] = .113596
                    signal.T.tofile(path.with_suffix('.fdt'))
                sources[path.stem[:-4]] = (subject, group, signal, events)
        args = ['example', '--root-path', str(root), '--cache-path', str(root / 'cache'),
                '--output-path', str(root / 'model.pt'), '--epochs', '1', '--batch-size', '4']
        with patch('sys.argv', args), redirect_stdout(io.StringIO()):
            EXAMPLE.main()
        return json.loads((root / 'model.pt.json').read_text()), sources

    def test_actual_set_fdt_training_participant_split_metrics_and_complete_safe_export(self):
        initial = {}
        def observe(**kwargs):
            model = EEGNet(**kwargs)
            if not initial:
                initial.update({name: value.clone() for name, value in model.state_dict().items()})
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(EXAMPLE, 'EEGNet', side_effect=observe) as constructor:
                report, source = self.run_example(root)
            parts = report['partitions']
            self.assertEqual(parts['train_subjects'], ['sub-01','sub-03','sub-05'])
            self.assertEqual(parts['test_subjects'], ['sub-02','sub-04','sub-06'])
            self.assertEqual(parts['test_subjects_by_label'], {'0':'sub-02','1':'sub-04','2':'sub-06'})
            self.assertEqual(len(parts['subject_labels']), 6)
            self.assertEqual(len(parts['test_records']), 4)
            self.assertFalse(set(parts['train_records']) & set(parts['test_records']))
            self.assertEqual((report['train_windows'], report['test_windows'], report['optimizer_steps']), (15,20,4))
            self.assertEqual((report['train_events'], report['test_events']), (9,12))
            self.assertEqual(report['test_condition_windows'], {'Stimulus':16,'Rest':4})
            payload = torch.load(root / 'model.pt', weights_only=True)
            self.assertEqual(json.loads(payload['partitions_json']), parts)
            self.assertEqual(constructor.call_args.kwargs, EXAMPLE.MODEL_CONFIG)
            self.assertEqual(set(json.loads(payload['model_config_json'])), set(inspect.signature(EEGNet).parameters))
            self.assertEqual(len(payload['state_dict']), 20)
            self.assertTrue(report['strict_reload_equal'])
            for name in ['block1.0.weight', 'lin.weight']:
                self.assertFalse(torch.equal(initial[name], payload['state_dict'][name]), name)
            self.assertEqual(report['preprocessing']['trial_types'], ['Stimulus','Rest'])
            self.assertEqual(report['preprocessing']['sfreq'],250)
            self.assertEqual(report['preprocessing']['constant_channels'],'equal_extrema_centered_exactly')
            self.assertIsNone(report['preprocessing']['l_freq'])
            self.assertEqual(report['preprocessing']['source_label_mapping']['Moderate AD'],1)
            values, targets = [], []
            for record_id in parts['test_records']:
                _, group, signal, events = source[record_id]
                for event in events:
                    first = round(event['onset']*250);last = round((event['onset']+event['duration'])*250)
                    for start in range(first,last-250+1,250):
                        values.append(signal[:,start:start+250])
                        targets.append({'Normal':0,'Mild AD':1,'Moderate AD':1,'MCI':2}[group])
            values=np.stack(values)
            values=(values-values.mean(-1,keepdims=True))/values.std(-1,keepdims=True)
            model=EEGNet(**json.loads(payload['model_config_json'])).requires_grad_(False).eval()
            model.load_state_dict(payload['state_dict'],strict=True)
            with torch.no_grad():
                logits=model(torch.tensor(values[:,None],dtype=torch.float32))
            truth=torch.tensor(targets)
            self.assertAlmostEqual(report['test_metrics']['test_loss'],F.cross_entropy(logits,truth).item(),places=5)
            self.assertAlmostEqual(report['test_metrics']['test_accuracy'],(logits.argmax(1)==truth).float().mean().item(),places=6)

    def test_constant_channel_stays_finite_and_zero_after_window_normalization(self):
        seen=[]
        def observe(**kwargs):
            model=EEGNet(**kwargs)
            def check(module,arguments):
                eeg=arguments[0]
                self.assertTrue(torch.isfinite(eeg).all())
                self.assertTrue(torch.equal(eeg[:,:,3],torch.zeros_like(eeg[:,:,3])))
                seen.append(len(eeg))
            model.register_forward_pre_hook(check)
            return model
        with tempfile.TemporaryDirectory() as directory,patch.object(EXAMPLE,'EEGNet',side_effect=observe):
            report,_=self.run_example(Path(directory),constant=True)
        self.assertTrue(seen)
        self.assertEqual(report['test_windows'],20)

    def test_missing_diagnosis_class_fails_before_training(self):
        entries=[row for row in ENTRIES if row[1] != 'MCI']
        with tempfile.TemporaryDirectory() as directory,patch.object(EXAMPLE,'ClassifierTrainer') as trainer:
            with self.assertRaisesRegex(ValueError,'all three diagnosis'):
                self.run_example(Path(directory),entries=entries)
            trainer.assert_not_called()

    def test_single_participant_in_a_class_fails_before_training(self):
        entries=[row for row in ENTRIES if row[0] != 'sub-06']
        with tempfile.TemporaryDirectory() as directory,patch.object(EXAMPLE,'ClassifierTrainer') as trainer:
            with self.assertRaisesRegex(ValueError,'two participants per diagnosis'):
                self.run_example(Path(directory),entries=entries)
            trainer.assert_not_called()

    def test_existing_model_or_report_is_preserved(self):
        for name in ['model.pt','model.pt.json']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/name).write_bytes(b'existing owned output')
                with self.assertRaisesRegex(ValueError,'fresh output'):
                    self.run_example(root)
                self.assertEqual((root/name).read_bytes(),b'existing owned output')

    def test_invalid_cli_fails_without_reading_data(self):
        for argument,value in [('--epochs','0'),('--batch-size','0'),('--batch-size','1')]:
            with self.subTest(argument=argument,value=value),patch('sys.argv',['example','--root-path','unused',argument,value]),patch.object(EXAMPLE,'ADAuditoryDataset') as dataset:
                with self.assertRaises(SystemExit):
                    EXAMPLE.main()
                dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
