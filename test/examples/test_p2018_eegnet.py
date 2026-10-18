from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal
import torch
from torch.nn import functional as F

from test.datasets.test_p2018 import write_record
from torcheeg.models import EEGNet


PATH = Path(__file__).resolve().parents[2] / 'examples/examples_p2018_eegnet.py'
SPEC = importlib.util.spec_from_file_location('p2018_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class TestP2018EEGNetExample(unittest.TestCase):
    def run_example(self, root, records=3, origin=30, source_fs=128):
        events = [(0, 'W'), (30, 'N1'), (60, 'undefined'), (90, 'N2'), (120, 'N3'), (150, 'R')]
        events = [(second + origin, stage) for second, stage in events]
        if origin:
            events = [(3, '(arousal_rera'), (10, 'arousal_rera)')] + events
        physical = None
        for index in range(1, records + 1):
            _, physical = write_record(root, name=f'tr03-{index:04d}', fs=source_fs,
                                       seconds=origin + 185, events=events)
        args = ['example', '--root-path', str(root), '--cache-path', str(root / 'cache'),
                '--output-path', str(root / 'model.pt'), '--epochs', '1', '--batch-size', '4']
        with patch('sys.argv', args), redirect_stdout(io.StringIO()):
            EXAMPLE.main()
        return json.loads((root / 'model.pt.json').read_text()), physical

    def test_real_wfdb_pipeline_training_metrics_and_export(self):
        initial = {}
        def observe(**kwargs):
            model = EEGNet(**kwargs)
            if not initial:
                initial.update({name: value.clone() for name, value in model.state_dict().items()})
            return model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(EXAMPLE, 'EEGNet', side_effect=observe) as constructor:
                report, physical = self.run_example(root)
            self.assertEqual((report['train_epochs'], report['test_epochs'], report['optimizer_steps']), (10, 5, 3))
            self.assertEqual(report['partitions'], dict(train_groups=['tr03-0001', 'tr03-0002'],
                  test_groups=['tr03-0003'], train_records=['tr03-0001', 'tr03-0002'], test_records=['tr03-0003']))
            provenance = report['preprocessing']['channel_provenance']
            self.assertEqual(set(provenance), {'tr03-0001', 'tr03-0002', 'tr03-0003'})
            self.assertEqual(provenance['tr03-0003'], dict(channel_names=['C3-M2', 'E1-M2'], channel_units=['uV', 'mV']))
            payload = torch.load(root / 'model.pt', weights_only=True)
            self.assertEqual(json.loads(payload['preprocessing_json']), report['preprocessing'])
            self.assertEqual(json.loads(payload['partitions_json']), report['partitions'])
            self.assertEqual(constructor.call_args.kwargs, EXAMPLE.MODEL_CONFIG)
            self.assertEqual(set(json.loads(payload['model_config_json'])), set(inspect.signature(EEGNet).parameters))
            self.assertEqual(len(payload['state_dict']), 20)
            self.assertTrue(report['strict_reload_equal'])
            for name in ('block1.0.weight', 'lin.weight'):
                self.assertFalse(torch.equal(initial[name], payload['state_dict'][name]), name)
            model = EEGNet(**json.loads(payload['model_config_json'])).requires_grad_(False).eval()
            model.load_state_dict(payload['state_dict'], strict=True)
            values = signal.resample(physical, int(len(physical) * 100 / 128), axis=0)
            sos = signal.butter(8, [.5, 30], btype='bandpass', fs=100, output='sos')
            values = signal.sosfiltfilt(sos, values, axis=0)
            values = np.stack([values[start:start + 3000].T for start in [3000, 6000, 12000, 15000, 18000]])
            values = (values - values.mean(-1, keepdims=True)) / values.std(-1, keepdims=True)
            eeg = torch.tensor(values[:, None], dtype=torch.float32)
            labels = torch.arange(5)
            with torch.no_grad():
                logits = model(eeg)
            self.assertAlmostEqual(report['test_metrics']['test_loss'], F.cross_entropy(logits, labels).item(), places=5)
            self.assertAlmostEqual(report['test_metrics']['test_accuracy'], (logits.argmax(1) == labels).float().mean().item(), places=6)

    def test_zero_stage_origin_and_actual_200hz_header(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = self.run_example(Path(directory), origin=0, source_fs=200)
            self.assertEqual((report['train_epochs'], report['test_epochs'], report['optimizer_steps']), (10, 5, 3))

    def test_single_record_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(EXAMPLE, 'ClassifierTrainer') as trainer:
            with self.assertRaisesRegex(ValueError, 'two complete'):
                self.run_example(Path(directory), records=1)
            trainer.assert_not_called()

    def test_existing_model_or_sidecar_is_preserved(self):
        for filename in ['model.pt', 'model.pt.json']:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / filename).write_bytes(b'existing owned output')
                with self.assertRaisesRegex(ValueError, 'fresh output'):
                    self.run_example(root)
                self.assertEqual((root / filename).read_bytes(), b'existing owned output')

    def test_channel_units_and_order_must_be_known_and_constant(self):
        import pandas as pd
        good = dict(record_id='tr03-0001', channel_names=json.dumps(['C3-M2', 'E1-M2']),
                    channel_units=json.dumps(['uV', 'mV']))
        for change in [dict(channel_units=json.dumps([None, 'mV'])),
                       dict(channel_units=json.dumps(['uV'])),
                       dict(channel_names=json.dumps(['E1-M2', 'C3-M2']))]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                EXAMPLE.channel_provenance(SimpleNamespace(info=pd.DataFrame([dict(good, **change)])))
        with self.assertRaisesRegex(ValueError, 'constant'):
            EXAMPLE.channel_provenance(SimpleNamespace(info=pd.DataFrame([good, dict(good, channel_units=json.dumps(['uV', 'V']))])))

    def test_invalid_cli_fails_without_dataset_work(self):
        for argument in ['--epochs', '--batch-size']:
            args = ['example', '--root-path', 'unused', argument, '0']
            with self.subTest(argument=argument), patch('sys.argv', args), patch.object(EXAMPLE, 'P2018Dataset') as dataset:
                with self.assertRaises(SystemExit):
                    EXAMPLE.main()
                dataset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
