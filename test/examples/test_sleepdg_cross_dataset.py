import importlib.util
from contextlib import redirect_stdout
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from torcheeg.datasets import SequenceDataset


PATH = Path(__file__).resolve().parents[2] / 'examples/examples_sleepdg_cross_dataset.py'
SPEC = importlib.util.spec_from_file_location('sleepdg_example', PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class Epochs(Dataset):
    def __init__(self, subjects=3, epochs=45, gap=False, constant=False):
        rows, arrays = [], []
        for subject in range(subjects):
            for night in (1, 2):
                for epoch in range(epochs):
                    onset = 30. * (epoch + (1 if gap and epoch >= 20 else 0))
                    label = 'Sleep stage W' if epoch < 5 or epoch >= 35 else 'Sleep stage N2'
                    record = f'{subject}-{night}'
                    rows.append(dict(subject_id=subject, session_id=night, record_id=record,
                                     _record_id=record, trial_id=record, clip_id=f'{record}_{epoch}',
                                     start_at=onset, end_at=onset + 30, label=label))
                    values = np.zeros((2, 3000)) if constant else np.stack([
                        np.arange(3000) / 3000 + epoch + subject,
                        -np.arange(3000) / 3000 + 2 * epoch - subject])
                    arrays.append(values)
        self.info = pd.DataFrame(rows)
        self.values = {row['clip_id']: value for row, value in zip(rows, arrays)}

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        info = self.info.iloc[index]
        return self.values[info['clip_id']], EXAMPLE.LABELS[info['label']]


class TestSleepDGWorkflow(unittest.TestCase):
    def test_subject_split_is_deterministic_and_keeps_repeated_visits_together(self):
        original = Epochs()
        train, valid = EXAMPLE.subject_partitions(original, 42)
        again = EXAMPLE.subject_partitions(original, 42)
        self.assertFalse(set(train.info.subject_id) & set(valid.info.subject_id))
        self.assertEqual(len(train) + len(valid), len(original))
        self.assertEqual(set(valid.info.session_id), {1, 2})
        pd.testing.assert_frame_equal(valid.info, again[1].info)
        train.info.loc[0, 'label'] = 'changed'
        self.assertNotEqual(original.info.loc[0, 'label'], 'changed')
        with self.assertRaisesRegex(ValueError, 'two participants'):
            EXAMPLE.subject_partitions(Epochs(subjects=1), 42)

    def test_complete_windows_preserve_gaps_drop_tails_and_do_not_mutate_source(self):
        original = Epochs(subjects=1, gap=True)
        before = original.info.copy(deep=True)
        view, dropped = EXAMPLE.complete_epochs(original)
        self.assertEqual(len(view), 80)
        self.assertEqual(dropped, [])
        sequences = SequenceDataset(view, sequence_length=20, stride=20)
        self.assertEqual(len(sequences), 4)
        self.assertEqual(sequences.info.start_at.tolist(), [0., 630., 0., 630.])
        pd.testing.assert_frame_equal(before, original.info)

    def test_short_records_are_reported_and_empty_partition_rejected(self):
        original = Epochs(subjects=1, epochs=10)
        with self.assertRaisesRegex(ValueError, 'No complete'):
            EXAMPLE.complete_epochs(original)
        original = Epochs(subjects=1)
        original.info = original.info.loc[(original.info.session_id == 1) | (original.info.start_at < 300)]
        view, dropped = EXAMPLE.complete_epochs(original)
        self.assertEqual(len(view), 40)
        self.assertEqual(dropped, ['0-2'])

    def test_wake_trim_has_exact_epoch_margins_and_rejects_all_wake(self):
        original = Epochs(subjects=1, epochs=100)
        # One 20-epoch sleep run followed by long wake; margin zero keeps it.
        original.info['label'] = 'Sleep stage W'
        original.info.loc[(original.info.start_at >= 600) & (original.info.start_at < 1200), 'label'] = 'Sleep stage N2'
        view, _ = EXAMPLE.complete_epochs(original, wake_margin=0)
        self.assertEqual(len(view), 40)
        self.assertEqual(view.info.start_at.min(), 600.)
        self.assertEqual(view.info.end_at.max(), 1200.)
        view, _ = EXAMPLE.complete_epochs(original, wake_margin=10)
        self.assertEqual(len(view), 80)
        self.assertEqual(view.info.start_at.min(), 300.)
        self.assertEqual(view.info.end_at.max(), 1500.)
        original.info['label'] = 'Sleep stage W'
        with self.assertRaisesRegex(ValueError, 'no scored sleep'):
            EXAMPLE.complete_epochs(original, wake_margin=60)

    def test_streaming_record_statistics_match_independent_population_moments(self):
        original = Epochs(subjects=2)
        view, _ = EXAMPLE.complete_epochs(original)
        normalized = EXAMPLE.RecordNormalizedDataset(view)
        for record, group in view.info.groupby('_record_id', sort=False):
            values = np.concatenate([view[int(index)][0] for index in group.index], axis=1)
            stats = normalized.statistics[record]
            np.testing.assert_allclose(stats['mean'], values.mean(axis=1), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(stats['scale'], values.std(axis=1, ddof=0), rtol=1e-13, atol=1e-13)
            self.assertEqual(stats['samples'], len(group) * 3000)
            actual = np.concatenate([normalized[int(index)][0].numpy() for index in group.index], axis=1)
            np.testing.assert_allclose(actual.mean(axis=1), 0, atol=2e-7)
            np.testing.assert_allclose(actual.std(axis=1), 1, atol=2e-7)
        self.assertNotIn('normalization', original.info)

    def test_normalization_is_unit_scale_invariant_and_handles_constant_channels(self):
        first = Epochs(subjects=1, epochs=20)
        second = Epochs(subjects=1, epochs=20)
        second.values = {key: value * 1e-6 for key, value in second.values.items()}
        one, two = EXAMPLE.RecordNormalizedDataset(first), EXAMPLE.RecordNormalizedDataset(second)
        for i in range(len(one)):
            torch.testing.assert_close(one[i][0], two[i][0], rtol=0, atol=0)
        constant = EXAMPLE.RecordNormalizedDataset(Epochs(subjects=1, epochs=20, constant=True))
        self.assertTrue(torch.equal(constant[0][0], torch.zeros(2, 3000)))

    def test_invalid_geometry_and_nonfinite_signals_fail_before_training(self):
        for value in [np.zeros((2, 2999)), np.zeros((1, 3000)), np.full((2, 3000), np.nan)]:
            original = Epochs(subjects=1, epochs=20)
            original.values[next(iter(original.values))] = value
            with self.assertRaisesRegex(ValueError, 'finite'):
                EXAMPLE.RecordNormalizedDataset(original)

    def test_domain_codes_are_dataset_constants_and_target_has_no_source_code(self):
        dataset = Epochs(subjects=2, epochs=20)
        source, metadata = EXAMPLE.prepare_sequences(dataset, 'hmc', domain=3)
        eeg, labels, domains = next(iter(DataLoader(source, batch_size=2)))
        self.assertEqual(tuple(eeg.shape), (2, 20, 2, 3000))
        self.assertEqual(tuple(labels.shape), (2, 20))
        self.assertTrue(torch.equal(domains, torch.full((2,), 3)))
        target, _ = EXAMPLE.prepare_sequences(dataset, 'shhs')
        self.assertEqual(len(target[0]), 2)
        self.assertEqual(metadata['subjects'], ['0', '1'])

    def test_sampler_balances_domains_and_covers_largest_source(self):
        sampler = EXAMPLE.SourceBatchSampler([2, 7, 4, 3], batch_size=8, seed=42)
        self.assertEqual(len(sampler), 4)
        batches = list(sampler)
        offsets = np.cumsum([0, 2, 7, 4, 3])
        for batch in batches:
            self.assertEqual(len(batch), 8)
            for left, right in zip(offsets[:-1], offsets[1:]):
                self.assertEqual(sum(left <= value < right for value in batch), 2)
        self.assertEqual(set(range(2, 9)), {index for batch in batches for index in batch if 2 <= index < 9})
        self.assertEqual(batches, list(EXAMPLE.SourceBatchSampler([2, 7, 4, 3], 8, 42)))
        self.assertNotEqual(batches, list(sampler))
        for lengths, size in [([], 4), ([1, 0], 4), ([1, 1, 1, 1], 3), ([1, 1, 1, 1], 5)]:
            with self.assertRaises(ValueError):
                EXAMPLE.SourceBatchSampler(lengths, size, 42)

    def test_checkpoint_binding_rejects_other_protocols(self):
        configuration = dict(target='shhs', scheduler_steps=2)
        state = dict(epoch=0)
        EXAMPLE.WorkflowState(configuration).on_save_checkpoint(None, None, state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owned.ckpt'
            torch.save(state, path)
            self.assertEqual(EXAMPLE.resume_epoch(path, configuration), 1)
            with self.assertRaisesRegex(ValueError, 'different workflow'):
                EXAMPLE.resume_epoch(path, dict(target='p2018', scheduler_steps=2))

    def test_native_workflow_trains_selects_exports_and_resumes_its_owned_checkpoint(self):
        initial = {}
        native = EXAMPLE.SleepDG
        watched = ('ae.encoder.epoch_encoder.encoder.0.weight', 'ae.encoder.fc_mu.weight',
                   'ae.decoder.upsample.0.weight', 'classifier.weight')
        def model(**kwargs):
            self.assertEqual(kwargs, EXAMPLE.MODEL)
            result = native(**kwargs)
            if not initial:
                initial.update({key: result.state_dict()[key].clone() for key in watched})
            return result
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = ['example', '--output-dir', directory, '--epochs', '1', '--batch-size', '4',
                    '--train-batches', '1', '--val-batches', '1', '--test-batches', '1',
                    '--scheduler-steps', '2']
            for name in EXAMPLE.SOURCES:
                args.extend([f'--{name}-root', str(output / name)])
            def source(name, root, cache):
                return Epochs(subjects=2, epochs=20)
            with patch.object(EXAMPLE, 'source_dataset', side_effect=source), \
                    patch.object(EXAMPLE, 'SleepDG', side_effect=model), \
                    patch('sys.argv', args), redirect_stdout(io.StringIO()):
                EXAMPLE.main()
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['target'], 'shhs')
            self.assertNotIn('shhs', report['source_domains'])
            self.assertEqual(set(report['source_domains'].values()), {0, 1, 2, 3})
            self.assertEqual(report['training_optimizer_steps'], 1)
            self.assertTrue(report['strict_reload_equal'])
            for name in report['source_domains']:
                partitions = report['partitions'][name]
                self.assertFalse(set(partitions['train']['subjects']) & set(partitions['validation']['subjects']))
            payload = torch.load(output / 'model.pt', weights_only=True)
            self.assertEqual(json.loads(payload['model_config_json'])['hid_channels'], 512)
            self.assertEqual(json.loads(payload['model_config_json'])['sequence_length'], 20)
            self.assertEqual(len(payload['state_dict']), 74)
            for key in watched:
                self.assertFalse(torch.equal(initial[key], payload['state_dict'][key]), key)
            last = report['last_checkpoint']
            resumed = args.copy()
            resumed[resumed.index('--epochs') + 1] = '2'
            resumed.extend(['--resume', last])
            with patch.object(EXAMPLE, 'source_dataset', side_effect=source), \
                    patch('sys.argv', resumed), redirect_stdout(io.StringIO()):
                EXAMPLE.main()
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['training_optimizer_steps'], 2)
            self.assertTrue(report['strict_reload_equal'])
            state = torch.load(report['last_checkpoint'], weights_only=False)
            self.assertEqual(state['lr_schedulers'][0]['last_epoch'], 2)
            self.assertEqual(state['lr_schedulers'][0]['T_max'], 2)
            optimizer = state['optimizer_states'][0]['state']
            self.assertTrue(all(float(item['step']) == 2 for item in optimizer.values()))

    def test_all_five_reader_settings_are_explicit_and_do_not_request_zero_filling(self):
        mapping = dict(sleepedf='SleepEDFxDataset', hmc='HMCDataset', isruc='ISRUCDataset',
                       shhs='SHHSDataset', p2018='P2018Dataset')
        for source, name in mapping.items():
            with self.subTest(source=source), patch.object(EXAMPLE, name) as constructor:
                EXAMPLE.source_dataset(source, 'source', Path('cache'))
                args = constructor.call_args.kwargs
                self.assertEqual(args['sfreq'], 100)
                self.assertEqual(len(args['channels']), 2)
                self.assertNotIn('online_transform', args)
                if source in ('sleepedf', 'hmc', 'isruc'):
                    self.assertEqual(args['missing_channels'], 'raise')
                if source == 'sleepedf':
                    self.assertEqual(args['studies'], ['cassette'])
                if source == 'isruc':
                    self.assertEqual(args['groups'], [0])
                if source == 'hmc':
                    self.assertIsNone(args['l_freq'])
                    self.assertIsNone(args['h_freq'])
                    self.assertEqual(args['start_at'], 0)
                    self.assertIsNone(args['end_at'])
        with self.assertRaises(ValueError):
            EXAMPLE.source_dataset('unknown', '', Path('cache'))


if __name__ == '__main__':
    unittest.main()
