import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np

from torcheeg import transforms
from torcheeg.datasets import MentalArithmeticDataset
from torcheeg.datasets.constants import (MENTAL_ARITHMETIC_ADJACENCY_MATRIX,
                                         MENTAL_ARITHMETIC_CHANNEL_LIST,
                                         MENTAL_ARITHMETIC_CHANNEL_LOCATION_DICT)
from torcheeg.datasets.module.mental_workload.mental_arithmetic import SELECTED_CHANNELS


class TestMentalArithmetic(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name, 'source')
        self.root.mkdir()
        self.record = ('Subject00_1.edf', 0, 'Subject00')
        self.eeg = np.arange(20 * 2400, dtype=float).reshape(20, 2400)

    def process(self, **kwargs):
        params = {'record': self.record, 'eeg_data': self.eeg, 'label': 0}
        params.update(kwargs)
        return list(MentalArithmeticDataset.process_record(**params))

    def test_default_windows_preserve_signals_units_and_metadata(self):
        items = self.process()
        self.assertEqual(len(items), 2)
        for index, item in enumerate(items):
            np.testing.assert_array_equal(item['eeg'], self.eeg[:19, index * 1000:(index + 1) * 1000])
            self.assertEqual(item['info'], {'subject_id': 'Subject00', 'trial_id': 1,
                'label': 0, 'clip_id': f'Subject00_1_{index}',
                'start_at': index * 1000, 'end_at': (index + 1) * 1000})
            self.assertEqual(item['key'], item['info']['clip_id'])

    def test_custom_windows_span_full_recording_without_hidden_five_second_blocks(self):
        items = self.process(chunk_size=800)
        self.assertEqual([item['info']['start_at'] for item in items], [0, 800, 1600])
        np.testing.assert_array_equal(np.concatenate([item['eeg'] for item in items], axis=-1), self.eeg[:19])

    def test_overlap_and_partial_final_window(self):
        items = self.process(chunk_size=400, overlap=200)
        self.assertEqual(len(items), 11)
        self.assertEqual(items[-1]['info']['end_at'], 2400)
        np.testing.assert_array_equal(items[0]['eeg'][:, 200:], items[1]['eeg'][:, :200])
        self.assertEqual(self.process(chunk_size=2500), [])

    def test_whole_recording_is_one_sample(self):
        items = self.process(chunk_size=-1, num_channel=20)
        self.assertEqual(len(items), 1)
        np.testing.assert_array_equal(items[0]['eeg'], self.eeg)
        self.assertEqual(items[0]['info']['end_at'], 2400)

    def test_before_trial_runs_once_and_offline_transform_runs_per_window(self):
        original = self.eeg.copy()
        calls = []
        def before_trial(eeg):
            calls.append(eeg.shape)
            eeg += 1
            return eeg
        items = self.process(chunk_size=800, before_trial=before_trial,
            offline_transform=lambda eeg: {'eeg': eeg * 2})
        self.assertEqual(calls, [(19, 2400)])
        np.testing.assert_array_equal(items[2]['eeg'], (self.eeg[:19, 1600:] + 1) * 2)
        np.testing.assert_array_equal(self.eeg, original)

    def test_hooks_must_preserve_time_coordinates_and_finite_values(self):
        for hook in (lambda eeg: eeg[:, :-1], lambda eeg: eeg * np.nan):
            with self.subTest(hook=hook), self.assertRaises(ValueError):
                self.process(before_trial=hook)

    def test_yielded_windows_do_not_share_mutable_source_storage(self):
        items = self.process(chunk_size=1000)
        original = self.eeg.copy()
        items[0]['eeg'][:] = -1
        np.testing.assert_array_equal(self.eeg, original)
        self.assertTrue((items[1]['eeg'] >= 0).all())

    def test_sessions_have_distinct_labels_trial_ids_and_clip_keys(self):
        background = self.process()[0]
        arithmetic = self.process(record=('Subject00_2.edf', 1, 'Subject00'), label=1)[0]
        self.assertEqual(arithmetic['info']['label'], 1)
        self.assertEqual(arithmetic['info']['trial_id'], 2)
        self.assertNotEqual(background['key'], arithmetic['key'])

    def test_invalid_window_and_channel_parameters_fail_before_io(self):
        cases = [{'chunk_size': n} for n in (0, -2, 1.5, True)]
        cases += [{'overlap': n} for n in (-1, 1000, 2000)]
        cases += [{'chunk_size': -1, 'overlap': 1}]
        cases += [{'num_channel': n} for n in (0, 21)]
        cases += [{'sampling_rate': n} for n in (0, -1, False)]
        for params in cases:
            with self.subTest(params=params), self.assertRaises(ValueError):
                MentalArithmeticDataset(root_path=str(self.root), io_path=str(self.root / 'never'), **params)
        self.assertFalse((self.root / 'never').exists())

    def test_invalid_signals_and_mismatched_labels_are_rejected(self):
        for eeg in (np.empty((19, 0)), np.empty((2, 20)), np.ones(20),
                    np.full((19, 20), np.inf), np.ones((19, 20), dtype=complex),
                    np.full((19, 20), 'bad')):
            with self.subTest(shape=eeg.shape), self.assertRaises(ValueError):
                self.process(eeg_data=eeg)
        with self.assertRaises(ValueError):
            self.process(label=1)

    def test_file_discovery_filters_sidecars_and_preserves_subject_ids(self):
        for name in ('Subject02_2.edf', 'Subject00_1.edf', 'Subject00_2.EDF', '.DS_Store', 'subject-info.csv'):
            (self.root / name).touch()
        (self.root / 'Subject99_1.edf').mkdir()
        dataset = MentalArithmeticDataset.__new__(MentalArithmeticDataset)
        records = dataset.set_records(str(self.root))
        self.assertEqual([(r[1], r[2]) for r in records], [(0, 'Subject00'), (1, 'Subject00'), (1, 'Subject02')])

    def test_invalid_or_empty_source_directory_has_a_clear_error(self):
        dataset = MentalArithmeticDataset.__new__(MentalArithmeticDataset)
        with self.assertRaises(FileNotFoundError):
            dataset.set_records(str(self.root / 'missing'))
        with self.assertRaisesRegex(ValueError, 'No MentalArithmetic'):
            dataset.set_records(str(self.root))
        (self.root / 'Subject00_3.edf').touch()
        with self.assertRaisesRegex(ValueError, 'Unrecognized'):
            dataset.set_records(str(self.root))

    def test_edf_reader_orders_channels_resamples_and_returns_microvolts(self):
        channels = list(reversed(SELECTED_CHANNELS)) + ['ECG ECG']
        info = mne.create_info(channels, 500, ch_types=['eeg'] * 20 + ['ecg'])
        values = [float(SELECTED_CHANNELS.index(name) + 1) if name in SELECTED_CHANNELS else 99. for name in channels]
        raw = mne.io.RawArray(np.repeat(np.array(values)[:, None] * 1e-6, 1000, axis=1), info, verbose='ERROR')
        with patch('torcheeg.datasets.module.mental_workload.mental_arithmetic.mne.io.read_raw_edf', return_value=raw) as reader:
            result = MentalArithmeticDataset.read_record(self.record)
        self.assertEqual(result['eeg_data'].shape, (19, 400))
        np.testing.assert_allclose(result['eeg_data'], np.repeat(np.arange(1, 20)[:, None], 400, axis=1), atol=1e-9)
        self.assertEqual(result['label'], 0)
        self.assertEqual(reader.call_args.args[0], self.record[0])

    def test_reference_channel_is_explicit_and_keeps_microvolt_units(self):
        raw = mne.io.RawArray(np.full((20, 500), 50e-6),
            mne.create_info(SELECTED_CHANNELS, 500, 'eeg'), verbose='ERROR')
        with patch('torcheeg.datasets.module.mental_workload.mental_arithmetic.mne.io.read_raw_edf', return_value=raw):
            result = MentalArithmeticDataset.read_record(self.record, num_channel=20, sampling_rate=500)
        self.assertEqual(result['eeg_data'].shape, (20, 500))
        np.testing.assert_allclose(result['eeg_data'], 50.)

    def test_lmdb_backend_reads_cached_windows_and_labels(self):
        for session in (1, 2):
            (self.root / f'Subject00_{session}.edf').touch()
        def read_record(record, **kwargs):
            return {'eeg_data': self.eeg.copy(), 'label': record[1]}
        with patch.object(MentalArithmeticDataset, 'read_record', side_effect=read_record):
            dataset = MentalArithmeticDataset(root_path=str(self.root),
                io_path=str(Path(self.directory.name, 'lmdb-cache')), verbose=False)
        self.assertEqual(len(dataset), 4)
        signal, info = dataset[2]
        np.testing.assert_array_equal(signal, self.eeg[:19, :1000])
        self.assertEqual(info['label'], 1)
        self.assertEqual(info['subject_id'], 'Subject00')

    def test_edf_reader_rejects_missing_channels(self):
        raw = mne.io.RawArray(np.ones((1, 100)), mne.create_info(['EEG Fp1'], 500, 'eeg'), verbose='ERROR')
        with patch('torcheeg.datasets.module.mental_workload.mental_arithmetic.mne.io.read_raw_edf', return_value=raw):
            with self.assertRaisesRegex(ValueError, 'Missing MentalArithmetic'):
                MentalArithmeticDataset.read_record(self.record)

    def test_constants_describe_only_the_nineteen_scalp_electrodes(self):
        self.assertEqual(SELECTED_CHANNELS[:19], ['EEG ' + name for name in MENTAL_ARITHMETIC_CHANNEL_LIST])
        self.assertEqual(len(MENTAL_ARITHMETIC_CHANNEL_LIST), 19)
        self.assertNotIn('A2-A1', MENTAL_ARITHMETIC_CHANNEL_LOCATION_DICT)
        self.assertEqual(np.asarray(MENTAL_ARITHMETIC_ADJACENCY_MATRIX).shape, (19, 19))
        coordinates = list(MENTAL_ARITHMETIC_CHANNEL_LOCATION_DICT.values())
        self.assertEqual(len({tuple(c) for c in coordinates}), 19)

    def test_pickle_cache_roundtrip_and_online_labels(self):
        for session in (1, 2):
            (self.root / f'Subject00_{session}.edf').touch()
        def read_record(record, **kwargs):
            return {'eeg_data': self.eeg.copy(), 'label': record[1]}
        parameters = {'root_path': str(self.root), 'io_path': str(Path(self.directory.name, 'cache')),
                      'io_mode': 'pickle', 'num_worker': 0, 'verbose': False,
                      'online_transform': transforms.ToTensor(),
                      'label_transform': transforms.Select('label')}
        with patch.object(MentalArithmeticDataset, 'read_record', side_effect=read_record):
            dataset = MentalArithmeticDataset(**parameters)
        self.assertEqual(len(dataset), 4)
        signal, label = dataset[2]
        self.assertEqual(tuple(signal.shape), (19, 1000))
        self.assertEqual(label, 1)
        with patch.object(MentalArithmeticDataset, 'read_record', side_effect=AssertionError('Cache reread source')):
            reused = MentalArithmeticDataset(**parameters)
        np.testing.assert_allclose(reused[2][0].numpy(), signal.numpy())
        self.assertEqual(reused[2][1], label)
        self.assertIn('sampling_rate=200', repr(reused))

    def test_fake_record_does_not_change_numpy_global_state(self):
        np.random.seed(71)
        before = np.random.get_state()
        first, second = MentalArithmeticDataset.fake_record(), MentalArithmeticDataset.fake_record()
        for a, b in zip(before, np.random.get_state()):
            np.testing.assert_equal(a, b)
        np.testing.assert_array_equal(first['eeg_data'], second['eeg_data'])
        self.assertEqual(self.process(**first)[0]['eeg'].shape, (19, 1000))


if __name__ == '__main__':
    unittest.main()
