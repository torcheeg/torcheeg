import copy
from datetime import datetime, timezone
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import mne
import numpy as np

from torcheeg.datasets import HMCDataset
from torcheeg.datasets.module.sleep_stage_detection.hmc import crop, interploate


PAIR = ('SN001.edf', 'SN001_sleepscoring.edf')
MODULE = 'torcheeg.datasets.module.sleep_stage_detection.hmc'


def raw_record(first_samp=0):
    sfreq = 100
    t = np.arange(180 * sfreq) / sfreq
    signal = np.stack([np.sin(t), np.cos(t)]) * 1e-6
    return mne.io.RawArray(signal, mne.create_info(['A', 'B'], sfreq, 'eeg'),
                           first_samp=first_samp, verbose=False)


def annotations():
    return mne.Annotations([0, 0, 30, 60, 90, 150], [180, 30, 30, 30, 60, 30],
                           ['Lights off@@EEG F4-A1', 'Sleep stage W', 'Sleep stage N1',
                            'Movement', 'Sleep stage N2', 'Sleep stage R'])


class TestHMCDataset(unittest.TestCase):
    def read(self, **kwargs):
        settings = dict(root_path='unused', channels=['B', 'A'], l_freq=None, h_freq=None,
                        sfreq=100, start_at=30, end_at=180)
        settings.update(kwargs)
        with patch(MODULE + '.read_raw_edf', return_value=raw_record()), \
                patch(MODULE + '.mne.read_annotations', return_value=annotations()):
            return HMCDataset.read_record(PAIR, **settings)

    def test_scored_events_ignore_markers_and_retain_original_crop_times_and_gaps(self):
        result = self.read()
        np.testing.assert_equal(result['epochs_onsets'], [30., 90., 120., 150.])
        self.assertEqual(result['epochs_label'], ['Sleep stage N1', 'Sleep stage N2',
                                                 'Sleep stage N2', 'Sleep stage R'])
        expected = raw_record().get_data()[[1, 0]]
        for signal, onset in zip(result['epochs_data'], result['epochs_onsets']):
            begin = int(onset * 100)
            np.testing.assert_allclose(signal, expected[:, begin:begin + 3000], rtol=0, atol=0)
        self.assertEqual(result['epochs_data'].shape, (4, 2, 3000))

    def test_partial_crop_does_not_shift_the_original_scoring_grid(self):
        result = self.read(start_at=15, end_at=149)
        np.testing.assert_equal(result['epochs_onsets'], [30., 90.])
        self.assertEqual(result['epochs_label'], ['Sleep stage N1', 'Sleep stage N2'])

    def test_whole_record_and_available_end_include_the_last_complete_epoch(self):
        whole = self.read(start_at=0, end_at=None)
        with self.assertWarnsRegex(UserWarning, 'recording ends'):
            truncated = self.read(start_at=0, end_at=300)
        for key in whole:
            np.testing.assert_equal(whole[key], truncated[key])
        np.testing.assert_equal(whole['epochs_onsets'], [0., 30., 90., 120., 150.])

    def test_resampling_keeps_times_labels_and_requested_channel_order(self):
        result = self.read(sfreq=50)
        reference = raw_record().resample(50).get_data()[[1, 0]]
        self.assertEqual(result['epochs_data'].shape, (4, 2, 1500))
        np.testing.assert_equal(result['epochs_onsets'], [30., 90., 120., 150.])
        for signal, onset in zip(result['epochs_data'], result['epochs_onsets']):
            begin = int(onset * 50)
            np.testing.assert_allclose(signal, reference[:, begin:begin + 1500], rtol=0, atol=0)

    def test_nonzero_sample_origin_survives_resampling_and_crop(self):
        with patch(MODULE + '.read_raw_edf', return_value=raw_record(first_samp=1000)), \
                patch(MODULE + '.mne.read_annotations', return_value=annotations()):
            result = HMCDataset.read_record(PAIR, root_path='unused', channels=['A'],
                                            l_freq=None, h_freq=None, sfreq=50,
                                            start_at=30, end_at=180)
        np.testing.assert_equal(result['epochs_onsets'], [30., 90., 120., 150.])
        self.assertEqual(result['epochs_data'].shape, (4, 1, 1500))

    def test_absolute_annotation_origin_keeps_original_relative_times(self):
        measurement = datetime(2000, 1, 1, tzinfo=timezone.utc)
        raw = raw_record(first_samp=1000).set_meas_date(measurement)
        original = annotations()
        scores = mne.Annotations(original.onset + 10., original.duration,
                                 original.description, orig_time=measurement)
        with patch(MODULE + '.read_raw_edf', return_value=raw), \
                patch(MODULE + '.mne.read_annotations', return_value=scores):
            result = HMCDataset.read_record(PAIR, root_path='unused', channels=['A'],
                                            l_freq=None, h_freq=None, sfreq=50,
                                            start_at=30, end_at=180)
        np.testing.assert_equal(result['epochs_onsets'], [30., 90., 120., 150.])

    def test_constructor_rejects_invalid_crop_and_channel_policy_before_cache_work(self):
        for args in [dict(start_at=-1), dict(end_at=20), dict(end_at=float('nan')),
                     dict(missing_channels='invalid')]:
            with self.subTest(args=args), patch(MODULE + '.BaseDataset.__init__') as parent:
                with self.assertRaises(ValueError):
                    HMCDataset(root_path='unused', io_path='unused', **args)
                parent.assert_not_called()

    def test_pairing_uses_record_identifiers_and_preserves_legacy_score_spelling(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ['SN002.sleepscoring.edf', 'SN001_sleepscoring.edf',
                         'SN001.edf', 'SN002.edf', 'SN003_sleepscoring.txt', 'README.txt']:
                Path(root, name).touch()
            Path(root, 'SN003.edf').mkdir()
            pairs = HMCDataset.set_records(None, root)
            self.assertEqual(pairs, [PAIR, ('SN002.edf', 'SN002.sleepscoring.edf')])

    def test_missing_or_ambiguous_pairs_raise_instead_of_dropping_later_records(self):
        cases = [['SN001.edf', 'SN002.edf', 'SN002_sleepscoring.edf'],
                 ['SN001_sleepscoring.edf'],
                 ['SN001.edf', 'SN001.sleepscoring.edf', 'SN001_sleepscoring.edf'],
                 ['SN001.edf', 'SN001_sleepscoring.edf', 'other.edf']]
        with tempfile.TemporaryDirectory() as root:
            for i, names in enumerate(cases):
                folder = Path(root, str(i))
                folder.mkdir()
                for name in names:
                    (folder / name).touch()
                with self.subTest(names=names), self.assertRaises(ValueError):
                    HMCDataset.set_records(None, str(folder))

    def test_mismatched_record_pair_fails_before_io(self):
        with patch(MODULE + '.read_raw_edf') as reader:
            for pair in [('SN001.edf', 'SN002_sleepscoring.edf'),
                         ('wrong.edf', 'SN001_sleepscoring.edf')]:
                with self.assertRaises(ValueError):
                    HMCDataset.read_record(pair)
            reader.assert_not_called()

    def test_missing_channel_policy_order_and_compatibility_default(self):
        raw = raw_record()
        output = interploate(raw, ['B', 'missing', 'A'])
        self.assertEqual(output.ch_names, ['B', 'missing', 'A'])
        np.testing.assert_equal(output.get_data()[1], 0)
        for channels, policy in [(['missing'], 'raise'), (['A', 'A'], 'zero'),
                                 ([], 'zero'), (['A'], 'bad')]:
            with self.subTest(channels=channels, policy=policy), self.assertRaises(ValueError):
                interploate(raw_record(), channels, policy)
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            self.read(channels=['missing'], missing_channels='raise')

    def test_invalid_or_empty_intervals_are_explicit(self):
        for start, end in [(False, None), (-1, None), (0, 0), (3, 2), (np.nan, None),
                           (0, np.inf), ('0', None), (180, None)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                crop(raw_record(), start, end)
        with self.assertRaisesRegex(ValueError, 'no complete scored'):
            self.read(start_at=60, end_at=90)
        with patch(MODULE + '.read_raw_edf', return_value=raw_record()), \
                patch(MODULE + '.mne.read_annotations', return_value=mne.Annotations([0], [180], ['Unknown'])):
            with self.assertRaisesRegex(ValueError, 'No complete scored'):
                HMCDataset.read_record(PAIR, start_at=0, end_at=None, channels=['A'],
                                       l_freq=None, h_freq=None)

    def test_process_record_preserves_metadata_and_applies_hooks(self):
        result = self.read()
        result['before_trial'] = lambda x: x * 2
        result['offline_transform'] = lambda eeg: {'eeg': eeg + 1}
        rows = list(HMCDataset.process_record(PAIR, **result))
        for i, row in enumerate(rows):
            self.assertEqual(row['key'], f'SN001_{i}')
            self.assertEqual(row['info']['subject_id'], 'SN001')
            self.assertEqual(row['info']['record_id'], 'SN001')
            self.assertEqual(row['info']['trial_id'], 'SN001')
            self.assertEqual(row['info']['start_at'], result['epochs_onsets'][i])
            self.assertEqual(row['info']['end_at'], result['epochs_onsets'][i] + 30)
            np.testing.assert_equal(row['eeg'], result['epochs_data'][i] * 2 + 1)

    def test_process_record_rejects_corrupt_metadata_before_yield(self):
        original = dict(epochs_data=np.zeros((2, 1, 3000)),
                        epochs_label=['Sleep stage W', 'Sleep stage R'], epochs_onsets=[0, 60])
        changes = [dict(epochs_label=['Sleep stage W']), dict(epochs_onsets=[0]),
                   dict(epochs_onsets=[30, 0]), dict(epochs_onsets=[0, 29]),
                   dict(epochs_onsets=[0, np.nan]), dict(epochs_onsets=[-30, 0]),
                   dict(epochs_label=['Sleep stage W', 'Lights off@@EEG F4-A1']),
                   dict(before_trial=lambda x: x[:1])]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                next(HMCDataset.process_record(PAIR, **dict(original, **change)))
        legacy = dict(original)
        legacy.pop('epochs_onsets')
        rows = list(HMCDataset.process_record(PAIR, **legacy))
        self.assertEqual([r['info']['start_at'] for r in rows], [0., 30.])

    def test_cached_dataset_samples_labels_and_times_match_reader(self):
        from torcheeg import transforms

        expected = self.read()
        with tempfile.TemporaryDirectory() as root:
            records = Path(root, 'records')
            records.mkdir()
            for name in PAIR:
                (records / name).touch()
            with patch(MODULE + '.read_raw_edf', side_effect=lambda *a, **kw: raw_record()), \
                    patch(MODULE + '.mne.read_annotations', side_effect=lambda *a: annotations()):
                dataset = HMCDataset(root_path=str(records), io_path=str(Path(root, 'cache')),
                                     io_mode='pickle', num_worker=0, verbose=False,
                                     channels=['B', 'A'], l_freq=None, h_freq=None,
                                     start_at=30, end_at=180,
                                     label_transform=transforms.Select('label'))
            self.assertEqual(len(dataset), 4)
            for i in range(len(dataset)):
                signal, label = dataset[i]
                np.testing.assert_equal(signal, expected['epochs_data'][i])
                self.assertEqual(label, expected['epochs_label'][i])
                self.assertEqual(dataset.read_info(i)['start_at'], expected['epochs_onsets'][i])
            reloaded = copy.copy(dataset)
            np.testing.assert_equal(reloaded[0][0], dataset[0][0])


if __name__ == '__main__':
    unittest.main()
