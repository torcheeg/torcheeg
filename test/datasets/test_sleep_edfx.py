import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np

from torcheeg.datasets import SleepEDFxDataset
from torcheeg.datasets.module.sleep_stage_detection.sleep_edfx import (
    interploate, list_records)


class TestSleepEDFxDataset(unittest.TestCase):
    record = ('SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf')

    def touch(self, root, *names):
        for name in names:
            Path(root, name).touch()

    def test_pairs_use_subject_and_night_not_directory_or_position(self):
        with tempfile.TemporaryDirectory(prefix='sleep-edf-') as root:
            self.touch(root, *self.record, 'SC4002E0-PSG.edf',
                       'SC4002EH-Hypnogram.edf', 'ST7021J0-PSG.edf',
                       'ST7021JM-Hypnogram.edf', 'README.txt')
            pairs = list_records(root)
            self.assertEqual(len(pairs), 3)
            for recording, scoring in pairs:
                self.assertEqual(Path(recording).name[:6], Path(scoring).name[:6])
                self.assertEqual(Path(recording).parent, Path(root))
            self.assertEqual(pairs, sorted(pairs))

    def test_missing_pair_does_not_shift_other_nights(self):
        for files in [('SC4001E0-PSG.edf', 'SC4011EH-Hypnogram.edf'),
                      (self.record[0],), (self.record[1],)]:
            with self.subTest(files=files), tempfile.TemporaryDirectory() as root:
                self.touch(root, *files)
                with self.assertRaisesRegex(ValueError, 'Unpaired'):
                    list_records(root)

    def test_duplicate_or_malformed_pair_is_rejected(self):
        cases = [(*self.record, 'SC4001EH-Hypnogram.edf'),
                 (*self.record, 'SC4001E1-PSG.edf'),
                 ('invalid-PSG.edf', 'invalid-Hypnogram.edf')]
        for files in cases:
            with self.subTest(files=files), tempfile.TemporaryDirectory() as root:
                self.touch(root, *files)
                with self.assertRaises(ValueError):
                    list_records(root)

    def test_subject_grouping_nights_studies_and_safe_ids(self):
        records = [self.record, ('SC4002E0-PSG.edf', 'SC4002EC-Hypnogram.edf'),
                   ('SC4011E0-PSG.edf', 'SC4011EH-Hypnogram.edf'),
                   ('ST7001J0-PSG.edf', 'ST7001JP-Hypnogram.edf')]
        infos = []
        for record in records:
            kwargs = SleepEDFxDataset.fake_record(record)
            kwargs['record'] = tuple(os.path.join('/tmp/sleep-edf-fixture', name)
                                     for name in record)
            infos.append(next(SleepEDFxDataset.process_record(**kwargs))['info'])
        self.assertEqual([info['subject_id'] for info in infos],
                         ['SC400', 'SC400', 'SC401', 'ST700'])
        self.assertEqual([info['session_id'] for info in infos], [1, 2, 1, 1])
        self.assertEqual(len({info['clip_id'] for info in infos}), 4)
        for info in infos:
            self.assertNotIn('/', info['clip_id'])
            self.assertEqual(info['record_id'], info['trial_id'])
            self.assertEqual(len(info['record_id']), 8)

    def test_labels_onsets_and_transforms_preserve_gaps(self):
        data = np.arange(6 * 2 * 3000).reshape(6, 2, 3000)
        labels = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
                  'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R']
        onsets = np.array([0, 30, 90, 120, 180, 210])
        items = list(SleepEDFxDataset.process_record(
            self.record, data, labels, epochs_onsets=onsets,
            before_trial=lambda eeg: eeg + 1,
            offline_transform=lambda eeg: {'eeg': eeg * 2}))
        self.assertEqual([item['info']['label'] for item in items],
                         ['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2',
                          'Sleep stage N3', 'Sleep stage N3', 'Sleep stage R'])
        np.testing.assert_array_equal([item['info']['start_at'] for item in items], onsets)
        np.testing.assert_array_equal([item['info']['end_at'] for item in items], onsets + 30)
        np.testing.assert_array_equal(items[2]['eeg'], (data[2] + 1) * 2)
        self.assertEqual(items[2]['key'], 'SC4001E0_2')

    def test_legacy_process_inputs_and_normalized_labels(self):
        kwargs = SleepEDFxDataset.fake_record(self.record)
        kwargs.pop('epochs_onsets')
        kwargs['epochs_label'] = ['Sleep stage W', 'Sleep stage N1',
                                  'Sleep stage N2', 'Sleep stage N3']
        items = list(SleepEDFxDataset.process_record(**kwargs))
        self.assertEqual([item['info']['start_at'] for item in items], [0, 30, 60, 90])
        self.assertEqual([item['info']['label'] for item in items], kwargs['epochs_label'])

    def test_invalid_epoch_metadata_and_pairs_fail_before_yield(self):
        kwargs = SleepEDFxDataset.fake_record(self.record)
        overrides = [dict(epochs_label=['Sleep stage W']),
                     dict(epochs_label=['Sleep stage ?'] * 4),
                     dict(epochs_onsets=[0, 30]), dict(epochs_onsets=[0, 30, 60, np.nan]),
                     dict(epochs_onsets=[0, 30, 60, np.inf]),
                     dict(epochs_onsets=[-30, 0, 30, 60]),
                     dict(epochs_onsets=[0, 30, 60, 60]),
                     dict(epochs_onsets=[0, 30, 60, 89]),
                     dict(before_trial=lambda eeg: eeg[:-1]),
                     dict(record=('SC4001E0-PSG.edf', 'SC4011EC-Hypnogram.edf'))]
        for override in overrides:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    next(SleepEDFxDataset.process_record(**dict(kwargs, **override)))

    def make_raw(self, first_samp=0):
        times = np.arange(19500) / 100.
        data = np.stack([1e-5 * np.sin(2 * np.pi * 2 * times),
                         2e-5 * np.cos(2 * np.pi * 3 * times)])
        return mne.io.RawArray(data, mne.create_info(
            ['EEG Fpz-Cz', 'EOG horizontal'], 100, ['eeg', 'eog']),
            first_samp=first_samp, verbose=False)

    def read_synthetic(self, first_samp=0, sfreq=100):
        annotations = mne.Annotations([0, 30, 60, 90, 120, 150, 180],
                                     [30] * 7,
                                     ['Sleep stage W', 'Movement time', 'Sleep stage 1',
                                      'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R',
                                      'Sleep stage 2'])
        module = 'torcheeg.datasets.module.sleep_stage_detection.sleep_edfx.'
        with patch(module + 'read_raw_edf', return_value=self.make_raw(first_samp)), \
                patch(module + 'mne.read_annotations', return_value=annotations):
            return SleepEDFxDataset.read_record(
                self.record, channels=['EOG horizontal', 'EEG Fpz-Cz'],
                sfreq=sfreq, missing_channels='raise')

    def test_real_mne_epoch_selection_labels_timestamps_and_channel_order(self):
        for first_samp, sfreq in [(0, 100), (700, 50)]:
            with self.subTest(first_samp=first_samp, sfreq=sfreq):
                result = self.read_synthetic(first_samp, sfreq)
                self.assertEqual(result['epochs_data'].shape, (5, 2, 30 * sfreq))
                self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage 1',
                                                          'Sleep stage 3', 'Sleep stage 4',
                                                          'Sleep stage R'])
                np.testing.assert_allclose(result['epochs_onsets'], [0, 60, 90, 120, 150])
                # The first selected channel is the EOG cosine, not the EEG sine.
                expected = self.make_raw(first_samp).pick_channels(
                    ['EOG horizontal', 'EEG Fpz-Cz'], ordered=True)
                expected.filter(.5, 30., fir_design='firwin').resample(sfreq)
                np.testing.assert_allclose(result['epochs_data'][0],
                                           expected.get_data()[:, :30 * sfreq])
                items = list(SleepEDFxDataset.process_record(self.record, **result))
                self.assertEqual(items[1]['info']['start_at'], 60)
                self.assertEqual(items[2]['info']['label'], items[3]['info']['label'])

    def test_missing_channel_policy_and_order(self):
        raw = self.make_raw()
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            interploate(raw.copy(), ['EEG Fpz-Cz', 'EEG Pz-Oz'], 'raise')
        result = interploate(raw.copy(), ['EOG horizontal', 'EEG Pz-Oz', 'EEG Fpz-Cz'])
        self.assertEqual(result.ch_names, ['EOG horizontal', 'EEG Pz-Oz', 'EEG Fpz-Cz'])
        np.testing.assert_array_equal(result.get_data()[1], 0)
        np.testing.assert_array_equal(result.get_data()[[0, 2]], raw.get_data()[[1, 0]])
        for channels, policy in [([], 'zero'), (['EEG Fpz-Cz'] * 2, 'zero'),
                                 (['EEG Fpz-Cz'], 'interpolate')]:
            with self.assertRaises(ValueError):
                interploate(raw.copy(), channels, policy)

    def test_scored_events_not_overlapping_annotation_descriptions(self):
        annotations = mne.Annotations([0, 0], [60, 30], ['Lights off', 'Sleep stage 2'])
        module = 'torcheeg.datasets.module.sleep_stage_detection.sleep_edfx.'
        with patch(module + 'read_raw_edf', return_value=self.make_raw()), \
                patch(module + 'mne.read_annotations', return_value=annotations):
            result = SleepEDFxDataset.read_record(self.record, missing_channels='zero')
        self.assertEqual(result['epochs_label'], ['Sleep stage 2'])
        np.testing.assert_array_equal(result['epochs_onsets'], [0])

    def test_long_annotations_expand_only_into_complete_epochs(self):
        annotations = mne.Annotations([0, 90], [90, 105],
                                     ['Sleep stage 2', 'Sleep stage ?'])
        module = 'torcheeg.datasets.module.sleep_stage_detection.sleep_edfx.'
        with patch(module + 'read_raw_edf', return_value=self.make_raw()), \
                patch(module + 'mne.read_annotations', return_value=annotations):
            result = SleepEDFxDataset.read_record(
                self.record, channels=['EEG Fpz-Cz', 'EOG horizontal'])
        self.assertEqual(result['epochs_label'], ['Sleep stage 2'] * 3)
        np.testing.assert_array_equal(result['epochs_onsets'], [0, 30, 60])
        self.assertEqual(result['epochs_data'].shape, (3, 2, 3000))

    def test_absent_or_incomplete_scored_intervals_raise(self):
        module = 'torcheeg.datasets.module.sleep_stage_detection.sleep_edfx.'
        for description, duration in [('Sleep stage ?', 60), ('Sleep stage W', 15)]:
            with self.subTest(description=description, duration=duration):
                annotations = mne.Annotations([0], [duration], [description])
                with patch(module + 'read_raw_edf', return_value=self.make_raw()), \
                        patch(module + 'mne.read_annotations', return_value=annotations):
                    with self.assertRaisesRegex(ValueError, 'No complete scored'):
                        SleepEDFxDataset.read_record(self.record)

    def test_cache_round_trip_preserves_metadata_and_data(self):
        result = self.read_synthetic()
        module = 'torcheeg.datasets.module.sleep_stage_detection.sleep_edfx.'
        with tempfile.TemporaryDirectory(prefix='sleep-edf-') as root:
            source = Path(root, 'source', 'sleep-cassette')
            source.mkdir(parents=True)
            self.touch(source, *self.record)
            with patch(module + 'SleepEDFxDataset.read_record', return_value=result):
                dataset = SleepEDFxDataset(root_path=str(source.parent), studies=['cassette'],
                                           missing_channels='raise', io_path=str(Path(root, 'cache')),
                                           io_mode='pickle', verbose=False)
            self.assertEqual(len(dataset), 5)
            signal, info = dataset[1]
            self.assertEqual(info['subject_id'], 'SC400')
            self.assertEqual(info['start_at'], 60)
            np.testing.assert_array_equal(signal, result['epochs_data'][1])
            with patch(module + 'SleepEDFxDataset.read_record', return_value=result):
                dataset = SleepEDFxDataset(root_path=str(source.parent), studies=['cassette'],
                                           missing_channels='raise', io_path=str(Path(root, 'cache')),
                                           io_mode='pickle', verbose=False)
            np.testing.assert_array_equal(dataset[1][0], signal)
            self.assertEqual(dataset[1][1], info)
            with patch(module + 'SleepEDFxDataset.read_record', return_value=result):
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    SleepEDFxDataset(root_path=str(source.parent), studies=['cassette'],
                                      missing_channels='zero', io_path=str(Path(root, 'cache')),
                                      io_mode='pickle', verbose=False)

    def test_studies_and_channel_policy_validation(self):
        for studies in [[], ['cassette', 'other'], ['cassette', 'cassette']]:
            with self.assertRaises(ValueError):
                SleepEDFxDataset(studies=studies)
        with self.assertRaises(ValueError):
            SleepEDFxDataset(missing_channels='interpolate')


if __name__ == '__main__':
    unittest.main()
