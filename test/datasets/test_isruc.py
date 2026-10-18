import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mne
import numpy as np

from torcheeg.datasets import ISRUCDataset
from torcheeg.datasets.module.sleep_stage_detection.isruc import epoching, format_subject_id, interploate


MODULE = 'torcheeg.datasets.module.sleep_stage_detection.isruc'
A_CHANNELS = ['F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1']
M_CHANNELS = ['F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1']


def make_raw(channels=None, first_samp=0, seconds=95):
    channels = A_CHANNELS if channels is None else channels
    time = np.arange(seconds * 100) / 100.
    values = np.stack([np.sin(time * (i + 1)) + i for i in range(len(channels))]) * 1e-6
    return mne.io.RawArray(values, mne.create_info(channels, 100, 'eeg'),
                           first_samp=first_samp, verbose=False)


class TestISRUCDataset(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        folder = self.root / 'Subgroup_1/1'
        folder.mkdir(parents=True)
        self.recording = folder / '1.rec'
        self.scoring = folder / '1_1.txt'
        self.recording.write_bytes(b'owned synthetic source')
        self.scoring.write_text('0\n1\n5\n')
        self.record = (str(self.recording), str(self.scoring), 0)

    def read(self, raw=None, **kwargs):
        raw = make_raw() if raw is None else raw
        settings = dict(l_freq=None, h_freq=None, root_path=str(self.root))
        settings.update(kwargs)
        temporary_paths = []

        def reader(path, preload):
            self.assertTrue(preload)
            self.assertEqual(Path(path).suffix, '.edf')
            self.assertEqual(Path(path).read_bytes(), self.recording.read_bytes())
            temporary_paths.append(Path(path))
            return raw.copy()

        with patch(MODULE + '.read_raw_edf', side_effect=reader):
            result = ISRUCDataset.read_record(self.record, **settings)
        self.assertTrue(temporary_paths)
        self.assertTrue(all(not path.exists() for path in temporary_paths))
        self.assertEqual(self.recording.read_bytes(), b'owned synthetic source')
        return result

    def test_exact_nonoverlapping_30_second_grid_and_partial_tail(self):
        raw = make_raw()
        result = self.read(raw)
        self.assertEqual(result['epochs_data'].shape, (3, 6, 3000))
        np.testing.assert_equal(result['epochs_onsets'], [0., 30., 60.])
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage N1', 'Sleep stage R'])
        expected = raw.get_data()[:, :9000].reshape(6, 3, 3000).transpose(1, 0, 2)
        np.testing.assert_equal(result['epochs_data'], expected)
        epochs = epoching(make_raw(seconds=120))
        np.testing.assert_equal(np.diff(epochs.events[:, 0]), [3000, 3000, 3000])

    def test_annotation_rejection_keeps_matching_labels_and_original_gaps(self):
        raw = make_raw()
        raw.set_annotations(mne.Annotations([30], [30], ['BAD_test']))
        result = self.read(raw)
        np.testing.assert_equal(result['epochs_onsets'], [0., 60.])
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage R'])
        self.assertEqual(len(result['epochs_data']), 2)

    def test_all_rejected_epochs_raise_instead_of_publishing_an_empty_record(self):
        raw = make_raw()
        raw.set_annotations(mne.Annotations([0], [95], ['BAD_all']))
        with self.assertRaisesRegex(ValueError, 'No complete scored'):
            self.read(raw)

    def test_resampling_and_nonzero_first_sample_keep_grid(self):
        result = self.read(make_raw(first_samp=1000), sfreq=137)
        self.assertEqual(result['epochs_data'].shape, (3, 6, 4110))
        np.testing.assert_equal(result['epochs_onsets'], [0., 30., 60.])
        self.assertEqual(result['epochs_label'], ['Sleep stage W', 'Sleep stage N1', 'Sleep stage R'])

    def test_both_published_reference_name_variants_select_real_signals(self):
        for names in (A_CHANNELS, M_CHANNELS):
            with self.subTest(names=names):
                raw = make_raw(names)
                result = self.read(raw)
                self.assertEqual(result['epochs_channels'], names)
                self.assertEqual(result['derived_channels'], {})
                self.assertEqual(result['zero_filled_channels'], [])
                self.assertTrue(np.any(result['epochs_data'] != 0))
                np.testing.assert_equal(result['epochs_data'][0], raw.get_data()[:, :3000])

    def test_bipolar_channel_is_an_actual_subtraction_and_can_be_disabled(self):
        raw = make_raw(['C3', 'A2', 'LOC'])
        result = self.read(raw, channels=[('C3-A2', 'C3-M2'), 'LOC-A2'], missing_channels='raise')
        self.assertEqual(result['epochs_channels'], ['C3-A2', 'LOC-A2'])
        self.assertEqual(result['derived_channels'], {'C3-A2': ['C3', 'A2'], 'LOC-A2': ['LOC', 'A2']})
        self.assertEqual(result['zero_filled_channels'], [])
        expected = raw.get_data()[[0, 2], :3000] - raw.get_data()[1, :3000]
        np.testing.assert_allclose(result['epochs_data'][0], expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.read(raw, channels=[('C3-A2', 'C3-M2')], derive_channels=False, missing_channels='raise')

    def test_recorded_candidate_is_preferred_to_computing_another_reference(self):
        raw = make_raw(['C3-M2', 'C3', 'A2'])
        result = self.read(raw, channels=[('C3-A2', 'C3-M2')], missing_channels='raise')
        self.assertEqual(result['epochs_channels'], ['C3-M2'])
        self.assertEqual(result['derived_channels'], {})
        np.testing.assert_equal(result['epochs_data'][0, 0], raw.get_data()[0, :3000])

    def test_missing_channels_and_requested_order_are_explicit(self):
        result = self.read(make_raw(['C3-A2']), channels=['missing', 'C3-A2'])
        self.assertEqual(result['epochs_channels'], ['missing', 'C3-A2'])
        self.assertEqual(result['zero_filled_channels'], ['missing'])
        np.testing.assert_equal(result['epochs_data'][:, 0], 0)
        candidates = self.read(make_raw(['C3-A2']), channels=[('F3-A2', 'F3-M2'), 'C3-A2'])
        self.assertEqual(candidates['zero_filled_channels'], ['F3-A2'])
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            self.read(make_raw(['C3-A2']), channels=['missing'], missing_channels='raise')

    def test_ambiguous_duplicate_and_malformed_channel_requests_raise(self):
        for channels in [[('F3-A2', 'F3-M2')], ['F3-A2', 'F3-A2'], [],
                         [('F3-A2', 'F3-A2')], [()], [3]]:
            raw = make_raw(A_CHANNELS + ['F3-M2'])
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                interploate(raw, channels)

    def test_invalid_score_codes_and_counts_never_silently_truncate(self):
        for codes in ('0\n1\n', '0\n1\n5\n0\n', '', '0\n4\n5\n'):
            self.scoring.write_text(codes)
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                self.read()

    def test_subject_identity_shared_between_nights_but_clip_ids_are_distinct(self):
        values = ISRUCDataset.fake_record(None)
        infos = []
        for night in (1, 2):
            record = (str(self.root / f'Subgroup_2/1_2/{night}/{night}.rec'),
                      str(self.root / f'Subgroup_2/1_2/{night}/{night}_1.txt'), 1)
            args = dict(values)
            args.pop('record')
            infos.append(next(ISRUCDataset.process_record(record, root_path=str(self.root), **args))['info'])
        self.assertEqual(infos[0]['subject_id'], infos[1]['subject_id'])
        self.assertEqual([row['session_id'] for row in infos], [1, 2])
        self.assertNotEqual(infos[0]['clip_id'], infos[1]['clip_id'])
        self.assertNotEqual(infos[0]['trial_id'], infos[1]['trial_id'])
        self.assertEqual(infos[0]['subject_id'], 'Subgroup_2_1_2')

    def test_metadata_preserves_gap_channel_provenance_and_hooks(self):
        result = self.read(make_raw(['C3', 'A2']), channels=['C3-A2', 'missing'])
        result['epochs_onsets'] = np.array([0., 60., 90.])
        rows = list(ISRUCDataset.process_record(self.record, root_path=str(self.root),
                    before_trial=lambda x: x * 2, offline_transform=lambda eeg: {'eeg': eeg + 1}, **result))
        self.assertEqual([r['info']['start_at'] for r in rows], [0., 60., 90.])
        self.assertEqual(rows[-1]['info']['end_at'], 120.)
        self.assertEqual(json.loads(rows[0]['info']['channel_names']), ['C3-A2', 'missing'])
        self.assertEqual(json.loads(rows[0]['info']['derived_channels']), {'C3-A2': ['C3', 'A2']})
        self.assertEqual(json.loads(rows[0]['info']['zero_filled_channels']), ['missing'])
        np.testing.assert_equal(rows[1]['eeg'], result['epochs_data'][1] * 2 + 1)
        self.assertEqual(format_subject_id(str(self.recording), str(self.root)), 'Subgroup_1_1')

    def test_process_rejects_bad_shapes_times_labels_and_provenance(self):
        original = ISRUCDataset.fake_record(None)
        original.pop('record')
        changes = [dict(epochs_label=['Sleep stage W']), dict(epochs_onsets=[0, 30]),
                   dict(epochs_onsets=[0, 30, 59, 90]), dict(epochs_onsets=[0, 30, 60, np.nan]),
                   dict(epochs_channels=['C3-A2']), dict(derived_channels={'other': ['C3', 'A2']}),
                   dict(zero_filled_channels=['other']), dict(derived_channels={'F3-A2': ['F3']}),
                   dict(before_trial=lambda x: x[:2])]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                next(ISRUCDataset.process_record(self.record, root_path=str(self.root), **dict(original, **change)))

    def test_all_groups_scorer_selection_and_deterministic_file_discovery(self):
        for group, subject, night in [(0, '2', None), (1, '1_2', '1'), (1, '1_2', '2'), (2, '1', None)]:
            folder = self.root / f'Subgroup_{group + 1}' / subject
            if night is not None:
                folder /= night
            folder.mkdir(parents=True, exist_ok=True)
            stem = night or subject
            for suffix in ('.rec', '_1.txt', '_2.txt'):
                (folder / (stem + suffix)).touch()
        self.recording.with_name('1_2.txt').touch()
        (self.root / 'Subgroup_1/README.txt').touch()
        dataset = SimpleNamespace(groups=[2, 1, 0], scorer=2)
        records = ISRUCDataset.set_records(dataset, str(self.root))
        self.assertEqual(len(records), 5)
        self.assertEqual([r[2] for r in records], [0, 0, 1, 1, 2])
        self.assertTrue(all(r[1].endswith('_2.txt') for r in records))
        self.scoring.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            ISRUCDataset.set_records(SimpleNamespace(groups=[0], scorer=1), str(self.root))

    def test_second_expert_is_read_without_mixing_scores(self):
        second = self.recording.with_name('1_2.txt')
        second.write_text('5\n2\n1\n')
        self.record = (str(self.recording), str(second), 0)
        result = self.read()
        self.assertEqual(result['epochs_label'], ['Sleep stage R', 'Sleep stage N2', 'Sleep stage N1'])
        row = next(ISRUCDataset.process_record(self.record, root_path=str(self.root), **result))
        self.assertEqual(row['info']['scorer_id'], 2)

    def test_cached_two_night_dataset_and_sequence_windows_keep_night_boundaries(self):
        from torcheeg import transforms
        from torcheeg.datasets import SequenceDataset

        for night in (1, 2):
            folder = self.root / f'Subgroup_2/1_2/{night}'
            folder.mkdir(parents=True)
            (folder / f'{night}.rec').write_text(str(night))
            (folder / f'{night}_1.txt').write_text('0\n1\n5\n')

        def reader(path, preload):
            night = int(Path(path).read_text())
            raw = make_raw(A_CHANNELS if night == 1 else M_CHANNELS)
            raw._data *= night
            return raw

        with patch(MODULE + '.read_raw_edf', side_effect=reader):
            dataset = ISRUCDataset(root_path=str(self.root), groups=[1], l_freq=None, h_freq=None,
                                   missing_channels='raise', io_path=str(self.root / 'cache'),
                                   io_mode='pickle', num_worker=0, verbose=False,
                                   label_transform=transforms.Select('label'))
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.info['subject_id'].nunique(), 1)
        self.assertEqual(dataset.info['record_id'].nunique(), 2)
        self.assertEqual(dataset.info['clip_id'].nunique(), 6)
        self.assertEqual(dataset[0][1], 'Sleep stage W')
        np.testing.assert_equal(dataset[3][0], dataset[0][0] * 2)
        self.assertEqual(json.loads(dataset.read_info(0)['channel_names']), A_CHANNELS)
        self.assertEqual(json.loads(dataset.read_info(3)['channel_names']), M_CHANNELS)
        sequences = SequenceDataset(dataset, sequence_length=2)
        self.assertEqual(len(sequences), 2)
        for index in range(2):
            positions = sequences.get_epoch_indices(index)
            self.assertEqual(dataset.info.iloc[list(positions)]['record_id'].nunique(), 1)
            self.assertEqual(list(sequences[index][1]), ['Sleep stage W', 'Sleep stage N1'])

    def test_mismatched_group_session_and_scoring_path_fail_before_io(self):
        for record in [(self.record[0], self.record[1], 1),
                       (self.record[0], str(self.root / 'Subgroup_1/2/2_1.txt'), 0),
                       (self.record[0], self.record[1], True)]:
            with self.subTest(record=record), patch(MODULE + '.read_raw_edf') as reader:
                with self.assertRaises(ValueError):
                    ISRUCDataset.read_record(record, root_path=str(self.root))
                reader.assert_not_called()

    def test_constructor_rejects_invalid_configuration_before_cache_creation(self):
        for kwargs in [dict(groups=[]), dict(groups=[0, 5]), dict(groups=[0, 0]),
                       dict(groups=[True]), dict(scorer=0), dict(scorer=True),
                       dict(missing_channels='invalid'), dict(derive_channels=1)]:
            with self.subTest(kwargs=kwargs), patch(MODULE + '.BaseDataset.__init__') as parent:
                with self.assertRaises(ValueError):
                    ISRUCDataset(root_path='unused', io_path='unused', **kwargs)
                parent.assert_not_called()


if __name__ == '__main__':
    unittest.main()
