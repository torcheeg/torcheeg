from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np
from scipy.io import savemat

from torcheeg import transforms
from torcheeg.datasets import BCICIV2bDataset


def write_record(root, filename='B01T.mat', invalid=None):
    sessions = []
    for index in range(3 if filename[3] == 'T' else 2):
        time = np.arange(6500) / 250.
        values = np.stack([np.sin(2 * np.pi * (4 + i) * time) * (i + 1) + index
                           for i in range(6)], axis=1)
        session = dict(X=values, trial=np.array([1, 2402, 4701]), y=np.array([1, 2, 1]),
                       fs=250, artifacts=np.array([0, 1, 0]), classes=['left hand', 'right hand'])
        if invalid:
            session.update(invalid)
        sessions.append(session)
    savemat(Path(root) / filename, {'data': np.array(sessions, dtype=object)})
    return sessions


class TestBCICIV2bDataset(unittest.TestCase):
    def test_actual_mat_default_waveforms_labels_and_one_based_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_record(directory)
            result = BCICIV2bDataset.read_record('B01T.mat', root_path=directory)
            rows = list(BCICIV2bDataset.process_record(record='B01T.mat', **result))
            self.assertEqual(len(rows), 9)
            np.testing.assert_array_equal(rows[0]['eeg'], source[0]['X'][375:1500, :3].T)
            np.testing.assert_array_equal(rows[1]['eeg'], source[0]['X'][2401 + 375:2401 + 1500, :3].T)
            self.assertEqual([row['info']['label'] for row in rows[:3]], [1, 2, 1])
            self.assertEqual(rows[1]['info']['source_trial_onset'], 2401)
            self.assertEqual(rows[0]['info']['signal_unit'], 'uV')
            self.assertEqual(rows[0]['info']['start_at'], 375)
            self.assertEqual(rows[0]['info']['end_at'], 1500)

    def test_train_and_evaluation_sessions_and_trial_ids_do_not_collide(self):
        with tempfile.TemporaryDirectory() as directory:
            all_rows = []
            for filename, sessions in [('B01T.mat', {1, 2, 3}), ('B01E.mat', {4, 5})]:
                write_record(directory, filename)
                rows = list(BCICIV2bDataset.process_record(record=filename,
                       **BCICIV2bDataset.read_record(filename, root_path=directory)))
                self.assertEqual({row['info']['session_id'] for row in rows}, sessions)
                all_rows.extend(rows)
            self.assertEqual(len({row['info']['trial_id'] for row in all_rows}), 15)
            self.assertEqual(len({row['key'] for row in all_rows}), 15)

    def test_resampling_offsets_and_original_record_time_use_the_output_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_record(directory)
            result = BCICIV2bDataset.read_record('B01T.mat', root_path=directory, sfreq=128)
            rows = list(BCICIV2bDataset.process_record(record='B01T.mat', **result))
            raw = mne.io.RawArray(source[0]['X'][:, :3].T * 1e-6,
                                 mne.create_info(['C3', 'Cz', 'C4'], 250, 'eeg'), verbose=False)
            raw.resample(128, verbose=False)
            values = raw.get_data() * 1e6
            self.assertEqual(rows[0]['eeg'].shape, (3, 576))
            first = round(2401 * 128 / 250)
            np.testing.assert_array_equal(rows[1]['eeg'], values[:, first + 192:first + 768])
            self.assertEqual(rows[1]['info']['source_start_seconds'], (first + 192) / 128)
            self.assertEqual(rows[1]['info']['source_end_seconds'], (first + 768) / 128)

    def test_continuous_filters_reference_and_explicit_imagery_window(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_record(directory)
            options = dict(sfreq=128, notch_freqs=50., l_freq=4., h_freq=38., ref_channels='average',
                           offset_start=750, offset_end=1750)
            result = BCICIV2bDataset.read_record('B01T.mat', root_path=directory, **options)
            rows = list(BCICIV2bDataset.process_record(record='B01T.mat', offset_start=750, offset_end=1750, **result))
            raw = mne.io.RawArray(source[0]['X'][:, :3].T * 1e-6,
                                 mne.create_info(['C3', 'Cz', 'C4'], 250, 'eeg'), verbose=False)
            raw.notch_filter(50., verbose=False).filter(4., 38., verbose=False)
            raw.set_eeg_reference('average', verbose=False)
            raw.resample(128, verbose=False)
            self.assertEqual(rows[0]['eeg'].shape, (3, 512))
            np.testing.assert_array_equal(rows[0]['eeg'], raw.get_data()[:, 384:896] * 1e6)
            np.testing.assert_allclose(rows[0]['eeg'].mean(0), 0, atol=1e-12)

    def test_artifact_policy_and_chunk_overlap_keep_source_trial_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            write_record(directory)
            values = BCICIV2bDataset.read_record('B01T.mat', root_path=directory)
            rows = list(BCICIV2bDataset.process_record(record='B01T.mat', chunk_size=500,
                        overlap=250, skip_trial_with_artifacts=True, **values))
            self.assertEqual(len(rows), 18)
            self.assertTrue(all(row['info']['artifact'] == 0 for row in rows))
            self.assertEqual([row['info']['start_at'] for row in rows[:3]], [375, 625, 875])
            self.assertEqual(len({row['info']['trial_id'] for row in rows}), 6)

    def test_explicit_eog_channels_keep_native_values_and_reference_type(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_record(directory)
            result = BCICIV2bDataset.read_record('B01T.mat', root_path=directory, num_channel=6)
            np.testing.assert_array_equal(result['samples'][0], source[0]['X'][:1750].T)
            result = BCICIV2bDataset.read_record('B01T.mat', root_path=directory, num_channel=6, ref_channels='average')
            np.testing.assert_allclose(result['samples'][0][:3].mean(0), 0, atol=1e-12)
            np.testing.assert_allclose(result['samples'][0][3:], source[0]['X'][:1750, 3:].T, rtol=1e-14, atol=1e-12)

    def test_eeg_layout_constants_match_the_first_three_channels(self):
        from torcheeg.datasets.constants import (BCICIV2B_CHANNEL_LIST, BCICIV2B_CHANNEL_LOCATION_DICT,
                                                 BCICIV2B_ADJACENCY_MATRIX, BCICIV2B_STANDARD_ADJACENCY_MATRIX)
        self.assertEqual(BCICIV2B_CHANNEL_LIST, ['C3', 'CZ', 'C4'])
        self.assertEqual(set(BCICIV2B_CHANNEL_LOCATION_DICT), set(BCICIV2B_CHANNEL_LIST))
        self.assertEqual(len({tuple(x) for x in BCICIV2B_CHANNEL_LOCATION_DICT.values()}), 3)
        for values in [BCICIV2B_ADJACENCY_MATRIX, BCICIV2B_STANDARD_ADJACENCY_MATRIX]:
            self.assertEqual(np.shape(values), (3, 3))
            self.assertTrue(np.isfinite(values).all())

    def test_invalid_settings_fail_before_io(self):
        cases = [dict(overlap=1125), dict(overlap=-1), dict(sfreq=0), dict(sfreq=128.),
                 dict(num_channel=7), dict(chunk_size=2000), dict(offset_start=-1),
                 dict(offset_end=1800), dict(l_freq=40, h_freq=4), dict(h_freq=125),
                 dict(notch_freqs=np.nan), dict(ref_channels='C3'), dict(skip_trial_with_artifacts='yes')]
        for options in cases:
            with self.subTest(options=options), patch('torcheeg.datasets.module.motor_imagery.bciciv_2b.BaseDataset.__init__') as parent:
                with self.assertRaises(ValueError):
                    BCICIV2bDataset(root_path='unused', io_path='unused', **options)
                parent.assert_not_called()

    def test_invalid_sessions_and_metadata_are_not_silently_skipped(self):
        cases = [dict(trial=[0, 2402, 4701]), dict(trial=[1, 1001, 4701]), dict(fs=200),
                 dict(y=[0, 1, 2]), dict(y=[1, 2]), dict(artifacts=[0, 2, 0]),
                 dict(X=np.full((6500, 6), np.nan)), dict(trial=[1., 2402., 4701.])]
        with tempfile.TemporaryDirectory() as directory:
            for options in cases:
                write_record(directory, invalid=options)
                with self.subTest(options=list(options)), self.assertRaises(ValueError):
                    BCICIV2bDataset.read_record('B01T.mat', root_path=directory)
            with self.assertRaises(ValueError):
                BCICIV2bDataset.read_record(dict(file_name='B01T.mat', subject_id=2), root_path=directory)

    def test_cache_and_online_label_transform_preserve_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            write_record(directory)
            write_record(directory, 'B01E.mat')
            dataset = BCICIV2bDataset(root_path=directory, io_path=str(Path(directory) / 'cache'),
                io_mode='pickle', verbose=False, online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose([transforms.Select('label'), transforms.Lambda(lambda x: x - 1)]))
            self.assertEqual(len(dataset), 15)
            self.assertEqual(set(dataset.info.session_id), {1, 2, 3, 4, 5})
            self.assertEqual(set(dataset.info.phase), {'train', 'test'})
            self.assertEqual(tuple(dataset[0][0].shape), (3, 1125))
            self.assertIn(dataset[0][1], [0, 1])

    def test_discovery_and_original_fake_record(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'B01T.mat').touch()
            (Path(directory) / 'README.mat').touch()
            self.assertEqual(BCICIV2bDataset.set_records(None, directory),
                [dict(file_name='B01T.mat', subject_id=1, phase='train')])
            (Path(directory) / 'B10T.mat').touch()
            with self.assertRaises(ValueError):
                BCICIV2bDataset.set_records(None, directory)
        row = next(BCICIV2bDataset.process_record(record='fake_record', **BCICIV2bDataset.fake_record()))
        self.assertEqual(row['eeg'].shape, (3, 1125))

    def test_hooks_do_not_modify_source_or_other_overlapping_windows(self):
        values = BCICIV2bDataset.fake_record()
        originals = [trial.copy() for trial in values['samples']]
        def change(eeg):
            eeg += 4
            return dict(eeg=eeg)
        rows = list(BCICIV2bDataset.process_record(record='fake_record', chunk_size=500,
                    overlap=250, offline_transform=change, **values))
        np.testing.assert_array_equal(values['samples'][0], originals[0])
        np.testing.assert_array_equal(rows[1]['eeg'], originals[0][:, 625:1125] + 4)
        with self.assertRaises(ValueError):
            list(BCICIV2bDataset.process_record(record='fake_record', before_trial=lambda x: x[:, :-1], **values))


if __name__ == '__main__':
    unittest.main()
