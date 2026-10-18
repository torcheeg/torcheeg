import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mne
import numpy as np
from scipy.io import savemat

from torcheeg import transforms
from torcheeg.datasets import BCICIV2aDataset


def write_record(root, filename='A01T.mat', invalid=None, calibration=3):
    runs = []
    for index in range(calibration + 2):
        time = np.arange(6500) / 250.
        values = np.stack([np.sin(2 * np.pi * (4 + i) * time) * (i + 1) + index
                           for i in range(25)], axis=1)
        run = dict(X=values, trial=np.array([1, 2402, 4701]), y=np.array([1, 4, 2]),
                   fs=250, artifacts=np.array([0, 1, 0]))
        if index < calibration:
            run.update(trial=np.array([], dtype=int), y=np.array([], dtype=int),
                       artifacts=np.array([], dtype=int))
        elif invalid:
            run.update(invalid)
        runs.append(run)
    savemat(Path(root) / filename, {'data': np.array(runs, dtype=object)})
    return runs


def rows_from(path, **options):
    read_options = {key: options[key] for key in
                    ('sfreq', 'num_channel', 'notch_freqs', 'l_freq', 'h_freq', 'ref_channels') if key in options}
    data = BCICIV2aDataset.read_record(str(path), **read_options)
    return list(BCICIV2aDataset.process_record(str(path), **data, **options))


class TestBCICIV2aDataset(unittest.TestCase):
    def test_default_mat_values_labels_and_one_based_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_record(directory)
            rows = rows_from(Path(directory) / 'A01T.mat')
            self.assertEqual(len(rows), 6)
            np.testing.assert_array_equal(rows[0]['eeg'], source[3]['X'][:1750, :22].T)
            np.testing.assert_array_equal(rows[1]['eeg'], source[3]['X'][2401:4151, :22].T)
            self.assertEqual([row['info']['label'] for row in rows[:3]], [1, 4, 2])
            self.assertEqual(rows[1]['info']['source_trial_onset'], 2401)
            self.assertEqual(rows[1]['info']['start_at'], 2401)
            self.assertEqual(rows[1]['info']['signal_unit'], 'uV')
            self.assertEqual(rows[0]['info']['subject_id'], 'A01')
            self.assertEqual(rows[0]['info']['session'], 'T')

    def test_constructor_preserves_all_existing_positional_arguments(self):
        parameters = list(inspect.signature(BCICIV2aDataset.__init__).parameters)
        expected = ['self', 'root_path', 'offset', 'chunk_size', 'overlap', 'num_channel',
                    'skip_trial_with_artifacts', 'online_transform', 'offline_transform',
                    'label_transform', 'before_trial', 'after_trial', 'after_session',
                    'after_subject', 'io_path', 'io_size', 'io_mode', 'num_worker', 'verbose']
        self.assertEqual(parameters[:len(expected)], expected)

    def test_offset_alias_and_chunks_stop_before_intertrial_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_record(directory)
            path = Path(directory) / 'A01T.mat'
            legacy = rows_from(path, offset=500, chunk_size=500, overlap=250)
            explicit = rows_from(path, offset_start=500, chunk_size=500, overlap=250)
            self.assertEqual(len(legacy), 24)
            np.testing.assert_array_equal(legacy[0]['eeg'], source[3]['X'][500:1000, :22].T)
            for left, right in zip(legacy, explicit):
                np.testing.assert_array_equal(left['eeg'], right['eeg'])
                self.assertLessEqual(left['info']['end_at'] - left['info']['source_trial_onset'], 1750)
            whole = rows_from(path, offset=500, chunk_size=-1)
            self.assertEqual(whole[0]['eeg'].shape, (22, 1250))

    def test_resampled_continuous_filters_and_source_offset_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_record(directory)
            options = dict(sfreq=128, notch_freqs=50., l_freq=4., h_freq=38.,
                           ref_channels='average', offset_start=500, offset_end=1500, chunk_size=-1)
            rows = rows_from(Path(directory) / 'A01T.mat', **options)
            raw = mne.io.RawArray(source[3]['X'][:, :22].T * 1e-6,
                                 mne.create_info([str(i) for i in range(22)], 250, 'eeg'), verbose=False)
            raw.notch_filter(50., verbose=False).filter(4., 38., verbose=False)
            raw.set_eeg_reference('average', verbose=False)
            raw.resample(128, verbose=False)
            first = round(2401 * 128 / 250)
            self.assertEqual(rows[1]['eeg'].shape, (22, 512))
            np.testing.assert_array_equal(rows[1]['eeg'], raw.get_data()[:, first + 256:first + 768] * 1e6)
            self.assertEqual(rows[1]['info']['source_start_seconds'], (first + 256) / 128)
            np.testing.assert_allclose(rows[1]['eeg'].mean(0), 0, atol=1e-12)

    def test_artifact_policy_keeps_original_trial_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            write_record(directory)
            rows = rows_from(Path(directory) / 'A01T.mat', skip_trial_with_artifacts=True)
            self.assertEqual(len(rows), 4)
            self.assertEqual([row['info']['trial_index'] for row in rows], [0, 2, 0, 2])
            self.assertTrue(all(row['info']['artifact'] == 0 for row in rows))

    def test_eog_channels_keep_values_and_do_not_enter_average_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_record(directory)
            rows = rows_from(Path(directory) / 'A01T.mat', num_channel=25, ref_channels='average')
            np.testing.assert_allclose(rows[0]['eeg'][22:], source[3]['X'][:1750, 22:].T, atol=1e-12)
            np.testing.assert_allclose(rows[0]['eeg'][:22].mean(0), 0, atol=1e-12)

    def test_a04t_short_calibration_and_run_trial_phase_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            all_rows = []
            for filename in ['A04T.mat', 'A04E.mat']:
                write_record(directory, filename, calibration=1 if filename[3] == 'T' else 3)
                rows = rows_from(Path(directory) / filename)
                self.assertEqual({row['info']['run'] for row in rows}, {1, 2} if filename[3] == 'T' else {3, 4})
                all_rows.extend(rows)
            self.assertEqual(len({row['info']['trial_id'] for row in all_rows}), 12)
            self.assertEqual(len({row['key'] for row in all_rows}), 12)

    def test_continuous_hook_and_overlapping_clips_do_not_mutate_sources(self):
        values = BCICIV2aDataset.fake_record()
        original = values['a_data'][0]['X'].copy()
        calls = []
        def before(eeg):
            calls.append(eeg.shape)
            eeg += 2
            return eeg
        def offline(eeg):
            eeg += 3
            return dict(eeg=eeg)
        rows = list(BCICIV2aDataset.process_record(**values, chunk_size=1000, overlap=500,
                    before_trial=before, offline_transform=offline))
        self.assertEqual(calls, [original.shape])
        np.testing.assert_array_equal(values['a_data'][0]['X'], original)
        np.testing.assert_array_equal(rows[1]['eeg'], original[:22, 500:1500] + 2 + 3)
        with self.assertRaises(ValueError):
            list(BCICIV2aDataset.process_record(**values, before_trial=lambda x: x[:, :-1]))

    def test_constructor_hook_retains_full_25_channel_run_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_record(directory)
            shapes = []
            def hook(eeg):
                shapes.append(eeg.shape)
                eeg[0] -= eeg[24]
                return eeg
            dataset = BCICIV2aDataset(root_path=directory, num_channel=3,
                before_trial=hook, io_mode='memory', verbose=False)
            self.assertEqual(shapes, [(25, 6500), (25, 6500)])
            expected = source[3]['X'][:1750, :3].T.copy()
            expected[0] -= source[3]['X'][:1750, 24]
            np.testing.assert_array_equal(dataset[0][0], expected)

    def test_invalid_settings_fail_before_cache_io(self):
        cases = [dict(overlap=1750), dict(overlap=-1), dict(sfreq=0), dict(sfreq=True),
                 dict(num_channel=26), dict(chunk_size=2000), dict(offset=-1),
                 dict(offset=1, offset_start=2), dict(offset_end=1800), dict(l_freq=40, h_freq=4),
                 dict(h_freq=125), dict(notch_freqs=np.nan), dict(ref_channels='C3'),
                 dict(skip_trial_with_artifacts='yes'), dict(sfreq=128), dict(offset=500)]
        for options in cases:
            with self.subTest(options=options), patch('torcheeg.datasets.module.motor_imagery.bciciv_2a.BaseDataset.__init__') as parent:
                with self.assertRaises(ValueError):
                    BCICIV2aDataset(root_path='unused', io_path='unused', **options)
                parent.assert_not_called()

    def test_invalid_mat_runs_raise_instead_of_dropping_trials(self):
        cases = [dict(trial=[0, 2402, 4701]), dict(trial=[1, 1001, 4701]),
                 dict(trial=[1, 2402, 6000]), dict(trial=[1., 2402.5, 4701.]), dict(fs=200),
                 dict(y=[0, 1, 2]), dict(y=[1, 2]), dict(artifacts=[0, 2, 0]),
                 dict(X=np.full((6500, 25), np.nan)), dict(X=np.zeros((6500, 24)))]
        with tempfile.TemporaryDirectory() as directory:
            for options in cases:
                write_record(directory, invalid=options)
                with self.subTest(options=list(options)), self.assertRaises(ValueError):
                    BCICIV2aDataset.read_record(str(Path(directory) / 'A01T.mat'))

    def test_cache_reuse_transforms_and_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            write_record(directory)
            write_record(directory, 'A01E.mat')
            options = dict(root_path=directory, io_path=str(Path(directory) / 'cache'),
                io_mode='pickle', verbose=False, online_transform=transforms.ToTensor(),
                label_transform=transforms.Compose([transforms.Select('label'), transforms.Lambda(lambda x: x - 1)]))
            dataset = BCICIV2aDataset(**options)
            self.assertEqual(len(dataset), 12)
            self.assertEqual(set(dataset.info.phase), {'train', 'test'})
            self.assertEqual(tuple(dataset[0][0].shape), (22, 1750))
            self.assertIn(dataset[0][1], [0, 1, 2, 3])
            reused = BCICIV2aDataset(**options)
            np.testing.assert_array_equal(dataset[0][0], reused[0][0])
            with self.assertRaises(ValueError):
                BCICIV2aDataset(**options, skip_trial_with_artifacts=True)

    def test_discovery_is_sorted_and_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ['A02E.mat', 'A01T.mat', 'README.mat', 'notes.txt']:
                (Path(directory) / name).touch()
            self.assertEqual([Path(x).name for x in BCICIV2aDataset.set_records(None, directory)],
                             ['A01T.mat', 'A02E.mat'])
            (Path(directory) / 'A10T.mat').touch()
            with self.assertRaises(ValueError):
                BCICIV2aDataset.set_records(None, directory)


if __name__ == '__main__':
    unittest.main()
