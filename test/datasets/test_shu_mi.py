import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal
from scipy.io import savemat

from torcheeg import transforms
from torcheeg.datasets import SHUMIDataset
from torcheeg.datasets.constants import (SHU_MI_CHANNEL_LIST, SHU_MI_CHANNEL_LOCATION_DICT,
                                        SHU_MI_ADJACENCY_MATRIX, SHU_MI_STANDARD_ADJACENCY_MATRIX)
from torcheeg.model_selection import LeaveOneSubjectOut


NAME = 'sub-001_ses-01_task_motorimagery_eeg.mat'


def trials(count=3):
    time = np.arange(1000) / 250.
    values = np.stack([10 * np.sin(2 * np.pi * (index % 15 + 1) * time) + index for index in range(32)])
    return np.stack([values + index for index in range(count)]).astype(np.float32)


class TestSHUMIDataset(unittest.TestCase):
    def test_actual_mat_read_resampling_units_and_label_orientations(self):
        with tempfile.TemporaryDirectory() as directory:
            data = trials()
            for labels in [np.array([1, 2, 1]), np.array([[1, 2, 1]]), np.array([[1], [2], [1]])]:
                savemat(Path(directory) / NAME, dict(data=data, labels=labels))
                for rate in [200, 250, 128]:
                    actual = SHUMIDataset.read_record(NAME, root_path=directory, sfreq=rate)
                    reference = data if rate == 250 else signal.resample(data, rate * 4, axis=2)
                    np.testing.assert_array_equal(actual['data'], reference)
                    np.testing.assert_array_equal(actual['labels'], [1, 2, 1])
                    # No hidden division by 100 or conversion to volts.
                    self.assertGreater(abs(actual['data'][0, 31]).max(), 20)

    def test_chunk_values_and_sample_offsets_remain_inside_trial(self):
        data = trials()[:, :, :800]
        rows = list(SHUMIDataset.process_record(NAME, data, np.array([1, 2, 1]), overlap=100, num_channel=3))
        self.assertEqual(len(rows), 21)
        self.assertEqual([row['info']['start_at'] for row in rows[:7]], list(range(0, 601, 100)))
        for index, row in enumerate(rows):
            trial, chunk = divmod(index, 7)
            np.testing.assert_array_equal(row['eeg'], data[trial, :3, chunk * 100:chunk * 100 + 200])
            self.assertEqual(row['info']['trial_index'], trial)
            self.assertEqual(row['info']['label'], [0, 1, 0][trial])
            self.assertEqual(row['info']['end_at'] - row['info']['start_at'], 200)
            self.assertEqual(json.loads(row['info']['channel_names']), SHU_MI_CHANNEL_LIST[:3])
            self.assertEqual(row['info']['signal_unit'], 'uV')
        self.assertEqual(len({row['key'] for row in rows}), 21)

    def test_full_trial_mode_and_incomplete_tail(self):
        data = trials()[:, :, :800]
        for size in [0, -1]:
            rows = list(SHUMIDataset.process_record(NAME, data, np.array([1, 2, 1]), chunk_size=size))
            self.assertEqual(len(rows), 3)
            np.testing.assert_array_equal(rows[2]['eeg'], data[2])
        rows = list(SHUMIDataset.process_record(NAME, data, np.array([1, 2, 1]), chunk_size=300))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[-1]['info']['end_at'], 600)

    def test_trial_identifiers_are_unique_across_sessions_and_subjects(self):
        identities = []
        for name in [NAME, NAME.replace('ses-01', 'ses-02'), NAME.replace('sub-001', 'sub-002')]:
            row = next(SHUMIDataset.process_record(name, trials(1)[:, :, :800], np.array([1])))
            identities.append(row['info']['trial_id'])
            self.assertEqual(row['info']['record_id'], name[:-4])
        self.assertEqual(len(set(identities)), 3)

    def test_hooks_do_not_mutate_source_and_preserve_trial_geometry(self):
        data = trials(1)[:, :, :800]
        original = data.copy()
        def before(eeg):
            eeg += 5
            return eeg
        def offline(eeg):
            eeg *= 2
            return dict(eeg=eeg)
        rows = list(SHUMIDataset.process_record(NAME, data, np.array([2]), overlap=100,
                                              before_trial=before, offline_transform=offline))
        np.testing.assert_array_equal(data, original)
        np.testing.assert_array_equal(rows[1]['eeg'], (original[0, :, 100:300] + 5) * 2)
        for hook in [lambda x: x[:, :-1], lambda x: x * np.nan, lambda x: x[:1]]:
            with self.assertRaisesRegex(ValueError, 'before_trial'):
                list(SHUMIDataset.process_record(NAME, data, np.array([1]), before_trial=hook))

    def test_invalid_settings_fail_before_cache_or_source_io(self):
        for options in [dict(overlap=200), dict(overlap=-1), dict(chunk_size=900), dict(sfreq=0),
                        dict(sfreq=200.), dict(chunk_size=True), dict(num_channel=0), dict(num_channel=33),
                        dict(chunk_size=-1, overlap=800)]:
            with self.subTest(options=options), patch('torcheeg.datasets.module.motor_imagery.shu_mi.BaseDataset.__init__') as parent:
                with self.assertRaises(ValueError):
                    SHUMIDataset(root_path='unused', io_path='unused', **options)
                parent.assert_not_called()

    def test_invalid_mat_arrays_labels_and_filenames_raise(self):
        good = trials()
        cases = [dict(data=good[:, :, :-1], labels=[1, 2, 1]),
                 dict(data=good[:, :2], labels=[1, 2, 1]),
                 dict(data=good * np.nan, labels=[1, 2, 1]),
                 dict(data=good.astype(complex), labels=[1, 2, 1]),
                 dict(data=good, labels=[0, 1, 2]), dict(data=good, labels=[1, 2]),
                 dict(data=good, labels=[1, 1.2, 2]), dict(data=good, labels=np.ones((3, 2))),
                 dict(data=good)]
        with tempfile.TemporaryDirectory() as directory:
            for values in cases:
                savemat(Path(directory) / NAME, values)
                with self.subTest(keys=list(values)), self.assertRaises(ValueError):
                    SHUMIDataset.read_record(NAME, root_path=directory)
        with self.assertRaisesRegex(ValueError, 'Expected sub-'):
            SHUMIDataset.read_record('unrelated.mat', root_path='unused')

    def test_record_discovery_ignores_other_files_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in [NAME, NAME.replace('ses-01', 'ses-03'), 'notes.mat', 'data.edf', '.hidden.mat']:
                (root / name).touch()
            self.assertEqual(SHUMIDataset.set_records(None, directory), [NAME, NAME.replace('ses-01', 'ses-03')])
            (root / 'sub-wrong_eeg.mat').touch()
            with self.assertRaises(ValueError):
                SHUMIDataset.set_records(None, directory)
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'No SHU'):
            SHUMIDataset.set_records(None, directory)

    def test_cached_dataset_transforms_and_subject_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject in [1, 2]:
                for session in [1, 2]:
                    name = f'sub-{subject:03d}_ses-{session:02d}_task_motorimagery_eeg.mat'
                    savemat(root / name, dict(data=trials(2), labels=[[1, 2]]))
            options = dict(root_path=directory, io_path=str(root / 'cache'), io_mode='pickle', verbose=False,
                           online_transform=transforms.ToTensor(), label_transform=transforms.Select('label'))
            dataset = SHUMIDataset(**options)
            signal, label = dataset[4]
            self.assertEqual(tuple(signal.shape), (32, 200))
            self.assertEqual(label, 1)
            self.assertEqual(len(dataset), 32)
            self.assertEqual(len(SHUMIDataset(**options)), 32)
            train, test = next(LeaveOneSubjectOut(split_path=str(root / 'split')).split(dataset, subject='sub-002'))
            self.assertEqual(set(train.info.subject_id), {'sub-001'})
            self.assertEqual(set(test.info.subject_id), {'sub-002'})
            self.assertEqual(len(train), 16)
            self.assertFalse(set(train.info.trial_id) & set(test.info.trial_id))

    def test_original_fake_record_and_electrode_constants(self):
        values = SHUMIDataset.fake_record(record='fake_record')
        row = next(SHUMIDataset.process_record(**values))
        self.assertEqual(row['eeg'].shape, (32, 200))
        self.assertEqual(len(SHU_MI_CHANNEL_LIST), 32)
        self.assertEqual(set(SHU_MI_CHANNEL_LOCATION_DICT), set(SHU_MI_CHANNEL_LIST))
        self.assertEqual(len({tuple(x) for x in SHU_MI_CHANNEL_LOCATION_DICT.values()}), 32)
        for matrix in [SHU_MI_ADJACENCY_MATRIX, SHU_MI_STANDARD_ADJACENCY_MATRIX]:
            self.assertEqual(np.shape(matrix), (32, 32))
            self.assertTrue(np.isfinite(matrix).all())


if __name__ == '__main__':
    unittest.main()
