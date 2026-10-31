import gc
import inspect
import multiprocessing
import os
import shutil
import tempfile
import unittest
from copy import copy

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import (BaseDataset, ConcatDataset, CSVFolderDataset,
                               FolderDataset, MNERawDataset, PairDataset,
                               constants)
from torcheeg.model_selection import (KFoldCrossSubject, Subcategory,
                                      train_test_split)

mne.set_log_level('CRITICAL')


class FakeDataset(BaseDataset):
    def __init__(self, num_subjects, num_trials, num_clips):
        data = []
        for subject_id in range(num_subjects):
            for trial_id in range(num_trials):
                for clip_idx in range(num_clips):
                    clip_id = f"{subject_id}_{trial_id}_{clip_idx}"
                    data.append({
                        'subject_id': subject_id,
                        'trial_id': trial_id,
                        'clip_id': clip_id
                    })

        self.info = pd.DataFrame(data)
        self.eeg_io_router = {}

    def __getitem__(self, index):
        row = self.info.iloc[index]
        return f"signal_{row['clip_id']}", f"label_{row['clip_id']}"

    def __len__(self):
        return len(self.info)


class TestBaseDataset(unittest.TestCase):
    def test_missing_record_hooks_identify_the_subclass(self):
        class IncompleteDataset(BaseDataset):
            pass

        dataset = IncompleteDataset.__new__(IncompleteDataset)
        for method, kwargs in (("set_records", {}),
                               ("read_record", {"record": "record"}),
                               ("process_record", {"record": "record"})):
            with self.subTest(method=method):
                with self.assertRaisesRegex(
                        NotImplementedError,
                        f"Method {method} is not implemented in class IncompleteDataset"):
                    getattr(dataset, method)(**kwargs)

    def test_record_hooks_can_be_overridden_as_static_methods(self):
        class StaticDataset(BaseDataset):
            @staticmethod
            def read_record(record, **kwargs):
                return {"samples": record}

            @staticmethod
            def process_record(record, samples, **kwargs):
                yield {"eeg": samples, "key": record}

        dataset = StaticDataset.__new__(StaticDataset)
        for owner in (StaticDataset, dataset):
            samples = owner.read_record(record="record")
            self.assertEqual(next(owner.process_record(record="record", **samples)),
                             {"eeg": "record", "key": "record"})


class TestFolderDataset(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def generate_dummy_raw(self, num_files):
        # Define EEG specifications
        sfreq = 128  # Sampling rate
        n_channels = 14  # Number of channels
        duration = 5  # Data collected for 5 seconds

        raw_list = []
        info_list = []

        for i in range(num_files):
            # Generate dummy EEG data
            n_samples = sfreq * duration
            data = np.random.randn(n_channels, n_samples)

            # Create MNE Raw object
            ch_names = [f'ch_{i+1:03}' for i in range(n_channels)]
            ch_types = ['eeg'] * n_channels
            info = mne.create_info(ch_names, sfreq, ch_types)
            raw = mne.io.RawArray(data, info)

            # Add to lists
            raw_list.append(raw)
            info_list.append({
                "trial_id": i % 3,  # 3 different trial types
                "subject_id": i // 3  # Every 3 trials belong to one subject
            })

        return raw_list, info_list

    def test_mne_raw_dataset(self):
        # Generate dummy raw objects
        raw_list, info_list = self.generate_dummy_raw(6)

        # Create dataset
        io_path = './tmp_out/eeg_raw'
        dataset = MNERawDataset(
            raw_list=raw_list,
            info_list=info_list,
            chunk_size=128,  # 1 second chunks
            overlap=0,
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Select('trial_id'),
            io_path=io_path
        )

        # Test dataset length
        # Each 5-second recording with chunk_size=128 (1s) should give 5 chunks
        expected_length = len(raw_list) * 5
        self.assertEqual(len(dataset), expected_length)

        # Test first item
        first_item = dataset[0]
        # Shape should be (channels, chunk_size)
        self.assertEqual(first_item[0].shape, (14, 128))
        # Label should be 0 (first trial_id)
        self.assertEqual(first_item[1], 0)

        # Test last item
        last_item = dataset[-1]
        self.assertEqual(last_item[0].shape, (14, 128))
        # Label should be 2 (last trial_id in the sequence 0,1,2,0,1,2)
        self.assertEqual(last_item[1], 2)

    def test_concat_dataset(self):
        dataset_a = FakeDataset(5, 5, 5)
        dataset_b = FakeDataset(5, 5, 5)

        concat_dataset = ConcatDataset(dataset_a, dataset_b)
        assert len(concat_dataset) == 5 * 5 * 5 * 2

    def test_pair_dataset(self):
        dataset_a = FakeDataset(5, 5, 5)
        dataset_b = FakeDataset(5, 5, 5)

        pair_dataset = PairDataset([dataset_a, dataset_b], join_key='clip_id')
        assert len(pair_dataset) == 5 * 5 * 5

    def generate_dummy_eeg_files(self, num_files, folder_path):
        # Define EEG file specifications
        sfreq = 128  # Sampling rate
        n_channels = 14  # Number of channels
        duration = 5  # Data collected for 5 seconds
        for i in range(num_files):
            # Generate dummy EEG data
            n_samples = sfreq * duration
            data = np.random.randn(n_channels, n_samples)

            # Create MNE Raw object
            ch_names = [f'ch_{i+1:03}' for i in range(n_channels)]
            ch_types = ['eeg'] * n_channels
            info = mne.create_info(ch_names, sfreq, ch_types)
            raw = mne.io.RawArray(data, info)

            # Save to file
            file_name = f'sub{i+1}.fif'
            file_path = os.path.join(folder_path, file_name)
            raw.save(file_path)
            # print(f'Saved EEG file {file_path}')

    def test_folder_dataset(self):
        # Generate dummy EEG files and save them to folders
        folder1 = './tmp_in/data/folder1'
        folder2 = './tmp_in/data/folder2'
        if not os.path.exists(folder1):
            os.makedirs(folder1)
            self.generate_dummy_eeg_files(2, folder1)
        if not os.path.exists(folder2):
            os.makedirs(folder2)
            self.generate_dummy_eeg_files(2, folder2)

        # Define input and output paths for folder_constructor
        io_path = './tmp_out/eeg_folder'
        root_path = './tmp_in/data'

        dataset = FolderDataset(io_path=io_path,
                                root_path=root_path,
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Compose([
                                    transforms.Select('label')
                                ]),
                                num_worker=0)

        self.assertEqual(len(dataset), 20)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (14, 128))
        last_item = dataset[-1]
        self.assertEqual(last_item[0].shape, (14, 128))

    def test_csv_folder_dataset(self):
        # Generate dummy EEG files and save them to folders
        folder1 = './tmp_in/data/folder1'
        folder2 = './tmp_in/data/folder2'

        if not os.path.exists(folder1):
            os.makedirs(folder1)
            self.generate_dummy_eeg_files(2, folder1)
        if not os.path.exists(folder2):
            os.makedirs(folder2)
            self.generate_dummy_eeg_files(2, folder2)

        # Define input and output paths for folder_constructor
        csv_path = './tmp_out/data.csv'
        io_path = './tmp_out/eeg_folder'

        df = pd.DataFrame({
            'subject_id': ['sub1', 'sub1', 'sub2', 'sub2'],
            'trial_id': [0, 1, 0, 1],
            'label': [0, 1, 0, 1],
            'file_path': [
                os.path.join(folder1, 'sub1.fif'),
                os.path.join(folder2, 'sub1.fif'),
                os.path.join(folder1, 'sub2.fif'),
                os.path.join(folder2, 'sub2.fif'),
            ]
        })

        df.to_csv(csv_path, index=False)

        dataset = CSVFolderDataset(csv_path=csv_path,
                                   io_path=io_path,
                                   online_transform=transforms.ToTensor(),
                                   label_transform=transforms.Select('label'),
                                   num_worker=0)

        self.assertEqual(len(dataset), 20)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (14, 128))
        last_item = dataset[-1]
        self.assertEqual(last_item[0].shape, (14, 128))


class TestLazyDatasetCopy(unittest.TestCase):
    _modes = ('pickle', 'lmdb')

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._counter = 0
        self._eeg = np.arange(800, dtype=float).reshape(2, 400)

    def _make_dataset(self, mode):
        self._counter += 1
        raw_list = [mne.io.RawArray(self._eeg + 1000 * subject,
                                    mne.create_info(['C3', 'C4'], 100, 'eeg'),
                                    verbose='ERROR') for subject in (1, 2)]
        return MNERawDataset(
            raw_list=raw_list,
            info_list=[{'subject_id': subject, 'trial_id': 0}
                       for subject in (1, 2)],
            chunk_size=100, io_mode=mode, lazy_threshold=0,
            io_path=os.path.join(self._directory.name, f'cache_{self._counter}'), verbose=False)

    def _assert_samples(self, dataset):
        self.assertTrue(dataset.is_lazy())
        self.assertEqual(dataset.eeg_io_router, {})
        for index in range(len(dataset)):
            signal, info = dataset[index]
            start = int(info['clip_id'].split('_')[-1])
            expected = self._eeg[:, start:start + 100] + \
                1000 * info['subject_id']
            np.testing.assert_array_equal(signal, expected)

    def test_copy_preserves_lazy_reads_after_original_is_deleted(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                duplicate = copy(dataset)
                self.assertEqual(duplicate.io_path, dataset.io_path)
                self.assertEqual(duplicate.io_mode, mode)
                del dataset
                gc.collect()
                self._assert_samples(duplicate)

    def test_metadata_and_path_dictionaries_are_independent(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                duplicate = copy(dataset)
                self.assertIsNot(duplicate.info, dataset.info)
                self.assertIsNot(duplicate.eeg_signal_paths,
                                 dataset.eeg_signal_paths)
                signal, _ = dataset[0]
                duplicate.info.loc[0, 'subject_id'] = 99
                duplicate.eeg_signal_paths['unused'] = 'not opened'
                actual, info = duplicate[0]
                np.testing.assert_array_equal(actual, signal)
                self.assertEqual(info['subject_id'], 99)
                self.assertEqual(dataset.info.loc[0, 'subject_id'], 1)
                self.assertNotIn('unused', dataset.eeg_signal_paths)

    def test_subcategory_partitions_read_all_samples(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                splitter = Subcategory(
                    criteria='subject_id', split_path=dataset.io_path + '-categories')
                subsets = list(splitter.split(dataset))
                self.assertEqual(len(subsets), 2)
                self.assertEqual([set(subset.info.subject_id)
                                 for subset in subsets], [{1}, {2}])
                for subset in subsets:
                    self.assertEqual(len(subset), 4)
                    self._assert_samples(subset)
                for subset in splitter.split(dataset):
                    self._assert_samples(subset)

    def test_subject_folds_preserve_lazy_loading_and_isolation(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                splitter = KFoldCrossSubject(
                    n_splits=2, split_path=dataset.io_path + '-folds')
                folds = list(splitter.split(dataset))
                self.assertEqual(len(folds), 2)
                for train, test in folds:
                    self.assertFalse(set(train.info.subject_id)
                                     & set(test.info.subject_id))
                    self.assertEqual(set(train.info.clip_id) | set(
                        test.info.clip_id), set(dataset.info.clip_id))
                    self._assert_samples(train)
                    self._assert_samples(test)

    def test_sample_split_reads_each_cached_row(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                train, test = train_test_split(dataset, test_size=0.25, random_state=42,
                                               split_path=dataset.io_path + '-samples')
                self.assertEqual((len(train), len(test)), (6, 2))
                self.assertFalse(set(train.info.clip_id)
                                 & set(test.info.clip_id))
                self.assertEqual(set(train.info.clip_id) | set(
                    test.info.clip_id), set(dataset.info.clip_id))
                self._assert_samples(train)
                self._assert_samples(test)

    def test_copy_keeps_the_existing_shared_storage_contract(self):
        for mode in self._modes:
            with self.subTest(mode=mode):
                dataset = self._make_dataset(mode)
                duplicate = copy(dataset)
                info = duplicate.read_info(0)
                replacement = np.full((2, 100), 7.)
                duplicate.write_eeg(
                    info['_record_id'], info['clip_id'], replacement)
                np.testing.assert_array_equal(dataset[0][0], replacement)
                np.testing.assert_array_equal(duplicate[0][0], replacement)

    def test_spawned_loader_reads_a_copied_lazy_dataset(self):
        if 'spawn' not in multiprocessing.get_all_start_methods():
            self.skipTest('requires spawn')
        expected = np.stack([self._eeg[:, start:start + 100] + 1000 * subject
                             for subject in (1, 2) for start in range(0, 400, 100)])
        for mode in self._modes:
            with self.subTest(mode=mode):
                duplicate = copy(self._make_dataset(mode))
                batches = list(DataLoader(duplicate, batch_size=2, num_workers=2,
                                          multiprocessing_context='spawn', timeout=30))
                torch.testing.assert_close(torch.cat([batch[0] for batch in batches]),
                                           torch.from_numpy(expected))
                self._assert_samples(duplicate)


class TestDatasetConstants(unittest.TestCase):
    def test_auditory_montages_preserve_order_and_cover_all_channels(self):
        for prefix, count in (('AVED', 32), ('DTU', 64), ('KUL', 64)):
            with self.subTest(dataset=prefix):
                channels = getattr(constants, prefix + '_CHANNEL_LIST')
                locations = getattr(constants, prefix +
                                    '_CHANNEL_LOCATION_DICT')
                adjacency = np.asarray(
                    getattr(constants, prefix + '_ADJACENCY_MATRIX'))
                self.assertEqual(len(channels), count)
                self.assertEqual(len(set(channels)), count)
                self.assertEqual(set(locations), set(channels))
                self.assertEqual(adjacency.shape, (count, count))
                np.testing.assert_array_equal(adjacency, adjacency.T)
                grid = transforms.ToGrid(locations)(
                    eeg=np.ones((count, 8)))['eeg']
                for channel in channels:
                    row, column = locations[channel]
                    np.testing.assert_array_equal(
                        grid[:, row, column], np.ones(8))
        self.assertEqual(constants.DTU_CHANNEL_LIST[47], 'CZ')
        self.assertEqual(constants.KUL_CHANNEL_LIST[47], 'CZ')
        self.assertEqual(
            constants.DTUPROCESSED_AVAILABLE_CHANNEL_LIST[-2:], ['EXG1', 'EXG2'])
        self.assertTrue({'O1', 'OZ', 'O2'} <= set(constants.AVED_CHANNEL_LIST))
        self.assertFalse({'HEOL', 'HEOR', 'VEOU', 'VEOL'} &
                         set(constants.AVED_CHANNEL_LIST))

    def test_sleep_channels_follow_reader_defaults(self):
        from torcheeg.datasets import (HMCDataset, ISRUCDataset, P2018Dataset,
                                       SleepEDFxDataset)
        for cls, channels in (
                (HMCDataset, constants.HMC_CHANNEL_LIST),
                (ISRUCDataset, constants.ISRUC_CHANNEL_LIST),
                (P2018Dataset, constants.P2018_CHANNEL_LIST),
                (SleepEDFxDataset, constants.SLEEPEDFX_RAW_CHANNEL_LIST)):
            with self.subTest(dataset=cls.__name__):
                self.assertEqual(inspect.signature(
                    cls.__init__).parameters['channels'].default, channels)

    def test_grid_adjacency_ignores_empty_cells_and_includes_diagonals(self):
        from torcheeg.datasets.constants.utils import \
            format_adj_list_from_location_list
        adjacency = format_adj_list_from_location_list(
            [['A', '-'], ['B', 'C']])
        self.assertEqual({key: set(value) for key, value in adjacency.items()},
                         {'A': {'B', 'C'}, 'B': {'A', 'C'}, 'C': {'A', 'B'}})


if __name__ == '__main__':
    unittest.main()
