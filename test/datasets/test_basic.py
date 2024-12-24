import os
import shutil
import unittest

import mne
import numpy as np
import pandas as pd

from torcheeg import transforms
from torcheeg.datasets import (BaseDataset, ConcatDataset, CSVFolderDataset,
                               FolderDataset, PairDataset, MNERawDataset)

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

if __name__ == '__main__':
    unittest.main()
