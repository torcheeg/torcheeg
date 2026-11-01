import os
import tempfile
import unittest

import mne
import numpy as np
import pandas as pd

from torcheeg import transforms
from torcheeg.datasets import CSVFolderDataset


class TestCsvFolderDataset(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._input_path = os.path.join(self._directory.name, 'data')
        self._output_path = os.path.join(self._directory.name, 'output')
        os.makedirs(self._output_path)

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

    def test_csv_folder_dataset(self):
        # Generate dummy EEG files and save them to folders
        folder1 = os.path.join(self._input_path, 'folder1')
        folder2 = os.path.join(self._input_path, 'folder2')

        if not os.path.exists(folder1):
            os.makedirs(folder1)
            self.generate_dummy_eeg_files(2, folder1)
        if not os.path.exists(folder2):
            os.makedirs(folder2)
            self.generate_dummy_eeg_files(2, folder2)

        # Define input and output paths for folder_constructor
        csv_path = os.path.join(self._output_path, 'data.csv')
        io_path = os.path.join(self._output_path, 'eeg_folder')

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
