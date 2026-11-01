import os
import tempfile
import unittest

import mne
import numpy as np

from torcheeg import transforms
from torcheeg.datasets import FolderDataset


class TestFolderDataset(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._input_path = os.path.join(self._directory.name, 'data')
        self._output_path = os.path.join(self._directory.name, 'output')
        os.makedirs(self._output_path)

    def _generate_eeg_files(self, num_files, folder_path):
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
        folder1 = os.path.join(self._input_path, 'folder1')
        folder2 = os.path.join(self._input_path, 'folder2')
        if not os.path.exists(folder1):
            os.makedirs(folder1)
            self._generate_eeg_files(2, folder1)
        if not os.path.exists(folder2):
            os.makedirs(folder2)
            self._generate_eeg_files(2, folder2)

        # Define input and output paths for folder_constructor
        io_path = os.path.join(self._output_path, 'eeg_folder')
        root_path = self._input_path

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


if __name__ == '__main__':
    unittest.main()
