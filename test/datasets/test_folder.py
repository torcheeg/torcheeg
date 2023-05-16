
import sys
sys.path.insert(0, "../torcheeg")

import os
import random
import shutil
import unittest
import numpy as np
from torcheeg import transforms
from torcheeg.datasets import FolderDataset
from torcheeg.datasets.functional import folder_constructor
import mne

class TestFolderDataset(unittest.TestCase):
    def setUp(self):
        if  os.path.isdir("./tmp_out/"):
            shutil.rmtree('./tmp_out/')
            os.mkdir('./tmp_out/')
            
        if  os.path.isdir("./tmp_in/"):
            shutil.rmtree('./tmp_in/')
            os.mkdir('./tmp_in/')

    def generate_dummy_eeg_files(self,num_files, folder_path):
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
            file_name = f'dummy_eeg_{i+1}.fif'
            file_path = os.path.join(folder_path, file_name)
            raw.save(file_path)
            # print(f'Saved EEG file {file_path}')

    def read_eeg_file(self, file_path, chunk_size = 100):
        # Load EEG file
        raw = mne.io.read_raw(file_path)
        # Convert raw to epochs
        epochs = mne.make_fixed_length_epochs(raw,duration=1)
        # Return EEG data
        return epochs

    def test_folder_constructor(self):
        # Generate dummy EEG files and save them to folders
        folder1 = './tmp_in/data/folder1'
        folder2 = './tmp_in/data/folder2'
        if not os.path.exists(folder1):
            os.makedirs(folder1)
        if not os.path.exists(folder2):
            os.makedirs(folder2)

        self.generate_dummy_eeg_files(2, folder1)
        self.generate_dummy_eeg_files(2, folder2)

        # Define input and output paths for folder_constructor
        io_path = './tmp_out/eeg_folder'
        root_path = './tmp_in/data'

        # Call folder_constructor and pass read_eeg_file as argument
        folder_constructor(io_path=io_path,
                            root_path=root_path,
                            num_channel=14,
                            read_func=self.read_eeg_file,
                            num_worker=0)


    def test_folder_dataset(self):
        # Generate dummy EEG files and save them to folders
        folder1 = './tmp_in/data/folder1'
        folder2 = './tmp_in/data/folder2'
        if not os.path.exists(folder1):
            os.makedirs(folder1)
        if not os.path.exists(folder2):
            os.makedirs(folder2)

        self.generate_dummy_eeg_files(2, folder1)
        self.generate_dummy_eeg_files(2, folder2)

        # Define input and output paths for folder_constructor
        io_path = './tmp_out/eeg_folder'
        root_path = './tmp_in/data'

        label_map = {'folder1':0,'folder2':1}
        dataset = FolderDataset(
            io_path=io_path,
            root_path=root_path,
            num_channel=14,
            read_func = self.read_eeg_file,
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Compose([
                transforms.Select('label'),
                transforms.Lambda(lambda x: label_map[x] )
            ]),
            num_worker=0)
        
        self.assertEqual(len(dataset), 20)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (14, 128))
        self.assertEqual(first_item[1], 0)
        last_item = dataset[-1]
        self.assertEqual(last_item[0].shape, (14, 128))
        self.assertEqual(last_item[1], 1)

if __name__ == '__main__':
    unittest.main()
