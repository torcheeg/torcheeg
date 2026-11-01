import os
import tempfile
import unittest

import mne
import numpy as np

from torcheeg import transforms
from torcheeg.datasets import MNERawDataset


class TestMNERawDataset(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._input_path = os.path.join(self._directory.name, 'data')
        self._output_path = os.path.join(self._directory.name, 'output')
        os.makedirs(self._output_path)

    def _generate_raw(self, num_files):
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
        raw_list, info_list = self._generate_raw(6)

        # Create dataset
        io_path = os.path.join(self._output_path, 'eeg_raw')
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


if __name__ == '__main__':
    unittest.main()
