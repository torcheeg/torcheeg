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
        sfreq = 128
        n_channels = 14
        duration = 5
        raw_list = []
        info_list = []
        for i in range(num_files):
            n_samples = sfreq * duration
            data = np.random.randn(n_channels, n_samples)
            ch_names = [f'ch_{i + 1:03}' for i in range(n_channels)]
            ch_types = ['eeg'] * n_channels
            info = mne.create_info(ch_names, sfreq, ch_types)
            raw = mne.io.RawArray(data, info)
            raw_list.append(raw)
            info_list.append({'trial_id': i % 3, 'subject_id': i // 3})
        return (raw_list, info_list)

    def test_mne_raw_dataset(self):
        raw_list, info_list = self._generate_raw(6)
        io_path = os.path.join(self._output_path, 'eeg_raw')
        dataset = MNERawDataset(
            raw_list=raw_list, info_list=info_list, chunk_size=128, overlap=0, io_path=io_path)
        expected_length = len(raw_list) * 5
        self.assertEqual(len(dataset), expected_length)
        first_item = dataset[0]
        self.assertEqual(first_item['eeg'].shape, (14, 128))
        self.assertEqual(first_item['info']['trial_id'], 0)
        last_item = dataset[-1]
        self.assertEqual(last_item['eeg'].shape, (14, 128))
        self.assertEqual(last_item['info']['trial_id'], 2)

    def test_persistent_cache_unions_metadata_fields_across_records(self):
        import pandas as pd
        raw_list, info_list = self._generate_raw(2)
        info_list[1]['note'] = 'second recording'
        dataset = MNERawDataset(raw_list=raw_list, info_list=info_list, chunk_size=128, io_path=os.path.join(
            self._output_path, 'optional_metadata'), io_mode='pickle', verbose=False)
        self.assertEqual(len(dataset), 10)
        self.assertTrue(pd.isna(dataset.info.iloc[0]['note']))
        self.assertEqual(dataset.info.iloc[5]['note'], 'second recording')


if __name__ == '__main__':
    unittest.main()
