import unittest

import pandas as pd

from torcheeg.datasets import BaseDataset, PairDataset


class _FakeDataset(BaseDataset):

    def __init__(self, num_subjects, num_trials, num_clips):
        data = []
        for subject_id in range(num_subjects):
            for trial_id in range(num_trials):
                for clip_idx in range(num_clips):
                    clip_id = f'{subject_id}_{trial_id}_{clip_idx}'
                    data.append({'subject_id': subject_id,
                                'trial_id': trial_id, 'clip_id': clip_id})
        self.info = pd.DataFrame(data)
        self.eeg_io_router = {}

    def __getitem__(self, index):
        row = self.info.iloc[index]
        return {"eeg": f"signal_{row['clip_id']}", "info": {"label": f"label_{row['clip_id']}"}}

    def __len__(self):
        return len(self.info)


class TestPairDataset(unittest.TestCase):

    def test_pair_dataset(self):
        dataset_a = _FakeDataset(5, 5, 5)
        dataset_b = _FakeDataset(5, 5, 5)
        pair_dataset = PairDataset([dataset_a, dataset_b], join_key='clip_id')
        self.assertEqual(len(pair_dataset), 5 * 5 * 5)
        for index in (0, len(pair_dataset) - 1):
            sample = pair_dataset[index]
            self.assertEqual(
                sample['eeg'], (dataset_a[index]['eeg'], dataset_b[index]['eeg']))
            self.assertEqual(sample['info']['sources'], [
                             dataset_a[index]['info'], dataset_b[index]['info']])

    def test_selection_preserves_constructor_transform(self):
        import numpy as np

        from torcheeg.datasets import MemoryDataset
        first, second = MemoryDataset(), MemoryDataset()
        for dataset in (first, second):
            dataset.extend({'eeg': np.array([index]), 'info': {'subject_id': index}}
                           for index in range(3))
        combined = PairDataset([first, second], join_key='subject_id')
        selected = combined[[2, 0, 2]]
        self.assertEqual(len(selected), 3)
        for index, original in enumerate((2, 0, 2)):
            np.testing.assert_array_equal(
                selected[index]['eeg'], combined[original]['eeg'])
        view = PairDataset([first, second], join_key='subject_id',
                           transform=lambda eeg, **kwargs: {'eeg': eeg, 'marker': True})[[2, 0, 2]]
        self.assertTrue(view[0]['marker'])
        self.assertNotIn('marker', selected[0])
        self.assertIn('PairDataset', repr(combined))
        combined.close()
