import unittest

import pandas as pd

from torcheeg.datasets import BaseDataset, PairDataset


class _FakeDataset(BaseDataset):
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


class TestPairDataset(unittest.TestCase):
    def test_pair_dataset(self):
        dataset_a = _FakeDataset(5, 5, 5)
        dataset_b = _FakeDataset(5, 5, 5)

        pair_dataset = PairDataset([dataset_a, dataset_b], join_key='clip_id')
        self.assertEqual(len(pair_dataset), 5 * 5 * 5)
        for index in (0, len(pair_dataset) - 1):
            signal_a, label_a = dataset_a[index]
            signal_b, label_b = dataset_b[index]
            self.assertEqual(
                pair_dataset[index], (signal_a, signal_b, label_a, label_b))


if __name__ == '__main__':
    unittest.main()
