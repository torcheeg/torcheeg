import unittest

import pandas as pd

from torcheeg.datasets import BaseDataset, ConcatDataset


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


class TestConcatDataset(unittest.TestCase):
    def test_concat_dataset(self):
        dataset_a = _FakeDataset(5, 5, 5)
        dataset_b = _FakeDataset(5, 5, 5)

        concat_dataset = ConcatDataset(dataset_a, dataset_b)
        self.assertEqual(len(concat_dataset), 5 * 5 * 5 * 2)
        self.assertEqual(concat_dataset[0], dataset_a[0])
        self.assertEqual(concat_dataset[len(dataset_a)], dataset_b[0])
        self.assertEqual(concat_dataset[-1], dataset_b[-1])


if __name__ == '__main__':
    unittest.main()
