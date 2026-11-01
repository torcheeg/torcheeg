import unittest

import pandas as pd

from torcheeg.datasets import BaseDataset, PairDataset


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


class TestPairDataset(unittest.TestCase):
    def test_pair_dataset(self):
        dataset_a = FakeDataset(5, 5, 5)
        dataset_b = FakeDataset(5, 5, 5)

        pair_dataset = PairDataset([dataset_a, dataset_b], join_key='clip_id')
        assert len(pair_dataset) == 5 * 5 * 5

        # print(f'Saved EEG file {file_path}')


if __name__ == '__main__':
    unittest.main()
