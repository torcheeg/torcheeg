import os
import random
import shutil
import unittest

import pandas as pd

from torcheeg.datasets import BaseDataset
from torcheeg.model_selection import train_test_split, train_test_split_cross_trial, train_test_split_groupby_trial


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

class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_train_test_split(self):
        dataset = FakeDataset(5, 5, 5)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        train_dataset, test_dataset = train_test_split(dataset=dataset, split_path=split_path, test_size=0.2)

        self.assertEqual(len(train_dataset), 100)
        self.assertEqual(len(test_dataset), 25)

    def test_train_test_split_cross_trial(self):
        dataset = FakeDataset(5, 5, 5)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        train_dataset, test_dataset = train_test_split_cross_trial(dataset=dataset, split_path=split_path, test_size=0.2)

        self.assertEqual(len(train_dataset), 100)
        self.assertEqual(len(test_dataset), 25)

    def test_train_test_split_groupby_trial(self):
        dataset = FakeDataset(5, 5, 5)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        train_dataset, test_dataset = train_test_split_groupby_trial(dataset=dataset, split_path=split_path, test_size=0.2)

        self.assertEqual(len(train_dataset), 100)
        self.assertEqual(len(test_dataset), 25)


if __name__ == '__main__':
    unittest.main()