import os
import random
import shutil
import unittest

import pandas as pd

from torcheeg.datasets import BaseDataset
from torcheeg.model_selection import (KFold, KFoldCrossSubject,
                                      KFoldCrossTrial, KFoldGroupbyTrial, Subcategory)


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
        self.eeg_io_router = {
            'fake_router': 'fake_router'
        }

    def __getitem__(self, index):
        row = self.info.iloc[index]
        return f"signal_{row['clip_id']}", f"label_{row['clip_id']}"

    def __len__(self):
        return len(self.info)


class TestKFoldCrossSubject(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_k_fold_cross_subject(self):
        dataset = FakeDataset(5, 5, 5)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        k_fold = KFoldCrossSubject(split_path=split_path, n_splits=5)

        split_num = 0
        train_dataset_len = [100, 100, 100, 100, 100]
        test_dataset_len = [25, 25, 25, 25, 25]
        for i, (train_dataset,
                test_dataset) in enumerate(k_fold.split(dataset)):
            self.assertEqual(len(train_dataset), train_dataset_len[i])
            self.assertEqual(len(test_dataset), test_dataset_len[i])
            split_num += 1
        self.assertEqual(split_num, 5)

    def test_k_fold(self):
        dataset = FakeDataset(5, 5, 5)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        k_fold = KFold(split_path=split_path, n_splits=5)

        split_num = 0
        train_dataset_len = [100, 100, 100, 100, 100]
        test_dataset_len = [25, 25, 25, 25, 25]
        for i, (train_dataset,
                test_dataset) in enumerate(k_fold.split(dataset)):
            self.assertEqual(len(train_dataset), train_dataset_len[i])
            self.assertEqual(len(test_dataset), test_dataset_len[i])
            split_num += 1
        self.assertEqual(split_num, 5)

    def test_kfold_cross_trial(self):
        dataset = FakeDataset(5, 5, 5)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        k_fold = KFoldCrossTrial(split_path=split_path, n_splits=5)

        split_num = 0
        train_dataset_len = [100, 100, 100, 100, 100]
        test_dataset_len = [25, 25, 25, 25, 25]
        for i, (train_dataset,
                test_dataset) in enumerate(k_fold.split(dataset)):
            self.assertEqual(len(train_dataset), train_dataset_len[i])
            self.assertEqual(len(test_dataset), test_dataset_len[i])
            split_num += 1
        self.assertEqual(split_num, 5)

    def test_kfold_groupby_trial(self):
        dataset = FakeDataset(5, 5, 5)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        k_fold = KFoldGroupbyTrial(split_path=split_path, n_splits=5)

        split_num = 0
        train_dataset_len = [100, 100, 100, 100, 100]
        test_dataset_len = [25, 25, 25, 25, 25]
        for i, (train_dataset,
                test_dataset) in enumerate(k_fold.split(dataset)):
            self.assertEqual(len(train_dataset), train_dataset_len[i])
            self.assertEqual(len(test_dataset), test_dataset_len[i])
            split_num += 1
        self.assertEqual(split_num, 5)

    def test_subcategory(self):
        dataset = FakeDataset(5, 5, 5)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        cv = Subcategory(split_path=split_path, criteria='subject_id')

        split_num = 0
        for _ in cv.split(dataset):
            split_num += 1
        self.assertEqual(split_num, 5)

if __name__ == '__main__':
    unittest.main()
