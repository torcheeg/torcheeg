import unittest
import os
import random
import shutil

from torcheeg.model_selection import KFoldPerSubjectCrossTrial
from torcheeg.datasets import DEAPDataset


class TestKFoldPerSubjectCrossTrial(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_k_fold_per_subject_cross_trial(self):
        io_path = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/data_preprocessed_python'

        dataset = DEAPDataset(io_path=io_path, root_path=root_path)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        k_fold = KFoldPerSubjectCrossTrial(split_path=split_path)

        split_num = 0
        for train_dataset, test_dataset in k_fold.split(dataset):
            self.assertEqual(len(train_dataset), 1920)
            self.assertEqual(len(test_dataset), 480)
            self.assertEqual(len(list(set(train_dataset.info['subject_id']))), 1)
            self.assertEqual(len(list(set(test_dataset.info['subject_id']))), 1)
            split_num += 1
        self.assertEqual(split_num, 32 * 5)


if __name__ == '__main__':
    unittest.main()