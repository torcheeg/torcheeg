import unittest
import os
import random
import shutil

from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.datasets import DEAPDataset


class TestKFoldGroupbyTrial(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_k_fold_trial(self):
        io_path = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/data_preprocessed_python'

        dataset = DEAPDataset(io_path=io_path, root_path=root_path)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        k_fold = KFoldGroupbyTrial(split_path=split_path)

        split_num = 0
        for train_dataset, test_dataset in k_fold.split(dataset):
            self.assertEqual(len(train_dataset), 61440)
            self.assertEqual(len(test_dataset), 15360)
            split_num += 1
        self.assertEqual(split_num, 5)


if __name__ == '__main__':
    unittest.main()