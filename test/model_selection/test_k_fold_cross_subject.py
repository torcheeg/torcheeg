import unittest
import os
import random
import shutil

from torcheeg.model_selection import KFoldCrossSubject
from torcheeg.datasets import DEAPDataset


class TestKFoldCrossSubject(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_k_fold(self):
        io_path = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/data_preprocessed_python'

        dataset = DEAPDataset(io_path=io_path, root_path=root_path)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        k_fold = KFoldCrossSubject(split_path=split_path)

        split_num = 0
        train_dataset_len = [60000, 60000, 62400, 62400, 62400]
        test_dataset_len = [16800, 16800, 14400, 14400, 14400]
        for i, (train_dataset,
                test_dataset) in enumerate(k_fold.split(dataset)):
            self.assertEqual(len(train_dataset), train_dataset_len[i])
            self.assertEqual(len(test_dataset), test_dataset_len[i])
            split_num += 1
        self.assertEqual(split_num, 5)


if __name__ == '__main__':
    unittest.main()