import unittest
import os
import random
import shutil

from torcheeg.model_selection import train_test_split_dataset
from torcheeg.datasets import DEAPDataset


class TestTrainTestSplitDataset(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./outputs/')
        os.mkdir('./outputs/')

    def test_train_test_split_dataset(self):
        io_path = f'./outputs/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = '/home/zhangzhi/Data/eeg-datasets/DEAP/data_preprocessed_python'

        dataset = DEAPDataset(io_path=io_path, root_path=root_path)

        split_path = f'./outputs/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        train_dataset, test_dataset = train_test_split_dataset(dataset=dataset, split_path=split_path)
        
        self.assertEqual(len(train_dataset), 61440)
        self.assertEqual(len(test_dataset), 15360)


if __name__ == '__main__':
    unittest.main()