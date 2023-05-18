import os
import random
import shutil
import unittest

import numpy as np

from torcheeg import transforms
from torcheeg.datasets import TSUBenckmarkDataset


def mean_reduce(eeg_list):
    return np.array(eeg_list).mean(axis=0)


class TestSSVEPDataset(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_tsu_benchmark_dataset(self):
        io_path = f'./tmp_out/tsu_benchmark_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/TSUBenchmark'

        dataset = TSUBenckmarkDataset(io_path=io_path,
                                      root_path=root_path,
                                      online_transform=transforms.ToTensor(),
                                      num_worker=4)

        self.assertEqual(len(dataset), 50400)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (64, 250))
        last_item = dataset[50399]
        self.assertEqual(last_item[0].shape, (64, 250))


if __name__ == '__main__':
    unittest.main()
