import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import TSUBenckmarkDataset
from torcheeg.datasets.functional import tsu_benchmark_constructor

import numpy as np


def mean_reduce(eeg_list):
    return np.array(eeg_list).mean(axis=0)


class TestSSVEPDataset(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_tsu_benchmark_constructor(self):
        io_path = f'./tmp_out/tsu_benchmark_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/TSUBenchmark'
        tsu_benchmark_constructor(io_path=io_path,
                                  root_path=root_path,
                                  num_worker=0)

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

    def test_tsu_benchmark_dataset_from_existing(self):
        io_path = f'./tmp_out/tsu_benchmark_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/TSUBenchmark'

        dataset = TSUBenckmarkDataset(io_path=io_path,
                                      root_path=root_path,
                                      online_transform=transforms.ToTensor(),
                                      num_worker=4)

        io_path = f'./tmp_out/tsu_benchmark_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        dataset = TSUBenckmarkDataset.from_existing(dataset, io_path)

        self.assertEqual(len(dataset), 50400)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (64, 250))
        last_item = dataset[50399]
        self.assertEqual(last_item[0].shape, (64, 250))

        io_path = f'./tmp_out/tsu_benchmark_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        dataset = TSUBenckmarkDataset.from_existing(
            dataset,
            io_path,
            offline_transform=transforms.MeanStdNormalize(),
            online_transform=transforms.ToTensor())

        self.assertEqual(len(dataset), 50400)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (64, 250))
        last_item = dataset[50399]
        self.assertEqual(last_item[0].shape, (64, 250))

    def test_tsu_benchmark_dataset_reduce_from_existing(self):
        io_path = f'./tmp_out/tsu_benchmark_scvrtkbywgiqjdalzfnx'
        root_path = './tmp_in/TSUBenchmark'

        dataset = TSUBenckmarkDataset(io_path=io_path,
                                      root_path=root_path,
                                      online_transform=transforms.ToTensor(),
                                      num_worker=4)

        io_path = f'./tmp_out/tsu_benchmark_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        dataset = TSUBenckmarkDataset.reduce_from_existing(
            dataset,
            io_path,
            reduce_fn=mean_reduce,
            reduce_by='trial_id',
            chunk_size_for_worker=10,
            num_worker=2,
            online_transform=transforms.ToTensor()
        )

        self.assertEqual(len(dataset), 40)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (64, 250))
        last_item = dataset[39]
        self.assertEqual(last_item[0].shape, (64, 250))


if __name__ == '__main__':
    unittest.main()
