import os
import numpy as np
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import NumpyDataset
from torcheeg.datasets.functional import numpy_constructor


class TestNumpyDataset(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_numpy_constructor(self):
        io_path = f'./tmp_out/numpy_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        X = np.random.randn(100, 32, 128)
        y = {
            'valence': np.random.randint(10, size=100),
            'arousal': np.random.randint(10, size=100)
        }

        numpy_constructor(io_path=io_path, X=X, y=y, num_worker=0)

    def test_numpy_dataset(self):
        io_path = f'./tmp_out/numpy_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        X = np.random.randn(100, 32, 128)
        y = {
            'valence': np.random.randint(10, size=100),
            'arousal': np.random.randint(10, size=100)
        }

        dataset = NumpyDataset(X=X,
                               y=y,
                               io_path=io_path,
                               offline_transform=transforms.Compose(
                                   [transforms.BandDifferentialEntropy()]),
                               online_transform=transforms.ToTensor(),
                               label_transform=transforms.Compose([
                                   transforms.Select('valence'),
                                   transforms.Binary(5.0),
                               ]),
                               num_worker=2,
                               num_samples_per_trial=50)
        
        self.assertEqual(len(dataset), 100)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 4))
        last_item = dataset[99]
        self.assertEqual(last_item[0].shape, (32, 4))


if __name__ == '__main__':
    unittest.main()
