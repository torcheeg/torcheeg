import os
import numpy as np
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import NumpyDataset


class TestNumpyDataset(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_numpy_dataset(self):
        io_path = f'./tmp_out/numpy_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        X = np.random.randn(100, 32, 128)
        y = np.random.randint(10, size=(100, 2))

        in_memory_space = [True, False]
        for in_memory in in_memory_space:
            dataset = NumpyDataset(X=X,
                                   y=y,
                                   io_path=io_path,
                                   offline_transform=transforms.Compose(
                                       [transforms.BandDifferentialEntropy()]),
                                   online_transform=transforms.ToTensor(),
                                   label_transform=transforms.Compose([
                                       transforms.Select('0'),
                                       transforms.Binary(5.0),
                                   ]),
                                   num_worker=2,
                                   num_samples_per_worker=50,
                                   in_memory=in_memory)

            self.assertEqual(len(dataset), 100)
            first_item = dataset[0]
            self.assertEqual(first_item[0].shape, (32, 4))
            last_item = dataset[99]
            self.assertEqual(last_item[0].shape, (32, 4))

        io_mode_space = ['lmdb', 'pickle']
        for io_mode in io_mode_space:
            io_path = f'./tmp_out/numpy_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
            dataset = NumpyDataset(X=X,
                                   y=y,
                                   io_path=io_path,
                                   offline_transform=transforms.Compose(
                                       [transforms.BandDifferentialEntropy()]),
                                   online_transform=transforms.ToTensor(),
                                   label_transform=transforms.Compose([
                                       transforms.Select('0'),
                                       transforms.Binary(5.0),
                                   ]),
                                   num_worker=2,
                                   num_samples_per_worker=50,
                                   io_mode=io_mode)

            self.assertEqual(len(dataset), 100)
            first_item = dataset[0]
            self.assertEqual(first_item[0].shape, (32, 4))
            last_item = dataset[99]
            self.assertEqual(last_item[0].shape, (32, 4))

    def test_numpy_dataset_from_files(self):
        io_path = f'./tmp_out/numpy_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        tmp_path = f'./tmp_out/tmp_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        X = np.random.randn(100, 32, 128)
        y = np.random.randint(10, size=(100, 2))

        X_path_1 = os.path.join(tmp_path, 'X_0.npy')
        X_path_2 = os.path.join(tmp_path, 'X_1.npy')
        np.save(X_path_1, X[:50])
        np.save(X_path_2, X[50:])
        y_path_1 = os.path.join(tmp_path, 'y_0.npy')
        y_path_2 = os.path.join(tmp_path, 'y_1.npy')
        np.save(y_path_1, y[:50])
        np.save(y_path_2, y[50:])

        dataset = NumpyDataset(
            X=[X_path_1, X_path_2],
            y=[y_path_1, y_path_2],
            io_path=io_path,
            offline_transform=transforms.Compose(
                [transforms.BandDifferentialEntropy()]),
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Compose([
                transforms.Select('0'),  # first dimension
                transforms.Binary(5.0)
            ]),
            num_worker=2,
            num_samples_per_worker=50)

        self.assertEqual(len(dataset), 100)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 4))
        last_item = dataset[99]
        self.assertEqual(last_item[0].shape, (32, 4))


if __name__ == '__main__':
    unittest.main()
