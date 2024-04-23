import os
import random
import shutil
import unittest

import numpy as np
import torch

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.transforms import (after_hook_normalize, after_hook_running_norm,
                                 after_hook_linear_dynamical_system,
                                 before_hook_normalize)


class TestHooks(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_before_hook(self):
        io_path = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/data_preprocessed_python'

        dataset = DEAPDataset(io_path=io_path,
                              root_path=root_path,
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]),
                              num_worker=4,
                              before_trial=before_hook_normalize)
        self.assertEqual(len(dataset), 76800)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 128))
        last_item = dataset[76799]
        self.assertEqual(last_item[0].shape, (32, 128))

    def test_after_hook(self):
        root_path = './tmp_in/data_preprocessed_python'
        io_path = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        dataset = DEAPDataset(io_path=io_path,
                              root_path=root_path,
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]),
                              num_worker=4,
                              after_subject=after_hook_normalize)
        self.assertEqual(len(dataset), 76800)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 128))
        last_item = dataset[76799]
        self.assertEqual(last_item[0].shape, (32, 128))

        fake_data_np = [np.random.rand(32, 128) for _ in range(32)]
        fake_data_torch = [torch.rand(32, 128) for _ in range(32)]

        # Test after_hook_normalize
        normalized_data = after_hook_normalize(fake_data_np)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        normalized_data = after_hook_normalize(fake_data_torch)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        # Test after_hook_running_norm
        normalized_data = after_hook_running_norm(fake_data_np)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        normalized_data = after_hook_running_norm(fake_data_torch)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        # Test after_hook_linear_dynamical_system
        normalized_data = after_hook_linear_dynamical_system(fake_data_np)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))

        normalized_data = after_hook_linear_dynamical_system(fake_data_torch)
        self.assertEqual(len(normalized_data), 32)
        self.assertEqual(normalized_data[0].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
