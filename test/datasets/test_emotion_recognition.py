import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset, DREAMERDataset, SEEDDataset
from torcheeg.datasets.functional import (deap_constructor,
                                          dreamer_constructor,
                                          seed_constructor)


class TestEmotionRecognitionDataset(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./outputs/')
        os.mkdir('./outputs/')

    def test_deap_constructor(self):
        io_path = f'./outputs/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = '~/Data/eeg-datasets/DEAP/data_preprocessed_python'
        deap_constructor(io_path=io_path,
                         root_path=root_path,
                         transform=transforms.BandDifferentialEntropy(),
                         num_worker=4)

    def test_deap_dataset(self):
        io_path = f'./outputs/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = '~/Data/eeg-datasets/DEAP/data_preprocessed_python'

        dataset = DEAPDataset(
            io_path=io_path,
            root_path=root_path,
            offline_transform=transforms.BandDifferentialEntropy(),
            label_transform=transforms.Compose([
                transforms.Select('valence'),
                transforms.Binary(5.0),
            ]),
            num_worker=4)
        self.assertEqual(len(dataset), 76800)
        self.assertEqual(len(dataset.eeg_io), 78080)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 4))
        last_item = dataset[76799]
        self.assertEqual(last_item[0].shape, (32, 4))

    def test_dreamer_constructor(self):
        io_path = f'./outputs/dreamer_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        mat_path = '~/Data/eeg-datasets/DREAMER/DREAMER.mat'

        dreamer_constructor(io_path=io_path, mat_path=mat_path, num_worker=4)

    def test_dreamer_dataset(self):
        io_path = f'./outputs/dreamer_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        mat_path = '~/Data/eeg-datasets/DREAMER/DREAMER.mat'

        dataset = DREAMERDataset(io_path=io_path, mat_path=mat_path)

        self.assertEqual(len(dataset), 85744)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (14, 128))
        last_item = dataset[85743]
        self.assertEqual(last_item[0].shape, (14, 128))

    def test_seed_constructor(self):
        io_path = f'./outputs/seed_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = '~/Data/eeg-datasets/SEED/Preprocessed_EEG'

        seed_constructor(io_path=io_path,
                         root_path=root_path,
                         transform=transforms.BandDifferentialEntropy(),
                         num_worker=9)

    def test_seed_dataset(self):
        io_path = f'./outputs/seed_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = '~/Data/eeg-datasets/SEED/Preprocessed_EEG'

        dataset = SEEDDataset(
            io_path=io_path,
            root_path=root_path,
            # offline_transform=transforms.BandDifferentialEntropy(),
            label_transform=transforms.Compose([
                transforms.Select('emotion'),
                transforms.Lambda(lambda x: x + 1),
            ]),
            num_worker=9)

        self.assertEqual(len(dataset), 152730)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (62, 4))
        last_item = dataset[152729]
        self.assertEqual(last_item[0].shape, (62, 4))


if __name__ == '__main__':
    unittest.main()
