import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import SleepEDFxDataset, HMCDataset, ISRUCDataset, P2018Dataset


class TestSleepStageDetectionDataset(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_sleep_edfx_dataset(self):
        io_path = f'./tmp_out/SleepEDF_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/sleep-edf-database-expanded-1.0.0'

        dataset = SleepEDFxDataset(io_path=io_path,
                                   root_path=root_path,
                                   online_transform=transforms.ToTensor(),
                                   num_worker=4)

        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (2, 3000))

    def test_hmc_dataset(self):
        io_path = f'./tmp_out/HMC_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/HMC/recordings'

        dataset = HMCDataset(io_path=io_path,
                             root_path=root_path,
                             online_transform=transforms.ToTensor(),
                             num_worker=4)
        
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (4, 3000))

    def test_isruc_dataset(self):
        io_path = f'./tmp_out/ISRUC_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/ISRUC-SLEEP'

        dataset = ISRUCDataset(io_path=io_path,
                               root_path=root_path,
                               online_transform=transforms.ToTensor(),
                               num_worker=4)

        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (6, 3000))

    def test_p2018_dataset(self):
        io_path = f'./tmp_out/P2018_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/P2018/training/'

        dataset = P2018Dataset(io_path=io_path,
                               root_path=root_path,
                               online_transform=transforms.ToTensor(),
                               num_worker=4)

        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (6, 3000))


if __name__ == '__main__':
    unittest.main()
