import os
import random
import shutil
import unittest

import numpy as np

from torcheeg import transforms
from torcheeg.datasets import SleepEDFxDataset


def mean_reduce(eeg_list):
    return np.array(eeg_list).mean(axis=0)


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

        self.assertEqual(len(dataset), 415593)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (2, 3000))
        last_item = dataset[50399]
        self.assertEqual(last_item[0].shape, (2, 3000))


if __name__ == '__main__':
    unittest.main()
