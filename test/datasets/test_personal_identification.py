import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import M3CVDataset


class TestPersonalIdentificationDataset(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_m3cv_dataset(self):
        io_path = f'./tmp_out/m3cv_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/aistudio'

        dataset = M3CVDataset(io_path=io_path,
                              root_path=root_path,
                              subset='Enrollment',
                              online_transform=transforms.ToTensor(),
                              num_channel=65,
                              num_worker=4)

        self.assertEqual(len(dataset), 57851)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (65, 1000))
        last_item = dataset[57850]
        self.assertEqual(last_item[0].shape, (65, 1000))

        io_path = f'./tmp_out/m3cv_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        dataset = M3CVDataset(io_path=io_path,
                              root_path=root_path,
                              subset='Testing',
                              online_transform=transforms.ToTensor(),
                              num_channel=65,
                              num_worker=4)

        self.assertEqual(len(dataset), 52942)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (65, 1000))
        last_item = dataset[52941]
        self.assertEqual(last_item[0].shape, (65, 1000))

        io_path = f'./tmp_out/m3cv_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        dataset = M3CVDataset(io_path=io_path,
                              root_path=root_path,
                              subset='Calibration',
                              online_transform=transforms.ToTensor(),
                              num_channel=65,
                              num_worker=4)

        self.assertEqual(len(dataset), 6070)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (65, 1000))
        last_item = dataset[6069]
        self.assertEqual(last_item[0].shape, (65, 1000))


if __name__ == '__main__':
    unittest.main()
