import os
import random
import shutil
import unittest

from torcheeg.datasets import M3CVDataset
from torcheeg.datasets.functional import m3cv_constructor


class TestPersonalIdentificationDataset(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_m3cv_constructor(self):
        io_path = f'./tmp_out/m3cv_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/aistudio'
        m3cv_constructor(io_path=io_path, root_path=root_path, num_worker=0)

    def test_m3cv_dataset(self):
        io_path = f'./tmp_out/m3cv_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/aistudio'

        dataset = M3CVDataset(io_path=io_path,
                              root_path=root_path,
                              subset='Enrollment',
                              num_worker=9)

        self.assertEqual(len(dataset), 57851)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (65, 1000))
        last_item = dataset[57850]
        self.assertEqual(last_item[0].shape, (65, 1000))

        io_path = f'./tmp_out/m3cv_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        dataset = M3CVDataset(io_path=io_path,
                              root_path=root_path,
                              subset='Testing', channel_num=65,
                              num_worker=9)

        self.assertEqual(len(dataset), 52942)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (65, 1000))
        last_item = dataset[52941]
        self.assertEqual(last_item[0].shape, (65, 1000))

        io_path = f'./tmp_out/m3cv_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        dataset = M3CVDataset(io_path=io_path,
                              root_path=root_path,
                              subset='Calibration', channel_num=65,
                              num_worker=9)

        self.assertEqual(len(dataset), 6070)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (65, 1000))
        last_item = dataset[6069]
        self.assertEqual(last_item[0].shape, (65, 1000))


if __name__ == '__main__':
    unittest.main()
