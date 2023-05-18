import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.functional.operators import from_existing


class TestOperators(unittest.TestCase):
    def setUp(self):
        # if os.path.exists('./tmp_out/'):
        #     shutil.rmtree('./tmp_out/')
        # os.mkdir('./tmp_out/')
        ...

    def test_from_existing(self):
        io_path_1 = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        # io_path_1 = './tmp_out/deap_abjxktvgfuedcopnzrwy'
        root_path = './tmp_in/data_preprocessed_python'

        dataset_1 = DEAPDataset(io_path=io_path_1,
                                root_path=root_path,
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Compose([
                                    transforms.Select('valence'),
                                    transforms.Binary(5.0),
                                ]),
                                num_worker=4)

        io_path_2 = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        # io_path_2 = ' ./tmp_out/deap_hsapkcnjuvetflgrzxdq'
        dataset_2 = from_existing(dataset_1,
                                  io_path=io_path_2,
                                  online_transform=transforms.ToTensor(),
                                  label_transform=transforms.Compose([
                                      transforms.Select('valence'),
                                      transforms.Binary(5.0),
                                  ]),
                                  num_worker=4)

        self.assertEqual(len(dataset_1), len(dataset_2))
        # check if the data is the same (the first 10 samples)
        for i in range(10):
            self.assertTrue((dataset_1[0][0] == dataset_2[0][0]).all())

if __name__ == '__main__':
    unittest.main()
