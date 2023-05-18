import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.functional.operators import from_existing_dataset


class TestOperators(unittest.TestCase):

    def setUp(self):
        shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_from_existing_dataset(self):
        io_path_1 = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/data_preprocessed_python'

        dataset_1 = DEAPDataset(io_path=io_path_1,
                                root_path=root_path,
                                label_transform=transforms.Compose([
                                    transforms.Select('valence'),
                                    transforms.Binary(5.0),
                                ]),
                                num_worker=4)

        io_path_2 = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        dataset_2 = from_existing_dataset(dataset=dataset_1,
                                          io_path=io_path_2,
                                          label_transform=transforms.Compose([
                                              transforms.Select('valence'),
                                              transforms.Binary(5.0),
                                          ]),
                                          num_worker=4)

        self.assertEqual(len(dataset_1), len(dataset_2))


if __name__ == '__main__':
    unittest.main()
