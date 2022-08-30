import unittest
import os
import random
import shutil

from torcheeg.model_selection import Subcategory
from torcheeg.datasets import M3CVDataset
from torcheeg import transforms


class TestSubcategory(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_k_fold(self):
        io_path = f'./tmp_out/m3cv_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/aistudio'

        dataset = M3CVDataset(io_path=io_path, root_path=root_path)

        split_path = f'./tmp_out/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        cv = Subcategory(split_path=split_path)

        split_num = 0
        for subdataset in cv.split(dataset):
            # print(len(subdataset))
            split_num += 1
        self.assertEqual(split_num, 13)


if __name__ == '__main__':
    unittest.main()