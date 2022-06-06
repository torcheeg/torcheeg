import unittest
import os
import random
import shutil

from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.datasets import DEAPDataset


class TestLeaveOneSubjectOut(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./outputs/')
        os.mkdir('./outputs/')

    def test_leave_one_subject_out(self):
        io_path = f'./outputs/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = '/home/zhangzhi/Data/eeg-datasets/DEAP/data_preprocessed_python'

        dataset = DEAPDataset(io_path=io_path, root_path=root_path)

        split_path = f'./outputs/split_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        k_fold = LeaveOneSubjectOut(split_path=split_path)

        split_num = 0
        for train_dataset, test_dataset in k_fold.split(dataset):
            self.assertEqual(len(train_dataset), 74400)
            self.assertEqual(len(test_dataset), 2400)
            split_num += 1
        self.assertEqual(split_num, 32)


if __name__ == '__main__':
    unittest.main()