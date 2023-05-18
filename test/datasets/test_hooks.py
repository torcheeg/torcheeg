import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.functional.hooks import before_trial_normalize, after_trial_normalize, after_trial_moving_avg


class TestHooks(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_before_trial_after_trial(self):
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
                              before_trial=before_trial_normalize)
        self.assertEqual(len(dataset), 76800)
        self.assertEqual(len(dataset.eeg_io), 78080)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 128))
        last_item = dataset[76799]
        self.assertEqual(last_item[0].shape, (32, 128))

        io_path = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        dataset = DEAPDataset(io_path=io_path,
                              root_path=root_path,
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]),
                              num_worker=4,
                              after_trial=after_trial_normalize)
        self.assertEqual(len(dataset), 76800)
        self.assertEqual(len(dataset.eeg_io), 78080)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 128))
        last_item = dataset[76799]
        self.assertEqual(last_item[0].shape, (32, 128))

        io_path = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        dataset = DEAPDataset(io_path=io_path,
                              root_path=root_path,
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]),
                              num_worker=4,
                              after_trial=after_trial_moving_avg)
        self.assertEqual(len(dataset), 76800)
        self.assertEqual(len(dataset.eeg_io), 78080)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 128))
        last_item = dataset[76799]
        self.assertEqual(last_item[0].shape, (32, 128))


if __name__ == '__main__':
    unittest.main()
