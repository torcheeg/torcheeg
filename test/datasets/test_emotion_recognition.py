import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import (AMIGOSDataset, DEAPDataset, DREAMERDataset,
                               MAHNOBDataset, SEEDDataset, SEEDFeatureDataset,
                               SEEDIVDataset, SEEDIVFeatureDataset,
                               MPEDFeatureDataset, BCI2022Dataset)


class TestEmotionRecognitionDataset(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_mped_feature_dataset(self):
        io_path = f'./tmp_out/mped_feature_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/EEG_feature'

        dataset = MPEDFeatureDataset(
            io_path=io_path,
            root_path=root_path,
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Compose([
                transforms.Select('emotion'),
                transforms.Lambda(lambda x: int(x) + 1),
            ]),
            num_worker=4)

        self.assertEqual(len(dataset), 129904)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (62, 5))
        last_item = dataset[129903]
        self.assertEqual(last_item[0].shape, (62, 5))

    def test_mahnob_dataset(self):
        io_path = f'./tmp_out/mahnob_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/Sessions'

        dataset = MAHNOBDataset(io_path=io_path,
                                root_path=root_path,
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Compose([
                                    transforms.Select('feltVlnc'),
                                    transforms.Binary(5.0),
                                ]),
                                num_worker=4)

        self.assertEqual(len(dataset), 16410)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 128))
        last_item = dataset[16409]
        self.assertEqual(last_item[0].shape, (32, 128))

    def test_amigos_dataset(self):
        io_path = f'./tmp_out/amigos_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/data_preprocessed'

        dataset = AMIGOSDataset(io_path=io_path,
                                root_path=root_path,
                                num_trial=16,
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Compose([
                                    transforms.Select('valence'),
                                    transforms.Binary(5.0),
                                ]),
                                num_worker=4)

        self.assertEqual(len(dataset), 45474)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (14, 128))
        last_item = dataset[45473]
        self.assertEqual(last_item[0].shape, (14, 128))

    def test_deap_dataset(self):
        io_path = f'./tmp_out/deap_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/data_preprocessed_python'

        dataset = DEAPDataset(io_path=io_path,
                              root_path=root_path,
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]),
                              num_worker=4)
        self.assertEqual(len(dataset), 76800)
        self.assertEqual(len(dataset.eeg_io), 78080)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (32, 128))
        last_item = dataset[76799]
        self.assertEqual(last_item[0].shape, (32, 128))

    def test_dreamer_dataset(self):
        io_path = f'./tmp_out/dreamer_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        mat_path = './tmp_in/DREAMER.mat'

        dataset = DREAMERDataset(io_path=io_path,
                                 mat_path=mat_path,
                                 online_transform=transforms.ToTensor(),
                                 label_transform=transforms.Compose([
                                     transforms.Select('valence'),
                                     transforms.Binary(3.0),
                                 ]),
                                 num_worker=4)

        self.assertEqual(len(dataset), 85744)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (14, 128))
        last_item = dataset[85743]
        self.assertEqual(last_item[0].shape, (14, 128))

    def test_seed_dataset(self):
        io_path = f'./tmp_out/seed_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/Preprocessed_EEG'

        dataset = SEEDDataset(io_path=io_path,
                              root_path=root_path,
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: int(x) + 1),
                              ]),
                              num_worker=4)

        self.assertEqual(len(dataset), 152730)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (62, 200))
        last_item = dataset[152729]
        self.assertEqual(last_item[0].shape, (62, 200))

    def test_seed_feature_dataset(self):
        io_path = f'./tmp_out/seed_feature_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/ExtractedFeatures'

        dataset = SEEDFeatureDataset(
            io_path=io_path,
            root_path=root_path,
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Compose([
                transforms.Select('emotion'),
                transforms.Lambda(lambda x: int(x) + 1),
            ]),
            num_worker=4)

        self.assertEqual(len(dataset), 152730)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (62, 5))
        last_item = dataset[152729]
        self.assertEqual(last_item[0].shape, (62, 5))

    def test_seed_iv_dataset(self):
        io_path = f'./tmp_out/seed_iv_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/eeg_raw_data'

        dataset = SEEDIVDataset(io_path=io_path,
                                root_path=root_path,
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Select('emotion'),
                                num_worker=4)

        self.assertEqual(len(dataset), 37575)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (62, 800))
        last_item = dataset[37574]
        self.assertEqual(last_item[0].shape, (62, 800))

    def test_seed_iv_feature_dataset(self):
        io_path = f'./tmp_out/seed_iv_feature_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/eeg_feature_smooth'

        dataset = SEEDIVFeatureDataset(
            io_path=io_path,
            root_path=root_path,
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Select('emotion'),
            num_worker=4)

        self.assertEqual(len(dataset), 37575)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (62, 5))
        last_item = dataset[37574]
        self.assertEqual(last_item[0].shape, (62, 5))

    def test_bci2022(self):
        io_path = f'./tmp_out/bci2022_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/TrainSet'

        dataset = BCI2022Dataset(io_path=io_path,
                                 root_path=root_path,
                                 online_transform=transforms.ToTensor(),
                                 label_transform=transforms.Select('emotion'),
                                 channel_num=30,
                                 num_worker=4)

        self.assertEqual(len(dataset), 146812)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (30, 250))
        last_item = dataset[146811]
        self.assertEqual(last_item[0].shape, (30, 250))


if __name__ == '__main__':
    unittest.main()
