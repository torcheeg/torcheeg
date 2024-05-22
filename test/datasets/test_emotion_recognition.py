import os
import random
import shutil
import unittest

from torcheeg import transforms
from torcheeg.datasets import (
    AMIGOSDataset, BCI2022Dataset, DEAPDataset, DREAMERDataset, FACEDDataset,
    FACEDFeatureDataset, MAHNOBDataset, MPEDFeatureDataset, SEEDDataset,
    SEEDFeatureDataset, SEEDIVDataset, SEEDIVFeatureDataset, SEEDVDataset,
    SEEDVFeatureDataset)


class TestEmotionRecognitionDataset(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_faced_dataset(self):
        io_path = f'./tmp_out/faced_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/Processed_data'

        dataset = FACEDDataset(io_path=io_path,
                               root_path=root_path,
                               online_transform=transforms.ToTensor(),
                               label_transform=transforms.Select('emotion'),
                               num_worker=4)
        self.assertEqual(len(dataset), 103320)  # 123 subjects * 28 videos * 30s
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (30, 250))
        last_item = dataset[103319]
        self.assertEqual(last_item[0].shape, (30, 250))

    def test_faced_feature_dataset(self):
        io_path = f'./tmp_out/faced_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/EEG_Features/DE'

        dataset = FACEDFeatureDataset(
            io_path=io_path,
            root_path=root_path,
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Select('emotion'),
            num_worker=4)

        self.assertEqual(len(dataset),
                         103320)  # 123 subjects * 28 videos * 30 s
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (30, 5))
        last_item = dataset[103319]
        self.assertEqual(last_item[0].shape, (30, 5))

    def test_seed_v_dataset(self):
        io_path = f'./tmp_out/seed_v_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/EEG_raw'

        dataset = SEEDVDataset(io_path=io_path,
                               root_path=root_path,
                               online_transform=transforms.ToTensor(),
                               label_transform=transforms.Select('emotion'),
                               num_worker=4)

        self.assertEqual(len(dataset), 29168)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (62, 800))
        last_item = dataset[29167]
        self.assertEqual(last_item[0].shape, (62, 800))

    def test_seed_v_feature_dataset(self):
        io_path = f'./tmp_out/seed_v_feature_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
        root_path = './tmp_in/EEG_DE_features'

        dataset = SEEDVFeatureDataset(
            io_path=io_path,
            root_path=root_path,
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Select('emotion'),
            num_worker=4)

        self.assertEqual(len(dataset), 29168)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (62, 5))
        last_item = dataset[29167]
        self.assertEqual(last_item[0].shape, (62, 5))

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
