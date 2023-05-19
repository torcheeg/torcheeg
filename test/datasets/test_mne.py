import os
import random
import shutil
import unittest

import mne
from torcheeg import transforms
from torcheeg.datasets import MNEDataset

mne.set_log_level('CRITICAL')

class TestMNEDataset(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp_out/'):
            shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_mne_dataset(self):
        io_path = f'./tmp_out/mne_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        metadata_list = [{
            'subject': 1,
            'run': 3
        }, {
            'subject': 1,
            'run': 7
        }, {
            'subject': 1,
            'run': 11
        }]

        epochs_list = []
        for metadata in metadata_list:
            physionet_path = mne.datasets.eegbci.load_data(metadata['subject'],
                                                           metadata['run'],
                                                           update_path=False)[0]

            raw = mne.io.read_raw_edf(physionet_path,
                                      preload=True,
                                      stim_channel='auto')
            events, _ = mne.events_from_annotations(raw)
            picks = mne.pick_types(raw.info,
                                   meg=False,
                                   eeg=True,
                                   stim=False,
                                   eog=False,
                                   exclude='bads')
            epochs_list.append(mne.Epochs(raw, events, picks=picks))

        dataset = MNEDataset(epochs_list=epochs_list,
                             metadata_list=metadata_list,
                             chunk_size=50,
                             overlap=0,
                             num_channel=58,
                             io_path=io_path,
                             offline_transform=transforms.Compose(
                                 [transforms.BandDifferentialEntropy()]),
                             online_transform=transforms.ToTensor(),
                             label_transform=transforms.Compose(
                                 [transforms.Select('event')]),
                             num_worker=2)

        self.assertEqual(len(dataset), 174)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (58, 4))
        last_item = dataset[173]
        self.assertEqual(last_item[0].shape, (58, 4))

    def test_mne_dataset_from_files(self):
        io_path = f'./tmp_out/mne_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

        metadata_list = [{
            'subject': 1,
            'run': 3
        }, {
            'subject': 1,
            'run': 7
        }, {
            'subject': 1,
            'run': 11
        }]

        epochs_list = []
        for metadata in metadata_list:
            physionet_path = mne.datasets.eegbci.load_data(metadata['subject'],
                                                           metadata['run'],
                                                           update_path=False)[0]
            raw = mne.io.read_raw_edf(physionet_path,
                                      preload=True,
                                      stim_channel='auto')
            events, _ = mne.events_from_annotations(raw)
            picks = mne.pick_types(raw.info,
                                   meg=False,
                                   eeg=True,
                                   stim=False,
                                   eog=False,
                                   exclude='bads')
            epochs_path = f'./tmp_out/tmp_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'
            mne.Epochs(raw, events, picks=picks).save(epochs_path)
            epochs_list.append(epochs_path)

        dataset = MNEDataset(epochs_list=epochs_list,
                             metadata_list=metadata_list,
                             chunk_size=50,
                             overlap=0,
                             num_channel=58,
                             io_path=io_path,
                             offline_transform=transforms.Compose(
                                 [transforms.BandDifferentialEntropy()]),
                             online_transform=transforms.ToTensor(),
                             label_transform=transforms.Compose(
                                 [transforms.Select('event')]),
                             num_worker=3)

        self.assertEqual(len(dataset), 174)
        first_item = dataset[0]
        self.assertEqual(first_item[0].shape, (58, 4))
        last_item = dataset[173]
        self.assertEqual(last_item[0].shape, (58, 4))


if __name__ == '__main__':
    unittest.main()
