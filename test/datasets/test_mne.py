import os
import random
import shutil
import unittest

import mne
import numpy as np
from torcheeg import transforms
from torcheeg.datasets import MNEDataset
from torcheeg.datasets.functional import mne_constructor


class TestMNEDataset(unittest.TestCase):
    def setUp(self):
        shutil.rmtree('./tmp_out/')
        os.mkdir('./tmp_out/')

    def test_numpy_constructor(self):
        io_path = f'./tmp_out/numpy_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

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
        mne_constructor(epochs_list, metadata_list, io_path=io_path, num_worker=0)

    def test_numpy_dataset(self):
        io_path = f'./tmp_out/numpy_{"".join(random.sample("zyxwvutsrqponmlkjihgfedcba", 20))}'

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


if __name__ == '__main__':
    unittest.main()
