import tempfile
import unittest

import numpy as np

from torcheeg.datasets import LMDBDataset


class TestLMDBDataset(unittest.TestCase):

    def test_reopen_and_append_preserve_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = LMDBDataset(directory)
            dataset.append({'eeg': np.ones((2, 4)), 'baseline': np.zeros(
                (2, 4)), 'info': {'subject_id': '001'}})
            dataset.close()
            dataset = LMDBDataset(directory)
            self.assertEqual(dataset[0]['info']['subject_id'], '001')
            np.testing.assert_array_equal(
                dataset[0]['baseline'], np.zeros((2, 4)))
            dataset.append({'eeg': np.full((2, 4), 2),
                           'info': {'subject_id': '002'}})
            self.assertEqual(len(dataset), 2)
            dataset.close()

    def test_selected_view_reads_in_worker_after_parent_close(self):
        import multiprocessing
        from test.datasets.test_io_dataset import prepare_label

        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as directory:
            dataset = LMDBDataset(directory)
            dataset.extend({'eeg': np.full((2, 4), index),
                            'info': {'label': index}} for index in range(3))
            selected = LMDBDataset(
                directory, label_transform=prepare_label)[[2, 0]]
            dataset.close()
            for method in multiprocessing.get_all_start_methods():
                with self.subTest(method=method):
                    loader = DataLoader(selected, batch_size=2, num_workers=1,
                                        multiprocessing_context=method)
                    batches = list(loader)
                    self.assertEqual(batches[0]['label'].tolist(), [2, 0])
                    np.testing.assert_array_equal(
                        batches[0]['eeg'][0], np.full((2, 4), 2))
            self.assertEqual(dataset[1]['info']['label'], 1)
            selected.close()
