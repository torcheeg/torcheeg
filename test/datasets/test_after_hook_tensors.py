from pathlib import Path
import tempfile
import unittest

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader

from torcheeg import transforms
from torcheeg.datasets import BaseDataset, MNERawDataset


def add_one(batch):
    return batch + 1


class TestAfterHookTensors(unittest.TestCase):

    def test_tensor_windows_keep_values_shape_dtype_and_order(self):
        for dtype in (torch.float32, torch.float64, torch.int64):
            for shape in ((2, 4), (4, 3, 3), ()):
                with self.subTest(dtype=dtype, shape=shape):
                    samples = [torch.full(shape, index, dtype=dtype) for index in range(4)]
                    actual = BaseDataset.hook_data_interface(add_one, samples)
                    self.assertEqual(len(actual), len(samples))
                    for index, sample in enumerate(actual):
                        self.assertEqual(sample.shape, samples[index].shape)
                        self.assertEqual(sample.dtype, dtype)
                        self.assertEqual(sample.device, samples[index].device)
                        torch.testing.assert_close(sample, samples[index] + 1)

    def test_single_window_keeps_other_singleton_dimensions(self):
        for shape in ((1, 4), (1, 1, 4), ()):
            with self.subTest(shape=shape):
                sample = torch.ones(shape)
                actual = BaseDataset.hook_data_interface(add_one, [sample])
                self.assertEqual(len(actual), 1)
                self.assertEqual(actual[0].shape, sample.shape)
                torch.testing.assert_close(actual[0], sample + 1)

    def test_trial_centering_example(self):
        def center_trial(batch):
            return batch - batch.mean(dim=-1, keepdim=True)

        samples = [torch.arange(8, dtype=torch.float32).reshape(2, 4) + index
                   for index in range(3)]
        actual = BaseDataset.hook_data_interface(center_trial, samples)
        self.assertEqual(len(actual), 3)
        expected = torch.tensor([[-1.5, -0.5, 0.5, 1.5]] * 2)
        for sample in actual:
            torch.testing.assert_close(sample, expected)

    def test_tensor_gradients_flow_through_each_returned_window(self):
        samples = [torch.full((2, 4), float(index), requires_grad=True) for index in range(3)]
        actual = BaseDataset.hook_data_interface(lambda batch: batch * 3, samples)
        self.assertEqual(len(actual), 3)
        sum(sample.sum() for sample in actual).backward()
        for sample in samples:
            torch.testing.assert_close(sample.grad, torch.full_like(sample, 3.))

    def test_numpy_windows_remain_unchanged(self):
        samples = [np.full((2, 4), index, dtype=np.float64) for index in range(3)]
        actual = BaseDataset.hook_data_interface(add_one, samples)
        self.assertEqual(len(actual), 3)
        for index, sample in enumerate(actual):
            self.assertEqual(sample.shape, (2, 4))
            np.testing.assert_array_equal(sample, np.full((2, 4), index + 1))
            np.testing.assert_array_equal(samples[index], np.full((2, 4), index))

    def test_hook_can_explicitly_return_numpy_from_tensor_input(self):
        samples = [torch.full((2, 4), float(index)) for index in range(3)]
        actual = BaseDataset.hook_data_interface(lambda batch: batch.numpy() + 1, samples)
        self.assertEqual(len(actual), 3)
        for index, sample in enumerate(actual):
            self.assertIsInstance(sample, np.ndarray)
            np.testing.assert_array_equal(sample, np.full((2, 4), index + 1))

    def test_nonarray_list_data_keeps_existing_behavior(self):
        samples = [{'value': index} for index in range(3)]
        actual = BaseDataset.hook_data_interface(
            lambda batch: [{'value': sample['value'] + 1} for sample in batch], samples)
        self.assertEqual(actual, [{'value': 1}, {'value': 2}, {'value': 3}])
        self.assertEqual(samples, [{'value': 0}, {'value': 1}, {'value': 2}])

    def test_real_mne_tensor_hooks_preserve_cached_windows_and_loader_batches(self):
        eeg = np.arange(800, dtype=float).reshape(2, 400)
        raw = mne.io.RawArray(eeg, mne.create_info(['C3', 'C4'], 100, 'eeg'), verbose='ERROR')
        with tempfile.TemporaryDirectory() as directory:
            dataset = MNERawDataset(
                raw_list=[raw], info_list=[{'subject_id': 1, 'trial_id': 0, 'session_id': 0}],
                chunk_size=100, io_mode='pickle', io_path=str(Path(directory) / 'cache'),
                offline_transform=transforms.ToTensor(), after_trial=add_one,
                after_session=add_one, after_subject=add_one, verbose=False)
            self.assertEqual(len(dataset), 4)
            for index in range(4):
                signal, info = dataset[index]
                self.assertEqual(signal.shape, (2, 100))
                self.assertEqual(signal.dtype, torch.float32)
                torch.testing.assert_close(signal, torch.tensor(eeg[:, index * 100:(index + 1) * 100] + 3,
                                                                 dtype=torch.float32))
                self.assertEqual(info['clip_id'], f'1_0_{index * 100}')
            batches = list(DataLoader(dataset, batch_size=2))
            self.assertEqual([batch[0].shape for batch in batches], [(2, 2, 100)] * 2)


if __name__ == '__main__':
    unittest.main()
