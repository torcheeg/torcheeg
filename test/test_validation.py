import os
import tempfile
import unittest

import numpy as np
import torch

from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.io import EEGSignalIO
from torcheeg.models import FBMSNet, ViT


class TestInputValidation(unittest.TestCase):
    def test_label_input_type_is_checked(self):
        with self.assertRaises(TypeError):
            transforms.Select('label')(y=[1, 2])

    def test_tensor_shape_is_checked(self):
        with self.assertRaises(ValueError):
            transforms.Resize(size=(4, 4))(eeg=torch.zeros(2, 3))

    def test_model_configuration_is_checked(self):
        with self.assertRaises(ValueError):
            ViT(chunk_size=129, t_patch_size=32)
        with self.assertRaises(ValueError):
            FBMSNet(in_channels=9, num_electrodes=22, chunk_size=129, stride_factor=4)
        with self.assertRaisesRegex(ValueError, 'temporal'):
            FBMSNet(in_channels=9, num_electrodes=22, chunk_size=512, temporal='unknown')

    def test_missing_dataset_directory_is_a_file_error(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(FileNotFoundError):
                DEAPDataset(root_path=os.path.join(path, 'missing'),
                            io_path=os.path.join(path, 'cache'), verbose=False)

    def test_io_backends_use_the_same_input_exceptions(self):
        for mode in ('memory', 'pickle', 'lmdb'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as path:
                io = EEGSignalIO(path, io_mode=mode)
                try:
                    with self.assertRaises(ValueError):
                        io.write_eeg(None)
                    with self.assertRaises(KeyError):
                        io.read_eeg('missing')
                    key = io.write_eeg(np.zeros((2, 3)))
                    np.testing.assert_array_equal(io.read_eeg(key), np.zeros((2, 3)))
                finally:
                    del io

    def test_unknown_io_mode_is_an_invalid_option(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ValueError):
                EEGSignalIO(path, io_mode='unknown')


if __name__ == '__main__':
    unittest.main()
