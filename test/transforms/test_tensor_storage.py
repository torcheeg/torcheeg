import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

from torcheeg.transforms import ToTensor


class TestTensorStorage(unittest.TestCase):
    def test_readonly_arrays_are_copied_without_warnings(self):
        for dtype in [np.float32, np.float64, np.int16]:
            with self.subTest(dtype=dtype):
                eeg = np.arange(12, dtype=dtype).reshape(3, 4)
                expected = eeg.copy()
                eeg.flags.writeable = False
                with warnings.catch_warnings():
                    warnings.simplefilter('error', UserWarning)
                    actual = ToTensor()(eeg=eeg)['eeg']
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32))
                self.assertNotEqual(actual.data_ptr(), eeg.ctypes.data)
                actual.add_(1)
                np.testing.assert_array_equal(eeg, expected)

    def test_readonly_noncontiguous_baseline_is_independent(self):
        eeg = np.arange(24, dtype=np.float32).reshape(3, 8)[:, ::2]
        baseline = eeg + 100
        eeg.flags.writeable = False
        baseline.flags.writeable = False
        expected = eeg.copy()
        expected_baseline = baseline.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            result = ToTensor(apply_to_baseline=True)(eeg=eeg, baseline=baseline)
        torch.testing.assert_close(result['eeg'], torch.tensor(expected))
        torch.testing.assert_close(result['baseline'], torch.tensor(expected_baseline))
        result['eeg'].mul_(2)
        result['baseline'].zero_()
        np.testing.assert_array_equal(eeg, expected)
        np.testing.assert_array_equal(baseline, expected_baseline)

    def test_readonly_memmap_file_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'signal.dat'
            expected = np.arange(12, dtype=np.float32).reshape(3, 4)
            expected.tofile(path)
            before = path.read_bytes()
            eeg = np.memmap(path, dtype=np.float32, mode='r', shape=(3, 4))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', UserWarning)
                    actual = ToTensor()(eeg=eeg)['eeg']
                self.assertNotEqual(actual.data_ptr(), eeg.ctypes.data)
                torch.testing.assert_close(actual, torch.tensor(expected))
                actual.add_(100)
                self.assertEqual(path.read_bytes(), before)
                np.testing.assert_array_equal(eeg, expected)
            finally:
                del eeg

    def test_writable_float32_keeps_existing_shared_storage(self):
        eeg = np.arange(12, dtype=np.float32).reshape(3, 4)
        actual = ToTensor()(eeg=eeg)['eeg']
        self.assertEqual(actual.data_ptr(), eeg.ctypes.data)
        actual[0, 0] = 23
        self.assertEqual(eeg[0, 0], 23)
        eeg[0, 1] = 31
        self.assertEqual(actual[0, 1].item(), 31)


if __name__ == '__main__':
    unittest.main()
