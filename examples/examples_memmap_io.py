"""
Load memory-mapped EEG with worker processes
============================================

This example uses synthetic arrays to demonstrate storage and data loading.
It does not train a model or establish a speed or memory benchmark.
Run it from an installed TorchEEG checkout::

    python examples/examples_memmap_io.py

Each read returns a read-only NumPy mapping. Copy the array before exposing
it as a writable tensor or applying an in-place transformation. The map's
file stays open for as long as that array or one of its views is retained.
"""

import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torcheeg.io import EEGSignalIO


class StoredEEGSamples(Dataset):
    def __init__(self, eeg_io):
        self.eeg_io = eeg_io
        self.sample_keys = eeg_io.keys()

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, index):
        eeg = self.eeg_io.read_eeg(self.sample_keys[index])
        return torch.from_numpy(np.array(eeg, copy=True))


def main():
    with tempfile.TemporaryDirectory(prefix='torcheeg-memmap-') as path:
        eeg_io = EEGSignalIO(path, io_mode='memmap')
        for index in range(6):
            eeg_io.write_eeg(np.full((2, 128), index, dtype=np.float32))
        loader = DataLoader(StoredEEGSamples(eeg_io), batch_size=2,
                            num_workers=2, multiprocessing_context='spawn')
        samples = torch.cat(list(loader))
        expected = torch.stack([torch.full((2, 128), float(index)) for index in range(6)])
        torch.testing.assert_close(samples, expected)
        print({'samples': len(samples), 'shape': tuple(samples.shape),
               'worker_processes': 2, 'values_verified': True})


if __name__ == '__main__':
    main()
