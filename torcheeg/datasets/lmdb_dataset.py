import os

from ..io import LMDBEEGIO, CSVInfoIO
from .io_dataset import IODataset


class LMDBDataset(IODataset):
    """Create or open sample storage backed by LMDB and CSV metadata."""

    def __init__(self, io_path, io_size=1048576, transform=None):
        self.io_path = os.fspath(io_path)
        super().__init__(
            LMDBEEGIO(os.path.join(self.io_path, 'eeg'), io_size),
            CSVInfoIO(os.path.join(self.io_path, 'info.csv')),
            transform=transform,
        )
