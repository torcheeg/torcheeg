from ..io import MemoryEEGIO, MemoryInfoIO
from .io_dataset import IODataset


class MemoryDataset(IODataset):
    """Store sample dictionaries in memory."""

    def __init__(self, eeg_io=None, info_io=None, transform=None, label_transform=None):
        if eeg_io is None:
            eeg_io = MemoryEEGIO()
        if info_io is None:
            info_io = MemoryInfoIO()
        if not isinstance(eeg_io, MemoryEEGIO) or not isinstance(info_io, MemoryInfoIO):
            raise TypeError(
                'MemoryDataset requires MemoryEEGIO and MemoryInfoIO.')
        super().__init__(eeg_io, info_io, transform=transform, label_transform=label_transform)
