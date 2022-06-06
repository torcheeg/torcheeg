import os
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

from torcheeg.io import EEGSignalIO, MetaInfoIO


class BaseDataset(Dataset, metaclass=ABCMeta):
    _repr_indent = 4

    def __init__(self, io_path: str):
        if not self.exist(io_path):
            raise RuntimeError(
                'Database IO does not exist, please regenerate database IO.')
        self.io_path = io_path

        meta_info_io_path = os.path.join(self.io_path, 'info.csv')
        eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

        info_io = MetaInfoIO(meta_info_io_path)
        self.eeg_io = EEGSignalIO(eeg_signal_io_path)

        self.info = info_io.read_all()

    def exist(self, io_path: str) -> bool:
        meta_info_io_path = os.path.join(io_path, 'info.csv')
        eeg_signal_io_path = eeg_signal_io_path = os.path.join(io_path, 'eeg')

        return os.path.exists(meta_info_io_path) and os.path.exists(
            eeg_signal_io_path)

    @abstractmethod
    def __getitem__(self, index: int) -> any:
        ...

    def __len__(self):
        return len(self.info)

    def __copy__(self) -> 'BaseDataset':
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

        result.eeg_io = EEGSignalIO(eeg_signal_io_path)
        return result

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Length: {}".format(self.__len__())]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)