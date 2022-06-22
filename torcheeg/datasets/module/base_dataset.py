import os

from typing import Dict

from torch.utils.data import Dataset
from torcheeg.io import EEGSignalIO, MetaInfoIO


class BaseDataset(Dataset):
    channel_location_dict = {}
    adjacency_matrix = []

    def __init__(self, io_path: str):
        if not self.exist(io_path):
            raise RuntimeError('Database IO does not exist, please regenerate database IO.')
        self.io_path = io_path

        meta_info_io_path = os.path.join(self.io_path, 'info.csv')
        eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

        info_io = MetaInfoIO(meta_info_io_path)
        self.eeg_io = EEGSignalIO(eeg_signal_io_path)

        self.info = info_io.read_all()

    def exist(self, io_path: str) -> bool:
        meta_info_io_path = os.path.join(io_path, 'info.csv')
        eeg_signal_io_path = eeg_signal_io_path = os.path.join(io_path, 'eeg')

        return os.path.exists(meta_info_io_path) and os.path.exists(eeg_signal_io_path)

    def __getitem__(self, index: int) -> any:
        raise NotImplementedError("Method __getitem__ is not implemented in class " + self.__class__.__name__)

    def __len__(self):
        return len(self.info)

    def __copy__(self) -> 'BaseDataset':
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

        result.eeg_io = EEGSignalIO(eeg_signal_io_path)
        return result

    @property
    def repr_body(self) -> Dict:
        return {'io_path': self.io_path}

    @property
    def repr_tail(self) -> Dict:
        return {'length': self.__len__()}

    def __repr__(self) -> str:
        # init info
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            # line end
            if i:
                format_string += ','
            format_string += '\n'
            # str param
            if isinstance(v, str):
                format_string += f"    {k}='{v}'"
            else:
                format_string += f"    {k}={v}"
        format_string += '\n)'
        # other info
        format_string += '\n'
        for i, (k, v) in enumerate(self.repr_tail.items()):
            if i:
                format_string += ', '
            format_string += f'{k}={v}'
        return format_string