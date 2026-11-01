import os
from typing import Any, Callable, Dict, Tuple, Union

import mne
import numpy as np
import pandas as pd

from ..utils import get_random_dir_path
from .base_data import BaseData
from .cached_dataset import CachedDataset


def default_read_fn(file_path, **kwargs):
    # Load EEG file
    raw = mne.io.read_raw(file_path)
    # Convert raw to epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=1)
    # Return EEG data
    return epochs


class CSVFolderData(BaseData):
    """Read and preprocess records for :class:`CSVFolderDataset`."""

    def __init__(self,
                 csv_path: str = './data.csv',
                 read_fn: Union[None, Callable] = default_read_fn,
                 transform: Union[None, Callable] = None,
                 **kwargs):

        super().__init__(
            transform=transform,
            csv_path=csv_path,
            read_fn=read_fn,
            **kwargs
        )

    def read_record(self,
                    record: Dict,
                    read_fn: Union[None, Callable] = None,
                    **kwargs) -> Dict:
        if read_fn is None:
            read_fn = self.read_fn
        file_path = record['file_path']
        trial_samples = read_fn(file_path, **kwargs)

        return {
            'trial_samples': trial_samples,
        }

    def process_record(self, record: Dict, trial_samples: mne.Epochs, **kwargs):
        file_path = record['file_path']
        events = [i[0] for i in trial_samples.events]
        events.append(events[-1] + np.diff(events)[0])
        for i, trial_signal in enumerate(trial_samples.get_data()):
            t_eeg = trial_signal
            file_name = os.path.basename(file_path)
            record_info = {
                **record, 'start_at': events[i], 'end_at': events[i + 1]}
            yield {'eeg': t_eeg, 'info': record_info}

    def set_records(self, csv_path: str = None, **kwargs):
        if csv_path is None:
            csv_path = self.csv_path
        df = pd.read_csv(csv_path)

        if not ('file_path' in df.columns):
            raise ValueError('file_path is required in csv file.')

        # df to a list of dict, each dict is a row
        df_list = df.to_dict('records')

        return df_list


class CSVFolderDataset(CachedDataset):

    r"""
    Read meta information from CSV file and read EEG data from folder according to the meta information. The CSV file should contain the following columns:

    - ``subject_id`` (Optional): The subject id of the EEG data. Commonly used in training and testing dataset split.
    - ``label`` (Optional): The label of the EEG data. Commonly used in training and testing dataset split.
    - ``file_path`` (Required): The path to the EEG data file.

    .. code-block:: python

        # data.csv
        # | subject_id | trial_id | label | file_path                 |
        # | ---------- | -------  | ----- | ------------------------- |
        # | sub1       | 0        | 0     | './data/label1/sub1.fif' |
        # | sub1       | 1        | 1     | './data/label2/sub1.fif' |
        # | sub1       | 2        | 2     | './data/label3/sub1.fif' |
        # | sub2       | 0        | 0     | './data/label1/sub2.fif' |
        # | sub2       | 1        | 1     | './data/label2/sub2.fif' |
        # | sub2       | 2        | 2     | './data/label3/sub2.fif' |

        from torcheeg.datasets import CSVFolderDataset
        from torcheeg import transforms

        def default_read_fn(file_path, **kwargs):
            # Load EEG file
            raw = mne.io.read_raw(file_path)
            # Convert raw to epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=1)
            # Return EEG data
            return epochs

        def prepare_training_1(sample):
            sample = dict(sample)
            sample['eeg'] = transforms.ToTensor()(sample['eeg'])
            return {'eeg': sample['eeg'], 'labels': transforms.Select('label')(sample['info'])}

        dataset = CSVFolderDataset(csv_path='./data.csv', read_fn=default_read_fn, num_worker=4).with_transform(prepare_training_1)

    Args:
        csv_path (str): The path to the CSV file.
        read_fn (Callable): Method for reading files in a folder. By default, this class provides methods for reading files using :obj:`mne.io.read_raw`. At the same time, we allow users to pass in custom file reading methods. The first input parameter of whose is file_path, and other parameters are additional parameters passed in when the class is initialized. For example, you can pass :obj:`chunk_size=32` to :obj:`FolderDataset`, then :obj:`chunk_size` will be received here.



        transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = CSVFolderData

    def __init__(self, csv_path: str = './data.csv', read_fn: Union[None, Callable] = default_read_fn, transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True, **kwargs):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'csv_path': csv_path, 'read_fn': read_fn, 'transform': transform, 'io_path': io_path,
                  'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        params.update(kwargs)
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'csv_path': self.csv_path, 'read_fn': self.read_fn, 'transform': self.transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
