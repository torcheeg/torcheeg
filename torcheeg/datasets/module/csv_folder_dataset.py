from typing import Any, Callable, Dict, Tuple, Union
import pandas as pd
import mne
import numpy as np

from .base_dataset import BaseDataset


def default_read_fn(file_path, **kwargs):
    # Load EEG file
    raw = mne.io.read_raw(file_path)
    # Convert raw to epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=1)
    # Return EEG data
    return epochs


class CSVFolderDataset(BaseDataset):
    '''
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

        def default_read_fn(file_path, **kwargs):
            # Load EEG file
            raw = mne.io.read_raw(file_path)
            # Convert raw to epochs
            epochs = mne.make_fixed_length_epochs(raw, duration=1)
            # Return EEG data
            return epochs

        dataset = CSVFolderDataset(csv_path='./data.csv',
                                   read_fn=default_read_fn,
                                   online_transform=transforms.ToTensor(),
                                   label_transform=transforms.Select('label'),
                                   num_worker=4)

    Args:
        csv_path (str): The path to the CSV file.
        read_fn (Callable): Method for reading files in a folder. By default, this class provides methods for reading files using :obj:`mne.io.read_raw`. At the same time, we allow users to pass in custom file reading methods. The first input parameter of whose is file_path, and other parameters are additional parameters passed in when the class is initialized. For example, you can pass :obj:`chunk_size=32` to :obj:`FolderDataset`, then :obj:`chunk_size` will be received here.
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/deap`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)
    '''
    def __init__(self,
                 csv_path: str = './data.csv',
                 read_fn: Union[None, Callable] = default_read_fn,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/folder',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False,
                 **kwargs):
        # pass all arguments to super class
        params = {
            'csv_path': csv_path,
            'read_fn': read_fn,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose,
            'in_memory': in_memory
        }
        params.update(kwargs)
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def process_record(file: Any = None,
                       offline_transform: Union[None, Callable] = None,
                       read_fn: Union[None, Callable] = None,
                       **kwargs):

        trial_info = file
        file_path = trial_info['file_path']

        trial_samples = read_fn(file_path, **kwargs)
        events = [i[0] for i in trial_samples.events]
        events.append(
            events[-1] +
            np.diff(events)[0])  # time interval between all events are same

        write_pointer = 0
        for i, trial_signal in enumerate(trial_samples.get_data()):
            t_eeg = trial_signal
            if not offline_transform is None:
                t = offline_transform(eeg=trial_signal)
                t_eeg = t['eeg']

            clip_id = f'{file_path}_{write_pointer}'
            write_pointer += 1

            record_info = {
                **trial_info,
                'start_at': events[i],
                'end_at': events[i + 1],
                'clip_id': clip_id
            }

            yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

    def set_records(self, csv_path: str = './data.csv', **kwargs):
        # read csv
        df = pd.read_csv(csv_path)

        assert 'file_path' in df.columns, 'file_path is required in csv file.'

        # df to a list of dict, each dict is a row
        df_list = df.to_dict('records')

        return df_list

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        signal = eeg
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=eeg)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'csv_path': self.csv_path,
                'read_fn': self.read_fn,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
