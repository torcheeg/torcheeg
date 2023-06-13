import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import mne
import numpy as np

from torcheeg.io import EEGSignalIO, MetaInfoIO

from .base_dataset import BaseDataset


def default_read_fn(file_path, **kwargs):
    # Load EEG file
    raw = mne.io.read_raw(file_path)
    # Convert raw to epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=1)
    # Return EEG data
    return epochs


class FolderDataset(BaseDataset):
    '''
    Read EEG samples and their corresponding labels from a fixed folder structure. This class allows two kinds of common file structures, :obj:`subject_in_label` and :obj:`label_in_subject`. Here, :obj:`subject_in_label` corresponds to the following file structure:

    .. code-block:: python

        tree
        # outputs
        label01
        |- sub01.edf
        |- sub02.edf
        label02
        |- sub01.edf
        |- sub02.edf

    And :obj:`label_in_subject` corresponds to the following file structure:

    .. code-block:: python

        tree
        # outputs
        sub01
        |- label01.edf
        |- label02.edf
        sub02
        |- label01.edf
        |- label02.edf

    Args:
        root_path (str): The path to the root folder. (default: :obj:`'./folder'`)
        structure (str): Folder structure, which affects how labels and subjects are mapped to EEG signal samples. Please refer to the above description of the structure of the two folders to select the correct parameters. (default: :obj:`'subject_in_label'`)
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
                 root_path: str = './folder',
                 structure: str = 'subject_in_label',
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
            'root_path': root_path,
            'structure': structure,
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
    def _load_data(file: Any = None,
                   offline_transform: Union[None, Callable] = None,
                   read_fn: Union[None, Callable] = None,
                   **kwargs):

        file_path, subject_id, label = file

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

            clip_id = f'{subject_id}_{label}_{write_pointer}'
            write_pointer += 1

            record_info = {
                'subject_id': subject_id,
                'trial_id': i,
                'start_at': events[i],
                'end_at': events[i + 1],
                'clip_id': clip_id,
                'label': label
            }

            yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

    @staticmethod
    def _set_files(root_path: str = './folder',
                   structure: str = 'subject_in_label',
                   **kwargs):
        # get all the subfolders
        subfolders = [str(i) for i in Path(root_path).iterdir() if i.is_dir()]
        # get all the files in the subfolders
        file_path_list = []
        for subfolder in subfolders:
            file_path_list += [
                str(i) for i in Path(subfolder).iterdir() if i.is_file()
            ]
        # get the subject id
        if structure == 'subject_in_label':
            # get the file name without the extension
            subjects = [i.split('/')[-1].split('.')[0] for i in file_path_list]
            labels = [i.split('/')[-2] for i in file_path_list]
        elif structure == 'label_in_subject':
            subjects = [i.split('/')[-2] for i in file_path_list]
            labels = [i.split('/')[-1].split('.')[0] for i in file_path_list]
        else:
            raise ValueError('Unknown folder mode: {}'.format(structure))

        file_path_subject_label = list(zip(file_path_list, subjects, labels))
        return file_path_subject_label

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg = self.read_eeg(eeg_index)

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
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
