from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import mne
import numpy as np

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


class FolderData(BaseData):
    """Read and preprocess records for :class:`FolderDataset`."""

    def __init__(self,
                 root_path: str = './folder',
                 structure: str = 'subject_in_label',
                 read_fn: Union[None, Callable] = default_read_fn,
                 transform: Union[None, Callable] = None,
                 **kwargs):

        super().__init__(
            transform=transform,
            root_path=root_path,
            structure=structure,
            read_fn=read_fn,
            **kwargs
        )

    def read_record(self,
                    record: Tuple,
                    read_fn: Union[None, Callable] = None,
                    **kwargs) -> Dict:
        if read_fn is None:
            read_fn = self.read_fn
        file_path, _, _ = record
        trial_samples = read_fn(file_path, **kwargs)

        return {
            'trial_samples': trial_samples,
        }

    def process_record(self, record: Tuple, trial_samples: mne.Epochs, **kwargs):
        _, subject_id, label = record
        events = [i[0] for i in trial_samples.events]
        events.append(events[-1] + np.diff(events)[0])
        for i, trial_signal in enumerate(trial_samples.get_data()):
            t_eeg = trial_signal
            record_info = {'subject_id': subject_id,
                           'start_at': events[i], 'end_at': events[i + 1], 'label': label}
            yield {'eeg': t_eeg, 'info': record_info}

    def set_records(self,
                    root_path: str = None,
                    structure: str = None,
                    **kwargs):
        if root_path is None:
            root_path = self.root_path
        if structure is None:
            structure = self.structure
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


class FolderDataset(CachedDataset):

    r"""
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

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import FolderDataset
        from torcheeg import transforms

        sfreq = 128  # Sampling rate
        n_channels = 14  # Number of channels
        duration = 5  # Data collected for 5 seconds
        for i in range(num_files):
            n_samples = sfreq * duration
            data = np.random.randn(n_channels, n_samples)

            ch_names = [f'ch_{i+1:03}' for i in range(n_channels)]
            ch_types = ['eeg'] * n_channels
            info = mne.create_info(ch_names, sfreq, ch_types)
            raw = mne.io.RawArray(data, info)

            file_name = f'sub{i+1}.fif'
            file_path = os.path.join('./root_folder/', file_name)
            raw.save(file_path)

        label_map = {'folder1': 0, 'folder2': 1}


        def transform_fn(eeg, baseline=None, **kwargs) -> dict:
            eeg_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            return {
                'eeg': eeg_transform(eeg),
                'baseline': eeg_transform(baseline) if baseline is not None else None,
            }


        def label_fn(info: dict, **kwargs) -> dict:
            label_transform = transforms.Compose([
                transforms.Select('label'),
                transforms.Lambda(lambda x: label_map[x]),
            ])
            return {
                'label': label_transform(info),
            }


        dataset = FolderDataset(
            root_path='./root_folder',
            structure='subject_in_label',
            num_channel=14,
            num_worker=4,
            online_transform=transform_fn,
            label_transform=label_fn,
        )

    Args:
        root_path (str): The path to the root folder. (default: :obj:`'./folder'`)
        structure (str): Folder structure, which affects how labels and subjects are mapped to EEG signal samples. Please refer to the above description of the structure of the two folders to select the correct parameters. (default: :obj:`'subject_in_label'`)
        read_fn (Callable): Method for reading files in a folder. By default, this class provides methods for reading files using :obj:`mne.io.read_raw`. At the same time, we allow users to pass in custom file reading methods. The first input parameter of whose is file_path, and other parameters are additional parameters passed in when the class is initialized. For example, you can pass :obj:`chunk_size=32` to :obj:`FolderDataset`, then :obj:`chunk_size` will be received here.



        offline_transform (Callable, optional): EEG transform applied before storage; returns a dictionary containing eeg. (default: :obj:`None`)
        online_transform (Callable, optional): EEG transform applied on each read. (default: :obj:`None`)
        label_transform (Callable, optional): Transform info into a dictionary containing label on each read. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = FolderData

    def __init__(self, root_path: str = './folder', structure: str = 'subject_in_label', read_fn: Union[None, Callable] = default_read_fn, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, label_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True, **kwargs):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'structure': structure, 'read_fn': read_fn, 'offline_transform': offline_transform, 'online_transform': online_transform, 'label_transform': label_transform,
                  'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        params.update(kwargs)
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel, 'offline_transform': self.offline_transform, 'online_transform': self.online_transform, 'label_transform': self.label_transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
