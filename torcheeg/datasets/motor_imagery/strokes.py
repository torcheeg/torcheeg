import os
import re
from typing import Any, Callable, Dict, Tuple, Union

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset
from ..utils import get_grid_adjacency


class StrokePatientsMIData(BaseData):
    """Read and preprocess records for :class:`StrokePatientsMIDataset`."""

    channels = [
        'FP1', 'FP2', 'Fz',  'F3',  'F4',  'F7',  'F8',  'FCz',
        'FC3', 'FC4', 'FT7', 'FT8', 'Cz',  'C3',  'C4',  'T3',
        'T4',  'CPz', 'CP3', 'CP4', 'TP7', 'TP8', 'Pz',  'P3',
        'P4',  'T5',  'T6',  'Oz',  'O1',  'O2',
    ]

    grid = [
        ['-',   'FP1', '-',   'FP2', '-'],
        ['F7',  'F3',  'Fz',  'F4',  'F8'],
        ['FT7', 'FC3', 'FCz', 'FC4', 'FT8'],
        ['T3',  'C3',  'Cz',  'C4',  'T4'],
        ['TP7', 'CP3', 'CPz', 'CP4', 'TP8'],
        ['T5',  'P3',  'Pz',  'P4',  'T6'],
        ['-',   'O1',  'Oz',  'O2',  '-'],
    ]

    adjacency = get_grid_adjacency(grid)

    def __init__(self,
                 root_path='./StrokePatientsMIDataset',
                 chunk_size: int = 500,
                 overlap: int = 0,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size,
            overlap=overlap
        )

        self.subjects_info = pd.read_csv(os.path.join(root_path,
                                                      'participants.tsv'),
                                         sep='\t')
        self.electodes_info = pd.read_csv(os.path.join(
            root_path, "task-motor-imagery_electrodes.tsv"),
            sep='\t')
        electodes_info2 = pd.read_csv(os.path.join(
            root_path, "task-motor-imagery_channels.tsv"),
            sep='\t')
        self.electodes_info = pd.merge(self.electodes_info,
                                       electodes_info2,
                                       on='name',
                                       how='outer')
        refence = {
            'name': 'CPz',
            'type': 'EEG',
            'status': 'good',
            'status_description': 'refence'
        }

        insert_index = self.electodes_info.index[
            self.electodes_info.index.get_loc(17)]
        self.electodes_info = pd.concat([
            self.electodes_info.iloc[:insert_index],
            pd.DataFrame([refence], index=[insert_index]),
            self.electodes_info.iloc[insert_index:]
        ])
        self.electodes_info.index = range(len(self.electodes_info))

        self.events_info = pd.read_csv(os.path.join(
            root_path, 'task-motor-imagery_events.tsv'),
            sep='\t')

    def read_record(self, record: str, **kwargs) -> Dict:
        fdata = loadmat(record)
        X, Y = fdata['eeg'][0][
            0]  # X.shape = [40trials, 33channels,4000timepoints]
        Y = Y[:, 0]
        eeg = X[:, :30, :]

        return {
            'eeg': eeg,
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trial = 40
        num_channel = 33
        num_timepoint = 4000

        eeg = np.random.randn(num_trial, num_channel, num_timepoint)

        return {
            'record': 'sub-01_task-motor-imagery_eeg.mat',
            'eeg': eeg
        }

    def process_record(self, record: str, eeg: np.ndarray, chunk_size: int = None, overlap: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        subject_id = int(re.findall(
            'sub-(\\d\\d)_task-motor-imagery_eeg.mat', record)[0])
        for trial_id, eeg_trial in enumerate(eeg):
            eeg_baseline = eeg_trial[:, :1000]
            label = 1 if trial_id % 2 else 0
            if not chunk_size > overlap:
                raise ValueError(
                    f"Arg 'chunk_size' must be larger than arg 'overlap'.Current chunksize is {chunk_size},overlap is {overlap}")
            start = 1000
            step = chunk_size - overlap
            end = start + step
            end_time_point = 3000
            baseline_yield_dict = {'eeg': eeg_baseline}
            while end <= end_time_point:
                eeg_clip = eeg_trial[:, start:end]
                record_info = {'label': label,
                               'trial_id': trial_id, 'subject_id': subject_id}
                yield from self._annotate_info({'eeg': eeg_clip, 'info': record_info, 'baseline': eeg_baseline})
                start, end = (start + step, end + step)

    def set_records(self, root_path=None, **kwargs):
        if root_path is None:
            root_path = self.root_path
        subject_dir = os.path.join(root_path, 'sourcedata')
        return [
            os.path.join(os.path.join(subject_dir, sub),
                         os.listdir(os.path.join(subject_dir, sub))[0])
            for sub in os.listdir(subject_dir)
        ]

    def _merge_info(self, info):
        subjects_info = self.subjects_info.copy()
        subjects_info['subject_id'] = np.array(
            list(map(lambda x: int(x[-2:]), self.subjects_info['Participant_ID'].values)))
        return pd.merge(info, subjects_info, on='subject_id')

    def _annotate_info(self, item):
        # The old Dataset record helpers can be used without source files.
        if not hasattr(self, 'subjects_info'):
            yield item
            return
        metadata = self._merge_info(pd.DataFrame([item['info']]))
        if metadata.empty:
            return
        else:
            for info in metadata.to_dict('records'):
                yield dict(item, info=info)


class StrokePatientsMIDataset(CachedDataset):

    r"""
    An EEG motor imagery dataset for brain computer interface in acute stroke patients. For more detail, please refer to following information.

    - Author: Haijie Liu et al.
    - Year: 2024
    - Download URL: https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035/5
    - Reference: Liu, Haijie, et al. "An EEG motor imagery dataset for brain computer interface in acute stroke patients." Scientific Data 11.1 (2024): 131.
    - Stimulus: A video of gripping motion is played on the computer, which leads the patient imagine grabbing the ball. This video stays playing for 4 s. Patients only imagine one hand movement.
    - Signals: Electroencephalogram (30 channels at 500Hz sample rate,. Electrooculography (including horizontal and vertical EOG) (totally 50 participants).
    - label: left hand and right hand.

    In order to use this dataset, the downlowd root path is required, containing the following files and directories:

    .. code-block:: python

        StrokePatientsMIDataset/
        ├── sourcedata/
        ├── edffile/
        ├── task-motor-imagery_events.tsv
        └── ...

    An example:

    .. code-block:: python

        from torcheeg.transforms import Select, BandSignal


        def transform_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                BandSignal(sampling_rate=500, band_dict={'frequency_range': [8, 40]}),
            ])
            label_transform = transforms.Compose([
                Select('label'),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = StrokePatientsMIDataset(
            root_path='./StrokePatientsMIDataset',
            chunk_size=500,
            overlap=0,
            io_path='./stroke_dataset_cache',
            online_transform=transform_fn,
        )
        print(dataset[0]['eeg'].shape)  # EEG shape(1,30,500)
        print(dataset[0]['label'])  # label (int)

    Args:
        root_path (str): Downloaded data files (unzipped) dir.  (default: :obj:`'./StrokePatientsMIDataset'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`500`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)



        offline_transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        online_transform (Callable, optional): Sample transform applied on each read. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = StrokePatientsMIData

    channels = StrokePatientsMIData.channels

    grid = StrokePatientsMIData.grid

    adjacency = StrokePatientsMIData.adjacency

    def __init__(self, root_path='./StrokePatientsMIDataset', chunk_size: int = 500, overlap: int = 0, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'overlap': overlap, 'offline_transform': offline_transform, 'online_transform': online_transform,
                  'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)
        self.subjects_info = self.data.subjects_info
        self.electodes_info = self.data.electodes_info
        self.events_info = self.data.events_info

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'offline_transform': self.offline_transform, 'online_transform': self.online_transform, 'io_path': self.io_path, 'io_size': self.io_size, 'io_mode': self.io_mode, 'num_worker': self.num_worker, 'verbose': self.verbose})


class StrokePatientsMIProcessedData(StrokePatientsMIData):
    """Read and preprocess records for :class:`StrokePatientsMIProcessedDataset`."""

    channels = [
        'FP1', 'FP2', 'Fz',  'F3',  'F4',  'F7',  'F8',  'FCz',
        'FC3', 'FC4', 'FT7', 'FT8', 'Cz',  'C3',  'C4',  'T3',
        'T4',  'CPz', 'CP3', 'CP4', 'TP7', 'TP8', 'Pz',  'P3',
        'P4',  'T5',  'T6',  'Oz',  'O1',  'O2',
    ]

    grid = [
        ['-',   'FP1', '-',   'FP2', '-'],
        ['F7',  'F3',  'Fz',  'F4',  'F8'],
        ['FT7', 'FC3', 'FCz', 'FC4', 'FT8'],
        ['T3',  'C3',  'Cz',  'C4',  'T4'],
        ['TP7', 'CP3', 'CPz', 'CP4', 'TP8'],
        ['T5',  'P3',  'Pz',  'P4',  'T6'],
        ['-',   'O1',  'Oz',  'O2',  '-'],
    ]

    adjacency = get_grid_adjacency(grid)

    def __init__(self,
                 root_path='./StrokePatientsMIDataset',
                 chunk_size: int = 500,
                 overlap: int = 0,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size,
            overlap=overlap
        )

    def read_record(self, record: str, **kwargs) -> Dict:
        edf_reader = mne.io.read_raw_edf(record, preload=True)
        epochs = mne.make_fixed_length_epochs(edf_reader,
                                              duration=8,
                                              preload=True)
        data = epochs.get_data(
        )  # shape(40,33,4000) -num_trial, channels, T—duration（8s）

        eeg = data[:, :30, :]

        return {
            'eeg': eeg,
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trial = 40
        num_channel = 30
        num_timepoint = 4000

        eeg = np.random.randn(num_trial, num_channel, num_timepoint)

        return {
            'record': 'sub-01_task-motor-imagery_eeg.edf',
            'eeg': eeg
        }

    def process_record(self, record: str, eeg: np.ndarray, chunk_size: int = None, overlap: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        subject_id = int(re.findall(
            'sub-(\\d\\d)_task-motor-imagery_eeg.edf', record)[0])
        for trial_id, eeg_trial in enumerate(eeg):
            eeg_baseline = eeg_trial[:, :1000]
            label = 1 if trial_id % 2 else 0
            if not chunk_size > overlap:
                raise ValueError(
                    f"Arg 'chunk_size' must be larger than arg 'overlap'.Current chunksize is {chunk_size},overlap is {overlap}")
            start = 1000
            step = chunk_size - overlap
            end = start + step
            end_time_point = 3000
            yield_dict = {'eeg': eeg_baseline}
            while end <= end_time_point:
                eeg_clip = eeg_trial[:, start:end]
                record_info = {'label': label,
                               'trial_id': trial_id, 'subject_id': subject_id}
                yield from self._annotate_info({'eeg': eeg_clip, 'info': record_info, 'baseline': eeg_baseline})
                start, end = (start + step, end + step)

    def set_records(self, root_path=None, **kwargs):
        if root_path is None:
            root_path = self.root_path
        subject_dir = os.path.join(root_path, 'edffile')
        return [
            os.path.join(
                os.path.join(subject_dir, sub, 'eeg'),
                os.listdir(os.path.join(subject_dir, sub, 'eeg'))[0])
            for sub in os.listdir(subject_dir)
        ]


class StrokePatientsMIProcessedDataset(StrokePatientsMIDataset):

    r"""
    An EEG motor imagery dataset for brain computer interface in acute stroke patients. This version is the processed version provided from authors, in which data have undergone baseline removal and 8-40 bandpass operation. For more detail, please refer to following information.

    - Author: Haijie Liu et al.
    - Year: 2024
    - Download URL: https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035/5
    - Reference: Liu, Haijie, et al. "An EEG motor imagery dataset for brain computer interface in acute stroke patients." Scientific Data 11.1 (2024): 131.
    - Stimulus: A video of gripping motion is played on the computer, which leads the patient imagine grabbing the ball. This video stays playing for 4 s. Patients only imagine one hand movement.
    - Signals: Electroencephalogram (30 channels at 500Hz sample rate,. Electrooculography (including horizontal and vertical EOG) (totally 50 participants).
    - label: left hand and right hand.

    In order to use this dataset, the downlowd root path is required, containing the following files and directories:

    - sourcedata (dir)
    - edffile (dir)
    - task-motor-imagery_events.tsv
    - ...

    An example:

    .. code-block:: python

        from torcheeg.transforms import Select, BandSignal


        def transform_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                BandSignal(sampling_rate=500, band_dict={'frequency_range': [8, 40]}),
            ])
            label_transform = transforms.Compose([
                Select('label'),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = StrokePatientsMIProcessedDataset(
            root_path='your unzipped root path',
            chunk_size=500,
            overlap=0,
            io_path='./stroke_dataset_cache',
            online_transform=transform_fn,
        )
        print(dataset[0]['eeg'].shape)  # EEG shape(1,30,500)
        print(dataset[0]['label'])  # label (int)

    Args:
        root_path (str): Downloaded data files (unzipped) dir.  (default: :obj:`'./StrokePatientsMIDataset'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`500`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)



        offline_transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        online_transform (Callable, optional): Sample transform applied on each read. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = StrokePatientsMIProcessedData

    channels = StrokePatientsMIProcessedData.channels

    grid = StrokePatientsMIProcessedData.grid

    adjacency = StrokePatientsMIProcessedData.adjacency

    def __init__(self, root_path='./StrokePatientsMIDataset', chunk_size: int = 500, overlap: int = 0, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        super().__init__(root_path=root_path, chunk_size=chunk_size, overlap=overlap,
                         offline_transform=offline_transform, online_transform=online_transform,
                         io_path=io_path, io_size=io_size, io_mode=io_mode,
                         num_worker=num_worker, verbose=verbose)
