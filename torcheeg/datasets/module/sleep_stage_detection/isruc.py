import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np
from mne.io import read_raw_edf

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


def interploate(raw: mne.io.BaseRaw, channels: List = ['F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1']):
    missing_channels = list(set(channels) - set(raw.ch_names))

    if len(missing_channels):
        info = mne.create_info(missing_channels, raw.info['sfreq'], 'eeg')
        zero_data = np.zeros((len(missing_channels), len(raw.times)))
        raw_missing = mne.io.RawArray(zero_data, info)
        raw.add_channels([raw_missing], force_update_info=True)

    raw = raw.pick_channels(channels)
    return raw


def filter(raw: mne.io.BaseRaw, l_freq: float = 0.5, h_freq: float = 30):
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    return raw


def downsample(epochs: mne.Epochs, sfreq: int = 100):
    epochs = epochs.resample(sfreq)
    return epochs


def epoching(raw, duration: float = 30):
    events = mne.make_fixed_length_events(
        raw, duration=duration-1. / raw.info['sfreq'])
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration-1. / raw.info['sfreq'],
                        baseline=None, preload=True, proj=False)
    return epochs


def format_subject_id(recording_file: str, root_path: str):
    recording_file = os.path.normpath(recording_file)
    root_path = os.path.normpath(root_path)
    rel_path = os.path.relpath(recording_file, root_path)
    path_parts = rel_path.split(os.sep)
    subject_parts = path_parts[:2]
    subject_id = "_".join(subject_parts)
    return subject_id


class ISRUCDataset(BaseDataset):
    r'''
    A polysomnographic (PSG) dataset named ISRUC-Sleep that was created aiming to help sleep researchers in their studies. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:
    
    - Author: Khalighi et al. 
    - Year: 2016
    - Download URL: https://sleeptight.isr.uc.pt/
    - Reference: Khalighi S, Sousa T, Santos J M, et al. ISRUC-Sleep: A comprehensive public dataset for sleep researchers[J]. Computer methods and programs in biomedicine, 2016, 124: 180-192.
    - Signals: Three groups of data. Group 1's data contains 100 subjects, with one recording session per subject; Group 2's data contains 8 subjects, with two recording sessions per subject; Group 3's data contains 10 healthy subjects. PSG recordings include electrophysiological signals, pneumological signals, and another contextual information of the subjects ('C3-A2', 'C4-A1', 'DC3', 'DC8', 'F3-A2', 'F4-A1', 'LOC-A2', 'O1-A2', 'O2-A1', 'ROC-A1', 'SaO2', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8').
    - Rating: Sleep stages were annotated in 30 second contiguous intervals (Sleep stage W, Sleep stage N1, Sleep stage N2, Sleep stage N3, Sleep stage R, Lights off@@EEG F4-A1).

    In order to use this dataset, the following file structure is required:
    
    .. code-block:: python
    
        ISRUC-SLEEP/
        ├── Subgroup_1/
        │   ├── 1/
        │   │   ├── 1.rec
        │   │   └── 1_1.txt
        │   └── ...
        ├── Subgroup_2/
        │   └── ...
        └── Subgroup_3/
            └── ...
    
    An example dataset:

    .. code-block:: python

        dataset = ISRUCDataset(root_path='./ISRUC-SLEEP',
                            sfreq=100,
                            channels=['F3-M2', 'C3-M2', 'O1-M2',
                                        'F4-M1', 'C4-M1', 'O2-M1'],
                            label_transform=transforms.Compose([
                                transforms.Select('label'),
                                transforms.Mapping({'Sleep stage W': 0,
                                                    'Sleep stage N1': 1,
                                                    'Sleep stage N2': 2,
                                                    'Sleep stage N3': 3,
                                                    'Sleep stage R': 4,
                                                    'Lights off@@EEG F4-A1': 0})
                            ]),
                            online_transform=transforms.Compose([
                                transforms.MeanStdNormalize(),
                                transforms.ToTensor(),
                            ]),
                            )
        print(dataset[0])
        # EEG signal (torch.Tensor[6, 3000]),
        # label (int)

    Args:
        root_path (str): Root path of the ISRUC dataset. (default: :obj:`'./ISRUC-SLEEP'`)
        groups (list): List of groups to include in the dataset. 0 for Subgroup_1, 1 for Subgroup_2, and 2 for Subgroup_3. (default: :obj:`[0, 1, 2]`)
        channels (list): List of EEG channels to use. Available channels are 'C3-A2', 'C4-A1', 'DC3', 'DC8', 'F3-A2', 'F4-A1', 'LOC-A2', 'O1-A2', 'O2-A1', 'ROC-A1', 'SaO2', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'. (default: :obj:`['F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1']`)
        l_freq (float): Low cut-off frequency in Hz. (default: :obj:`0.5`)
        h_freq (float): High cut-off frequency in Hz. (default: :obj:`30`)
        sfreq (int): The sampling frequency to resample the signal to in Hz. (default: :obj:`100`)
        online_transform (Callable, optional): The transformation of the EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the second value of each element in the dataset. (default: :obj:`None`)
        io_path (str, optional): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    '''
    def __init__(self,
                 root_path: str = './ISRUC-SLEEP',
                 groups: List = [0, 1, 2],
                 channels: List = ['F3-M2',
                                   'C3-M2',
                                   'O1-M2',
                                   'F4-M1',
                                   'C4-M1',
                                   'O2-M1'],
                 l_freq: float = 0.5,
                 h_freq: float = 30,
                 sfreq: int = 100,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 **kwargs):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        assert 0 in groups or 1 in groups or 2 in groups, "groups must be a subset of [0, 1, 2]"

        params = {
            'root_path': root_path,
            'groups': groups,
            'channels': channels,
            'l_freq': l_freq,
            'h_freq': h_freq,
            'sfreq': sfreq,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }
        params.update(kwargs)
        self.__dict__.update(params)

        super().__init__(**params)

    @staticmethod
    def process_record(file: Any = None,
                       root_path: str = './ISRUC-SLEEP',
                       channels: List = ['F3-M2',
                                         'C3-M2',
                                         'O1-M2',
                                         'F4-M1',
                                         'C4-M1',
                                         'O2-M1'],
                       l_freq: float = 0.5,
                       h_freq: float = 30,
                       sfreq: int = 100,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        recording_file, scoring_file, group = file
        subject_id = format_subject_id(recording_file, root_path)

        raw = None
        with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
            temp_edf_path = os.path.join(temp_dir, 'temp.edf')
            shutil.copy2(recording_file, temp_edf_path)
            raw = read_raw_edf(temp_edf_path, preload=True)

        raw = filter(raw, l_freq, h_freq)
        raw = downsample(raw, sfreq)
        raw = interploate(raw, channels)
        epochs = epoching(raw)

        epochs_data = epochs.get_data()
        label2id = {'0': 'Sleep stage W',
                    '1': 'Sleep stage N1',
                    '2': 'Sleep stage N2',
                    '3': 'Sleep stage N3',
                    '5': 'Sleep stage R', }

        epochs_label = []
        for line in open(scoring_file).readlines():
            line_str = line.strip()
            if line_str != '':
                epochs_label.append(label2id[line_str])

        if before_trial:
            epochs_data = before_trial(epochs_data)

        for i, (epoch_data, epoch_label) in enumerate(zip(epochs_data, epochs_label)):
            if offline_transform is not None:
                epoch_data = offline_transform(eeg=epoch_data)['eeg']

            clip_id = f"{subject_id}_{i}"

            record_info = {
                'clip_id': clip_id,
                'label': epoch_label,
                'start_at': i * 30,
                'end_at': (i + 1) * 30,
                'subject_id': subject_id,
                'trial_id': 0,
                'group_id': group
            }

            yield {'eeg': epoch_data, 'key': clip_id, 'info': record_info}

    def set_records(self, root_path, **kwargs):
        recording_files = []
        scoring_files = []
        groups = []

        if 0 in self.groups:
            for sub_dir in os.listdir(os.path.join(os.path.join(root_path, 'Subgroup_1'))):
                recording_files.append(os.path.join(
                    root_path, 'Subgroup_1', sub_dir, f'{sub_dir}.rec'))
                scoring_files.append(os.path.join(
                    root_path, 'Subgroup_1', sub_dir, f'{sub_dir}_1.txt'))
                groups.append(0)

        if 1 in self.groups:
            for sub_dir in os.listdir(os.path.join(os.path.join(root_path, 'Subgroup_2'))):
                recording_files.append(os.path.join(
                    root_path, 'Subgroup_2', sub_dir, f'1/1.rec'))
                scoring_files.append(os.path.join(
                    root_path, 'Subgroup_2', sub_dir, f'1/1_1.txt'))
                groups.append(1)

                recording_files.append(os.path.join(
                    root_path, 'Subgroup_2', sub_dir, f'2/2.rec'))
                scoring_files.append(os.path.join(
                    root_path, 'Subgroup_2', sub_dir, f'2/2_1.txt'))
                groups.append(1)

        if 2 in self.groups:
            for sub_dir in os.listdir(os.path.join(os.path.join(root_path, 'Subgroup_3'))):
                recording_files.append(os.path.join(
                    root_path, 'Subgroup_3', sub_dir, f'{sub_dir}.rec'))
                scoring_files.append(os.path.join(
                    root_path, 'Subgroup_3', sub_dir, f'{sub_dir}_1.txt'))
                groups.append(2)

        recording_scoring_pairs = []
        for recording_file, scoring_file, group in zip(recording_files, scoring_files, groups):
            if recording_file[:-4] == scoring_file[:-6]:
                recording_scoring_pairs.append(
                    (recording_file, scoring_file, group))
        return recording_scoring_pairs

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
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
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
