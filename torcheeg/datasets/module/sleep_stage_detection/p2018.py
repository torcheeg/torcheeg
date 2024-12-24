import os
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import wfdb
from scipy import signal
from wfdb.processing import resample_multichan

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


class P2018Dataset(BaseDataset):
    r'''
    The PhysioNet/Computing in Cardiology Challenge 2018 (P2018), is a widely-used sleep stage detection dataset. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Mohammad et al. 
    - Year: 2018
    - Download URL: https://physionet.org/content/challenge-2018/1.0.0/
    - Reference: Ghassemi M M, Moody B E, Lehman L W H, et al. You snooze, you win: the physionet/computing in cardiology challenge 2018[C]//2018 Computing in Cardiology Conference (CinC). IEEE, 2018, 45: 1-4.
    - Signals: 1,985 subjects which were monitored at an MGH sleep laboratory. The data were partitioned into balanced training (n = 994), and test sets (n = 989). The subjects had a variety of physiological signals recorded as they slept through the night including: electroencephalography (EEG), electrooculography (EOG), electromyography (EMG), electrocardiology (EKG), and oxygen saturation (SaO2). ('ABD', 'AIRFLOW', 'C3-M2', 'C4-M1', 'CHEST', 'Chin1-Chin2', 'E1-M2', 'ECG', 'F3-M2', 'F4-M1', 'O1-M2', 'O2-M1', 'SaO2').
    - Rating: Sleep stages were annotated in 30 second contiguous intervals (Sleep stage W, Sleep stage N1, Sleep stage N2, Sleep stage N3, Sleep stage R, Lights off@@EEG F4-A1).

    In order to use this dataset, the following file structure is required:

    .. code-block:: python

        P2018/
        └── training/
            ├── tr03-0005
            ├── tr03-0029
            ├── tr03-0052
            ├── tr03-0061
            └── ...

    An example dataset:

    .. code-block:: python

        dataset = P2018Dataset(root_path='./P2018/training/', sfreq=100, 
                           channels=['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'],
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
        root_path (str): Root path of the P2018 dataset. (default: :obj:`'./P2018/training/'`)
        channels (list): List of EEG channels to use. Available channels are 'ABD', 'AIRFLOW', 'C3-M2', 'C4-M1', 'CHEST', 'Chin1-Chin2', 'E1-M2', 'ECG', 'F3-M2', 'F4-M1', 'O1-M2', 'O2-M1', 'SaO2'. (default: :obj:`['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']`)
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
                 root_path: str = './P2018/training/',
                 channels: List = ['F3-M2',
                                   'F4-M1',
                                   'C3-M2',
                                   'C4-M1',
                                   'O1-M2',
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

        params = {
            'root_path': root_path,
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
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def fake_record(record: Tuple, **kwargs) -> Dict:
        num_epochs = 4
        num_channel = 6
        num_timepoint = 3000

        epochs_data = np.random.rand(num_epochs, num_channel, num_timepoint)
        epochs_label = ['Sleep stage W', 'Sleep stage N1',
                        'Sleep stage N2', 'Sleep stage N3']

        return {
            'record': ('fake_record', 'fake_record'),
            'epochs_data': epochs_data,
            'epochs_label': epochs_label,
        }

    @staticmethod
    def read_record(record: Tuple,
                    channels: List = ['F3-M2',
                                      'F4-M1',
                                      'C3-M2',
                                      'C4-M1',
                                      'O1-M2',
                                      'O2-M1'],
                    l_freq: float = 0.5,
                    h_freq: float = 30,
                    sfreq: int = 100, **kwargs) -> Dict:
        recording_file, scoring_file = record

        epochs_data, _ = wfdb.rdsamp(
            recording_file[:-4], channel_names=channels)
        ann = wfdb.rdann(scoring_file[:-8], scoring_file[-7:])

        epochs_data, ann = resample_multichan(epochs_data, ann, 200, sfreq)

        nyq = 0.5 * sfreq
        low = l_freq / nyq
        high = h_freq / nyq

        b, a = signal.butter(8, [low, high], 'bandpass')
        epochs_data = signal.filtfilt(b, a, epochs_data, axis=0)

        epochs_data = epochs_data[ann.sample[0]:, :]
        temp = epochs_data.shape[0] % 3000
        if temp != 0:
            epochs_data = epochs_data[:-temp]
        epochs_num = epochs_data.shape[0] // 3000

        epochs_data = epochs_data.reshape(-1, 3000, len(channels))
        epochs_data = epochs_data.transpose(0, 2, 1)

        label2id = {'W': 'Sleep stage W',
                    'N1': 'Sleep stage N1',
                    'N2': 'Sleep stage N2',
                    'N3': 'Sleep stage N3',
                    'R': 'Sleep stage R'}

        ann_labels = []
        start = 0
        for i, label in enumerate(ann.aux_note):
            if label in label2id.keys():
                if start == 0:
                    start = ann.sample[i]
                ann_labels.append((ann.sample[i]-start, label))

        epochs_label = []
        begin = 0
        end = 0
        for k in range(len(ann_labels)-1):
            begin = int(ann_labels[k][0]) // 3000
            end = int(ann_labels[k+1][0]) // 3000
            for i in range(begin, end):
                epochs_label.append(label2id[ann_labels[k][1]])

        for i in range(end, epochs_num):
            epochs_label.append(label2id[ann_labels[-1][1]])

        return {
            'epochs_data': epochs_data,
            'epochs_label': epochs_label,
        }

    @staticmethod
    def process_record(record: Tuple,
                       epochs_data: np.ndarray,
                       epochs_label: List,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        recording_file, scoring_file = record

        subject_id = os.path.basename(os.path.dirname(recording_file))

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
                'trial_id': 0
            }

            yield {'eeg': epoch_data, 'key': clip_id, 'info': record_info}

    def set_records(self, root_path, **kwargs):
        subjects_id = os.listdir(root_path)
        subjects_id.sort()

        recording_scoring_pairs = []
        for subject_id in subjects_id:
            if not os.path.isdir(os.path.join(root_path, subject_id)):
                continue

            recording_scoring_pairs.append(
                (os.path.join(root_path, subject_id, f'{subject_id}.mat'),
                 os.path.join(root_path, subject_id, f'{subject_id}.arousal'))
            )

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
