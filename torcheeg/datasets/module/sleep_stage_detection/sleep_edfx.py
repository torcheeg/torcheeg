import os
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np
from mne.io import read_raw_edf

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


def list_records(root_path):
    recording_files = []
    scoring_files = []

    files = os.listdir(root_path)

    for file in files:
        if 'PSG' in file:
            recording_files.append(os.path.join(root_path, file))
        if 'Hypnogram' in file:
            scoring_files.append(os.path.join(root_path, file))

    recording_files.sort()
    scoring_files.sort()

    recording_scoring_pairs = []
    for recording_file, scoring_file in zip(recording_files, scoring_files):
        if recording_file[:6] == scoring_file[:6]:
            recording_scoring_pairs.append((recording_file, scoring_file))

    return recording_scoring_pairs


def interploate(raw: mne.io.BaseRaw, channels: List = ['EEG Fpz-Cz', 'EEG Pz-Oz']):
    missing_channels = list(set(channels) - set(raw.ch_names))
    if missing_channels:
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


class SleepEDFxDataset(BaseDataset):
    '''
    Sleep-EDF Database Expanded (sleep-edfx), is a widely-used sleep stage detection dataset. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Kemp et al.
    - Year: 2000
    - Download URL: https://www.physionet.org/content/sleep-edfx/1.0.0/
    - Reference: Kemp B, Zwinderman A H, Tuk B, et al. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG[J]. IEEE Transactions on Biomedical Engineering, 2000, 47(9): 1185-1194.
    - Signals: 197 whole-night PolySomnoGraphic sleep recordings, containing EEG, EOG, chin EMG, and event markers ('EEG Fpz-Cz', 'EEG Pz-Oz', 'EMG submental', 'EOG horizontal', 'Marker').
    - Rating: Sleep stages were annotated in 30 second contiguous intervals (Sleep stage W, Sleep stage N1, Sleep stage N2, Sleep stage N3, Sleep stage R, Lights off@@EEG F4-A1).

    In order to use this dataset, the following file structure is required:

    .. code-block:: python

        sleep-edf-database-expanded-1.0.0/
        ├── sleep-telemetry/
        │   ├── ST7222JA-Hypnogram.edf
        │   ├── ST7061J0-PSG.edf
        │   ├── ST7041JO-Hypnogram.edf
        │   └── ST7021J0-PSG.edf
        └── sleep-telemetry/
            └── ...

    An example dataset:

    .. code-block:: python

        dataset = SleepEDFxDataset(root_path='./sleep-edf-database-expanded-1.0.0/', sfreq=100,
                               channels=['EEG Fpz-Cz', 'EEG Pz-Oz'],
                               label_transform=transforms.Compose([
                                   transforms.Select('label'),
                                   transforms.Mapping({'Sleep stage W': 0,
                                                       'Sleep stage N1': 1,
                                                       'Sleep stage N2': 2,
                                                       'Sleep stage N3': 3,
                                                       'Sleep stage R': 4,
                                                       'Lights off@@EEG F4-A1': 0})
                               ]),
                               online_transform=transforms.ToTensor(),
                               )
        print(dataset[0])
        # EEG signal (torch.Tensor[6, 3000]),
        # label (int)

    Args:
        root_path (str): Root path of the Sleep-EDF Database Expanded dataset. (default: :obj:`'./sleep-edf-database-expanded-1.0.0/'`)
        studies (list): List of study types to include, can be 'cassette' and/or 'telemetry'. (default: :obj:`['cassette', 'telemetry']`)
        channels (list): List of EEG channels to use. (default: :obj:`['EEG Fpz-Cz', 'EEG Pz-Oz']`)
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
                 root_path: str = './sleep-edf-database-expanded-1.0.0/',
                 studies: List = ['cassette', 'telemetry'],
                 channels: List = ['EEG Fpz-Cz',
                                   'EEG Pz-Oz'],
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

        assert 'casstte' in studies or 'telemetry' in studies, 'studies must contain either "cassette" or "telemetry"'

        params = {
            'root_path': root_path,
            'studies': studies,
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
    def read_record(record: Tuple,
                    channels: List = ['EEG Fpz-Cz',
                                      'EEG Pz-Oz'],
                    l_freq: float = 0.5,
                    h_freq: float = 30,
                    sfreq: int = 100, **kwargs) -> Dict:
        recording_file, scoring_file = record

        raw = read_raw_edf(recording_file, preload=True)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        raw = filter(raw, l_freq, h_freq)
        raw = downsample(raw, sfreq)
        raw = interploate(raw, channels)

        annotation = mne.read_annotations(scoring_file)
        raw.set_annotations(annotation, emit_warning=False)
        events, event_id = mne.events_from_annotations(
            raw, chunk_duration=30.)

        if 'Sleep stage ?' in event_id.keys():
            event_id.pop('Sleep stage ?')
        if 'Movement time' in event_id.keys():
            event_id.pop('Movement time')

        tmax = 30. - 1. / raw.info['sfreq']
        epochs = mne.Epochs(raw=raw, events=events,
                            event_id=event_id, tmin=0., tmax=tmax, baseline=None)

        epochs_data = epochs.get_data()
        epochs_label = []
        for epoch_annotation in epochs.get_annotations_per_epoch():
            epochs_label.append(epoch_annotation[0][2])

        return {
            'epochs_data': epochs_data,
            'epochs_label': epochs_label,
        }

    @staticmethod
    def fake_record(record: Tuple, **kwargs) -> Dict:
        num_epochs = 4
        num_channel = 2
        num_timepoint = 3000

        epochs_data = np.random.rand(num_epochs, num_channel, num_timepoint)
        epochs_label = ['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3']

        return {
            'record': ('fake_record', 'fake_record'),
            'epochs_data': epochs_data,
            'epochs_label': epochs_label,
        }

    @staticmethod
    def process_record(record: Tuple,
                       epochs_data: np.ndarray,
                       epochs_label: List[str],
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        recording_file, scoring_file = record

        subject_id = recording_file.split('-')[0]

        if before_trial:
            epochs_data = before_trial(epochs_data)

        label2id = {'Sleep stage W': 'Sleep stage W',
                    'Sleep stage 1': 'Sleep stage N1',
                    'Sleep stage 2': 'Sleep stage N2',
                    'Sleep stage 3': 'Sleep stage N3',
                    'Sleep stage 4': 'Sleep stage N3',
                    'Sleep stage R': 'Sleep stage R'}

        for i, (epoch_data, epoch_label) in enumerate(zip(epochs_data, epochs_label)):
            if offline_transform is not None:
                epoch_data = offline_transform(eeg=epoch_data)['eeg']

            clip_id = f"{subject_id}_{i}"

            record_info = {
                'clip_id': clip_id,
                'label': label2id[epoch_label],
                'start_at': i * 30,
                'end_at': (i + 1) * 30,
                'subject_id': subject_id,
                'trial_id': 0
            }

            yield {'eeg': epoch_data, 'key': clip_id, 'info': record_info}

    def set_records(self, root_path, **kwargs):
        recording_scoring_pairs = []

        if 'cassette' in self.studies:
            recording_scoring_pairs += list_records(
                os.path.join(root_path, 'sleep-cassette'))

        if 'telemetry' in self.studies:
            recording_scoring_pairs += list_records(
                os.path.join(root_path, 'sleep-telemetry'))

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
