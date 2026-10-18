import os
import re
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np
from mne.io import read_raw_edf

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


def _record_info(record: Tuple) -> Dict:
    recording_file, scoring_file = record
    recording_name = os.path.basename(recording_file)
    scoring_name = os.path.basename(scoring_file)
    recording_match = re.fullmatch(r'((SC4|ST7)\d{2})([12])[A-Z0-9]{2}-PSG\.edf',
                                  recording_name)
    scoring_match = re.fullmatch(r'(SC4|ST7)\d{2}[12][A-Z0-9]{2}-Hypnogram\.edf',
                                scoring_name)
    if recording_match is None or scoring_match is None:
        raise ValueError('Expected official Sleep-EDF PSG and Hypnogram filenames.')
    if recording_name[:6] != scoring_name[:6]:
        raise ValueError('PSG and Hypnogram must belong to the same subject and night.')
    return {
        'subject_id': recording_match.group(1),
        'session_id': int(recording_match.group(3)),
        'record_id': recording_name[:-8],
    }


def list_records(root_path):
    recording_files = {}
    scoring_files = {}
    for filename in sorted(os.listdir(root_path)):
        path = os.path.join(root_path, filename)
        if not os.path.isfile(path):
            continue
        if filename.endswith('-PSG.edf'):
            files = recording_files
        elif filename.endswith('-Hypnogram.edf'):
            files = scoring_files
        else:
            continue
        key = filename[:6]
        if key in files:
            raise ValueError(f'Ambiguous Sleep-EDF files for subject/night {key}.')
        files[key] = path

    if recording_files.keys() != scoring_files.keys():
        missing_scores = sorted(recording_files.keys() - scoring_files.keys())
        missing_recordings = sorted(scoring_files.keys() - recording_files.keys())
        raise ValueError('Unpaired Sleep-EDF files: missing Hypnogram for '
                         f'{missing_scores}; missing PSG for {missing_recordings}.')
    pairs = [(recording_files[key], scoring_files[key])
             for key in sorted(recording_files)]
    for pair in pairs:
        _record_info(pair)
    return pairs


def interploate(raw: mne.io.BaseRaw,
                channels: List = ['EEG Fpz-Cz', 'EEG Pz-Oz'],
                missing_channels: str = 'zero'):
    # Keep the historical helper name and zero-fill default for compatibility.
    if missing_channels not in ('zero', 'raise'):
        raise ValueError('missing_channels must be "zero" or "raise".')
    if not channels or len(channels) != len(set(channels)):
        raise ValueError('channels must contain unique channel names.')
    absent_channels = [channel for channel in channels if channel not in raw.ch_names]
    if absent_channels:
        if missing_channels == 'raise':
            raise ValueError(f'Missing required channels: {absent_channels}.')
        info = mne.create_info(absent_channels, raw.info['sfreq'], 'eeg')
        zero_data = np.zeros((len(absent_channels), len(raw.times)))
        raw_missing = mne.io.RawArray(zero_data, info)
        raw.add_channels([raw_missing], force_update_info=True)
    return raw.pick_channels(channels, ordered=True)


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

    .. code-block:: text

        sleep-edf-database-expanded-1.0.0/
        ├── sleep-telemetry/
        │   ├── ST7021J0-PSG.edf
        │   └── ST7021JM-Hypnogram.edf
        └── sleep-cassette/
            ├── SC4001E0-PSG.edf
            ├── SC4001EC-Hypnogram.edf
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
        # EEG signal (torch.Tensor[2, 3000]),
        # label (int)

    Records are paired by the first six filename characters (subject and night).
    Missing or ambiguous pairs raise an error. Metadata uses the first five
    characters for ``subject_id`` (both nights share a subject), the night number
    for ``session_id``, and the eight-character PSG identifier for ``record_id``
    and ``trial_id``. ``start_at`` and ``end_at`` are seconds from the recording
    start, including gaps between scored epochs. Sequence workflows must check
    these times and record boundaries before grouping epochs.

    Use a new ``io_path`` when upgrading from the previous reader. Old caches
    contain different subject/clip identifiers and fabricated contiguous times.
    Keep old caches for reproducibility. See :doc:`/user_guide/sleepedf`.

    Args:
        root_path (str): Root path of the Sleep-EDF Database Expanded dataset. (default: :obj:`'./sleep-edf-database-expanded-1.0.0/'`)
        studies (list): List of study types to include, can be 'cassette' and/or 'telemetry'. (default: :obj:`['cassette', 'telemetry']`)
        channels (list): List of EEG channels to use. (default: :obj:`['EEG Fpz-Cz', 'EEG Pz-Oz']`)
        l_freq (float): Low cut-off frequency in Hz. (default: :obj:`0.5`)
        h_freq (float): High cut-off frequency in Hz. (default: :obj:`30`)
        sfreq (int): The sampling frequency to resample the signal to in Hz. (default: :obj:`100`)
        missing_channels (str): Use ``'zero'`` for the historical zero-fill behavior or ``'raise'`` to reject absent requested channels. Channel order follows ``channels``. (default: :obj:`'zero'`)
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
                 missing_channels: str = 'zero',
                 **kwargs):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        if not studies or len(studies) != len(set(studies)) or set(studies) - {'cassette', 'telemetry'}:
            raise ValueError('studies must contain unique cassette and/or telemetry entries.')
        if missing_channels not in ('zero', 'raise'):
            raise ValueError('missing_channels must be "zero" or "raise".')

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
            'verbose': verbose,
            'missing_channels': missing_channels
        }
        params.update(kwargs)
        self.__dict__.update(params)

        super().__init__(**params)

    @staticmethod
    def read_record(record: Tuple,
                    channels: List = ['EEG Fpz-Cz', 'EEG Pz-Oz'],
                    l_freq: float = 0.5,
                    h_freq: float = 30,
                    sfreq: int = 100,
                    missing_channels: str = 'zero', **kwargs) -> Dict:
        _record_info(record)
        recording_file, scoring_file = record
        # Unique event IDs distinguish source stages 3 and 4 until label mapping.
        stage_ids = {'Sleep stage W': 1, 'Sleep stage 1': 2,
                     'Sleep stage 2': 3, 'Sleep stage 3': 4,
                     'Sleep stage 4': 5, 'Sleep stage R': 6}
        with read_raw_edf(recording_file, preload=True) as raw:
            raw = interploate(raw, channels, missing_channels)
            raw = filter(raw, l_freq, h_freq)
            raw = downsample(raw, sfreq)
            raw.set_annotations(mne.read_annotations(scoring_file), emit_warning=False)
            # Attached annotation onsets include first_time. Convert explicitly
            # so resampled or cropped recordings retain their sample origin.
            events = []
            for annotation in raw.annotations:
                if annotation['description'] not in stage_ids:
                    continue
                count = int(np.floor((annotation['duration'] + 1e-8) / 30.))
                onsets = annotation['onset'] - raw.first_time + np.arange(count) * 30.
                samples = np.round(onsets * raw.info['sfreq']).astype(int) + raw.first_samp
                events.extend((sample, 0, stage_ids[annotation['description']])
                              for sample in samples)
            if not events:
                raise ValueError('No complete scored 30-second epochs were found.')
            events = np.asarray(sorted(events), dtype=int)
            tmax = 30. - 1. / raw.info['sfreq']
            epochs = mne.Epochs(raw, events, event_id=stage_ids, tmin=0., tmax=tmax,
                                baseline=None, preload=True, on_missing='ignore')
            epochs_data = epochs.get_data()
            id_to_stage = {value: key for key, value in stage_ids.items()}
            epochs_label = [id_to_stage[event] for event in epochs.events[:, 2]]
            # Retain gaps caused by unscored, movement or rejected intervals.
            epochs_onsets = (epochs.events[:, 0] - raw.first_samp) / raw.info['sfreq']
        return {'epochs_data': epochs_data, 'epochs_label': epochs_label,
                'epochs_onsets': epochs_onsets}

    @staticmethod
    def fake_record(record: Tuple, **kwargs) -> Dict:
        epochs_data = np.random.rand(4, 2, 3000)
        return {
            'record': ('SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf'),
            'epochs_data': epochs_data,
            'epochs_label': ['Sleep stage W', 'Sleep stage 1',
                             'Sleep stage 2', 'Sleep stage 3'],
            'epochs_onsets': np.arange(4) * 30.,
        }

    @staticmethod
    def process_record(record: Tuple,
                       epochs_data: np.ndarray,
                       epochs_label: List[str],
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       epochs_onsets: Union[None, np.ndarray] = None,
                       **kwargs):
        record_info = _record_info(record)
        if len(epochs_data) != len(epochs_label):
            raise ValueError('Epoch data and labels must have the same length.')
        if epochs_onsets is None:
            # Historical direct process_record calls can still supply contiguous epochs.
            epochs_onsets = np.arange(len(epochs_data)) * 30.
        epochs_onsets = np.asarray(epochs_onsets, dtype=float)
        if (epochs_onsets.shape != (len(epochs_data),)
                or not np.isfinite(epochs_onsets).all()
                or np.any(epochs_onsets < 0)
                or np.any(np.diff(epochs_onsets) < 30. - 1e-6)):
            raise ValueError('Epoch onsets must be finite, nonnegative and at least 30 seconds apart.')
        label_map = {'Sleep stage W': 'Sleep stage W',
                     'Sleep stage 1': 'Sleep stage N1',
                     'Sleep stage 2': 'Sleep stage N2',
                     'Sleep stage 3': 'Sleep stage N3',
                     'Sleep stage 4': 'Sleep stage N3',
                     'Sleep stage R': 'Sleep stage R',
                     'Sleep stage N1': 'Sleep stage N1',
                     'Sleep stage N2': 'Sleep stage N2',
                     'Sleep stage N3': 'Sleep stage N3'}
        if any(label not in label_map for label in epochs_label):
            raise ValueError('Epoch labels must contain scored sleep stages only.')
        if before_trial is not None:
            epochs_data = before_trial(epochs_data)
            if len(epochs_data) != len(epochs_label):
                raise ValueError('before_trial must preserve the number and order of epochs.')
        for i, (epoch_data, epoch_label, onset) in enumerate(
                zip(epochs_data, epochs_label, epochs_onsets)):
            if offline_transform is not None:
                epoch_data = offline_transform(eeg=epoch_data)['eeg']
            clip_id = f"{record_info['record_id']}_{i}"
            info = dict(record_info, clip_id=clip_id, label=label_map[epoch_label],
                        start_at=float(onset), end_at=float(onset + 30.),
                        trial_id=record_info['record_id'])
            yield {'eeg': epoch_data, 'key': clip_id, 'info': info}

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
                'io_size': self.io_size,
                'missing_channels': self.missing_channels
            })
