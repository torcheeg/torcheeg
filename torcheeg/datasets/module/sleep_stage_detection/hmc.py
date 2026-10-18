import os
import re
import warnings
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
    match = re.fullmatch(r'(SN\d{3})\.edf', recording_name)
    score_match = re.fullmatch(r'(SN\d{3})[_.]sleepscoring\.edf', scoring_name)
    if match is None or score_match is None or match.group(1) != score_match.group(1):
        raise ValueError('Expected an HMC recording and sleep scoring file with the same SN identifier.')
    return {'subject_id': match.group(1), 'record_id': match.group(1)}


def _validate_crop(start_at, end_at):
    for name, value in [('start_at', start_at), ('end_at', end_at)]:
        if name == 'end_at' and value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be a finite nonnegative time in seconds.')
    if end_at is not None and end_at <= start_at:
        raise ValueError('end_at must be greater than start_at.')


def crop(raw: mne.io.BaseRaw, start_at: float = 15 * 60,
         end_at: Union[None, float] = 15 * 60 + 6 * 60 * 60):
    _validate_crop(start_at, end_at)
    duration = raw.n_times / raw.info['sfreq']
    if start_at >= duration:
        raise ValueError('start_at must precede the end of the recording.')
    if end_at is not None and end_at > duration:
        warnings.warn('The recording ends before end_at; using its available samples.',
                      UserWarning, stacklevel=2)
    stop = min(end_at, duration) if end_at is not None else duration
    # end_at is exclusive, so exactly 30 seconds contains one complete epoch.
    return raw.crop(tmin=start_at, tmax=stop - 1. / raw.info['sfreq'])


def interploate(raw: mne.io.BaseRaw,
                channels: List = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2'],
                missing_channels: str = 'zero'):
    # Preserve the historical helper name and zero-fill default.
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


class HMCDataset(BaseDataset):
    r'''
    Haaglanden Medisch Centrum sleep staging database (HMC) is a widely-used sleep stage detection dataset. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Diego et al. 
    - Year: 2022
    - Download URL: https://physionet.org/content/hmc-sleep-staging/1.1/
    - Reference: Alvarez-Estevez D, Rijsman R M. Inter-database validation of a deep learning approach for automatic sleep scoring[J]. PloS one, 2021, 16(8): e0256111.
    - Signals: A collection of 151 whole-night polysomnographic (PSG) sleep recordings (85 Male, 66 Female, mean Age of 53.9 ± 15.4) collected during 2018 at the Haaglanden Medisch Centrum (HMC, The Netherlands) sleep center. The dataset contains electroencephalographic (EEG), electrooculographic (EOG), chin electromyographic (EMG), and electrocardiographic (ECG) activity ('ECG', 'EEG C3-M2', 'EEG C4-M1', 'EEG F4-M1', 'EEG O2-M1', 'EMG chin', 'EOG E1-M2', 'EOG E2-M2'). 
    - Rating: Sleep stages were annotated in 30 second contiguous intervals (Sleep stage W, Sleep stage N1, Sleep stage N2, Sleep stage N3, Sleep stage R, Lights off@@EEG F4-A1).

    In order to use this dataset, the following file structure is required:

    .. code-block:: text

        HMC/
        └── recordings/
            ├── SN001.edf
            ├── SN001_sleepscoring.edf
            ├── SN002.edf
            ├── SN002_sleepscoring.edf
            └── ...

    An example dataset:

    .. code-block:: python

        dataset = HMCDataset(root_path='./HMC/recordings',
                            sfreq=100,
                            channels=['EEG F4-M1', 'EEG C4-M1',
                                    'EEG O2-M1', 'EEG C3-M2'],
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
        # EEG signal (torch.Tensor[4, 3000]),
        # label (int)

    Files are paired by the released ``SNXXX`` recording identifier. Missing or
    duplicate pairs raise an error. Official ``SNXXX_sleepscoring.edf`` and the
    historical ``SNXXX.sleepscoring.edf`` spelling are accepted. Metadata retains
    the original recording's scored epoch times, including crop offsets and
    unscored gaps. See :doc:`/user_guide/hmc` for crop and cache migration notes.

    Args:
        root_path (str): Root path of the HMC dataset containing .edf and .sleepscoring.edf files. (default: :obj:`'./HMC/recordings'`)
        channels (list): List of EEG channels to use. Available channels are 'ECG', 'EEG C3-M2', 'EEG C4-M1', 'EEG F4-M1', 'EEG O2-M1', 'EMG chin', 'EOG E1-M2', 'EOG E2-M2'. (default: :obj:`['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']`)
        l_freq (float): Low cut-off frequency in Hz. (default: :obj:`0.5`)
        h_freq (float): High cut-off frequency in Hz. (default: :obj:`30`)
        sfreq (int): The sampling frequency to resample the signal to in Hz. (default: :obj:`100`)
        start_at (float): Start of the selected interval in seconds from the recording start. (default: :obj:`900`)
        end_at (float, optional): Exclusive interval end. None uses the available recording end. The historical default selects six hours after start_at=900. (default: :obj:`22500`)
        missing_channels (str): Historical zero-fill behavior ('zero'), or reject absent requested channels ('raise'). Channel order follows channels. (default: :obj:`'zero'`)
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
                 root_path: str = './HMC/recordings',
                 channels: List = ['EEG F4-M1',
                                   'EEG C4-M1',
                                   'EEG O2-M1',
                                   'EEG C3-M2'],
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
                 start_at: float = 15 * 60,
                 end_at: Union[None, float] = 15 * 60 + 6 * 60 * 60,
                 missing_channels: str = 'zero',
                 **kwargs):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        _validate_crop(start_at, end_at)
        if missing_channels not in ('zero', 'raise'):
            raise ValueError('missing_channels must be "zero" or "raise".')

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
            'verbose': verbose,
            'start_at': start_at,
            'end_at': end_at,
            'missing_channels': missing_channels
        }
        params.update(kwargs)
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def read_record(record: Tuple,
                    root_path: str = './HMC/recordings',
                    channels: List = ['EEG F4-M1',
                                      'EEG C4-M1',
                                      'EEG O2-M1',
                                      'EEG C3-M2'],
                    start_at: float = 15 * 60,
                    end_at: Union[None, float] = 15 * 60 + 6 * 60 * 60,
                    l_freq: float = 0.5,
                    h_freq: float = 30,
                    sfreq: int = 100, missing_channels: str = 'zero', **kwargs) -> Dict:
        _record_info(record)
        _validate_crop(start_at, end_at)
        recording_file, scoring_file = record
        stage_ids = {'Sleep stage W': 1, 'Sleep stage N1': 2,
                     'Sleep stage N2': 3, 'Sleep stage N3': 4, 'Sleep stage R': 5}
        with read_raw_edf(os.path.join(root_path, recording_file), preload=True) as raw:
            raw = interploate(raw, channels, missing_channels)
            raw = filter(raw, l_freq, h_freq)
            raw = downsample(raw, sfreq)
            origin = raw.first_samp
            raw.set_annotations(mne.read_annotations(os.path.join(root_path, scoring_file)),
                                emit_warning=False)
            # Form the scoring grid before crop can truncate an annotation.
            # Event codes identify stages even when lights markers overlap.
            events = []
            for annotation in raw.annotations:
                if annotation['description'] not in stage_ids:
                    continue
                count = int(np.floor((annotation['duration'] + 1e-8) / 30.))
                onsets = annotation['onset'] - raw.first_time + np.arange(count) * 30.
                samples = np.round(onsets * raw.info['sfreq']).astype(int) + origin
                events.extend((sample, 0, stage_ids[annotation['description']]) for sample in samples)
            if not events:
                raise ValueError('No complete scored 30-second epochs were found.')
            events = np.asarray(sorted(events), dtype=int)
            raw = crop(raw, start_at, end_at)
            epochs = mne.Epochs(raw, events, event_id=stage_ids, tmin=0.,
                                tmax=30. - 1. / raw.info['sfreq'], baseline=None,
                                preload=True, on_missing='ignore')
            epochs_data = epochs.get_data()
            if not len(epochs_data):
                raise ValueError('The selected interval contains no complete scored epochs.')
            id_to_stage = {value: key for key, value in stage_ids.items()}
            epochs_label = [id_to_stage[event] for event in epochs.events[:, 2]]
            epochs_onsets = (epochs.events[:, 0] - origin) / raw.info['sfreq']
        return {'epochs_data': epochs_data, 'epochs_label': epochs_label,
                'epochs_onsets': epochs_onsets}

    @staticmethod
    def fake_record(record: Tuple, **kwargs) -> Dict:
        num_epochs = 4
        num_channel = 4
        num_timepoint = 3000

        epochs_data = np.random.rand(num_epochs, num_channel, num_timepoint)
        epochs_label = ['Sleep stage W', 'Sleep stage N1',
                        'Sleep stage N2', 'Sleep stage N3']

        return {
            'record': ('SN001.edf', 'SN001_sleepscoring.edf'),
            'epochs_data': epochs_data,
            'epochs_label': epochs_label,
            'epochs_onsets': np.arange(num_epochs) * 30.,
        }

    @staticmethod
    def process_record(record: Tuple,
                       epochs_data: np.ndarray,
                       epochs_label: List,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       epochs_onsets: Union[None, np.ndarray] = None,
                       **kwargs):

        record_info = _record_info(record)
        if len(epochs_data) != len(epochs_label):
            raise ValueError('Epoch data and labels must have the same length.')
        if epochs_onsets is None:
            # Preserve direct calls with already contiguous historical epochs.
            epochs_onsets = np.arange(len(epochs_data)) * 30.
        epochs_onsets = np.asarray(epochs_onsets, dtype=float)
        if (epochs_onsets.shape != (len(epochs_data),)
                or not np.isfinite(epochs_onsets).all()
                or np.any(epochs_onsets < 0)
                or np.any(np.diff(epochs_onsets) < 30. - 1e-6)):
            raise ValueError('Epoch onsets must be finite, nonnegative and at least 30 seconds apart.')
        stages = {'Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R'}
        if any(label not in stages for label in epochs_label):
            raise ValueError('Epoch labels must contain scored sleep stages only.')
        if before_trial is not None:
            epochs_data = before_trial(epochs_data)
            if len(epochs_data) != len(epochs_label):
                raise ValueError('before_trial must preserve the number and order of epochs.')
        for i, (epoch_data, epoch_label, onset) in enumerate(zip(epochs_data, epochs_label, epochs_onsets)):
            if offline_transform is not None:
                epoch_data = offline_transform(eeg=epoch_data)['eeg']
            clip_id = f"{record_info['record_id']}_{i}"
            info = dict(record_info, clip_id=clip_id, label=epoch_label,
                        start_at=float(onset), end_at=float(onset + 30.),
                        trial_id=record_info['record_id'])
            yield {'eeg': epoch_data, 'key': clip_id, 'info': info}

    def set_records(self, root_path, **kwargs):
        recordings, scores = {}, {}
        for filename in sorted(os.listdir(root_path)):
            if not os.path.isfile(os.path.join(root_path, filename)) or not filename.endswith('.edf'):
                continue
            recording = re.fullmatch(r'(SN\d{3})\.edf', filename)
            scoring = re.fullmatch(r'(SN\d{3})[_.]sleepscoring\.edf', filename)
            if recording is None and scoring is None:
                raise ValueError(f'Unrecognized HMC EDF filename: {filename}.')
            files, key = (recordings, recording.group(1)) if recording else (scores, scoring.group(1))
            if key in files:
                raise ValueError(f'Ambiguous HMC files for {key}.')
            files[key] = filename
        if recordings.keys() != scores.keys():
            raise ValueError('Unpaired HMC files: missing scoring for '
                             f'{sorted(recordings.keys() - scores.keys())}; missing recording for '
                             f'{sorted(scores.keys() - recordings.keys())}.')
        return [(recordings[key], scores[key]) for key in sorted(recordings)]

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
                'start_at': self.start_at,
                'end_at': self.end_at,
                'missing_channels': self.missing_channels
            })
