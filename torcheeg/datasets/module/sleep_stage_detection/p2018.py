import json
import os
import re
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import wfdb
from scipy import signal
from wfdb.processing import resample_sig

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


_STAGES = {'W': 'Sleep stage W', 'N1': 'Sleep stage N1',
           'N2': 'Sleep stage N2', 'N3': 'Sleep stage N3', 'R': 'Sleep stage R'}


def _record_info(record: Tuple) -> Dict:
    recording, scoring = map(Path, record)
    if (recording.suffix != '.mat' or scoring.suffix != '.arousal'
            or not re.fullmatch(r'(tr|te)\d{2}-\d{4}', recording.stem)
            or os.path.abspath(recording.with_suffix('')) != os.path.abspath(scoring.with_suffix(''))):
        raise ValueError('Expected matching P2018 signal and arousal files with one record identifier.')
    return {'subject_id': recording.stem, 'record_id': recording.stem}


def _settings(channels, sfreq, l_freq, h_freq):
    if (not isinstance(channels, (list, tuple)) or not channels
            or any(not isinstance(name, str) or not name for name in channels)
            or len(set(channels)) != len(channels)):
        raise ValueError('channels must contain unique channel names.')
    if isinstance(sfreq, (bool, np.bool_)) or not isinstance(sfreq, Real) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError('sfreq must be a finite positive sampling rate.')
    samples = round(30. * sfreq)
    if samples < 1 or not np.isclose(samples, 30. * sfreq, rtol=0, atol=1e-6):
        raise ValueError('sfreq must give an integer sample count in a 30-second epoch.')
    for name, value in [('l_freq', l_freq), ('h_freq', h_freq)]:
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 < value < sfreq / 2.):
            raise ValueError(f'{name} must be between zero and the output Nyquist frequency, or None.')
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError('l_freq must be below h_freq.')
    return int(samples)


def _stage_events(annotation, source_fs, source_samples):
    samples = np.asarray(annotation.sample)
    notes = annotation.aux_note
    if (not isinstance(notes, (list, tuple, np.ndarray))
            or samples.ndim != 1 or not np.issubdtype(samples.dtype, np.integer)
            or len(samples) != len(notes) or np.any(samples < 0)
            or np.any(samples > source_samples) or np.any(np.diff(samples) < 0)):
        raise ValueError('Annotation samples must be ordered integer positions within the recording.')
    events = []
    previous_sample, previous_stage = None, None
    for sample, raw_note in zip(samples, notes):
        if not isinstance(raw_note, str):
            raise ValueError('Annotation notes must be strings.')
        note = raw_note.strip().strip('\x00')
        if note in _STAGES:
            stage = _STAGES[note]
        elif note.casefold() in ('?', 'u', 'undefined', 'unknown', 'unscored'):
            stage = None
        elif not note or note.startswith('(') or note.endswith(')'):
            continue
        else:
            raise ValueError(f'Unknown sleep-stage annotation: {note}.')
        if previous_sample is not None and sample == previous_sample:
            if stage != previous_stage:
                raise ValueError('Conflicting sleep stages at the same sample.')
            continue
        previous_sample, previous_stage = int(sample), stage
        # Repeated same-stage markers do not split a continuously scored run.
        if not events or stage != events[-1][1]:
            events.append((int(sample), stage))
    if not events or not any(stage is not None for _, stage in events):
        raise ValueError('No scored sleep-stage annotations were found.')
    return np.asarray([sample / source_fs for sample, _ in events]), [stage for _, stage in events]


class P2018Dataset(BaseDataset):
    r'''
    The PhysioNet/Computing in Cardiology Challenge 2018 (P2018), is a widely-used sleep stage detection dataset. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Mohammad et al. 
    - Year: 2018
    - Download URL: https://physionet.org/content/challenge-2018/1.0.0/
    - Reference: Ghassemi M M, Moody B E, Lehman L W H, et al. You snooze, you win: the physionet/computing in cardiology challenge 2018[C]//2018 Computing in Cardiology Conference (CinC). IEEE, 2018, 45: 1-4.
    - Signals: 1,985 subjects which were monitored at an MGH sleep laboratory. The data were partitioned into balanced training (n = 994), and test sets (n = 989). The subjects had a variety of physiological signals recorded as they slept through the night including: electroencephalography (EEG), electrooculography (EOG), electromyography (EMG), electrocardiology (EKG), and oxygen saturation (SaO2). ('ABD', 'AIRFLOW', 'C3-M2', 'C4-M1', 'CHEST', 'Chin1-Chin2', 'E1-M2', 'ECG', 'F3-M2', 'F4-M1', 'O1-M2', 'O2-M1', 'SaO2').
    - Rating: W, N1, N2, N3, REM and undefined sleep stages in 30-second intervals. This class selects complete five-class scored epochs, separately from the challenge arousal-detection target.

    In order to use this dataset, the following file structure is required:

    .. code-block:: text

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
                                                   'Sleep stage R': 4})
                           ]),
                           online_transform=transforms.Compose([
                               transforms.MeanStdNormalize(),
                               transforms.ToTensor(),
                           ]),
                           )
        print(dataset[0])
        # EEG signal (torch.Tensor[6, 3000]),
        # label (int)

    Signal rates and physical units come from the WFDB header. EEG/EOG remain
    in the header's native units (normally microvolts), not MNE's volts.
    Resampling retains the WFDB FFT path. The eighth-order Butterworth filter
    uses second-order sections for numerical stability. Set both cutoffs to
    None to disable filtering. Each epoch has 30 * sfreq samples.

    Sleep-stage annotations define the scoring origin, not an earlier arousal
    event. Metadata preserves original times, including unscored leading data
    and gaps; mixed or undefined epochs are omitted. Use a new io_path after
    upgrading this reader. See :doc:`/user_guide/p2018` for the label/time and
    preprocessing protocol, units and cache migration.

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

        _settings(channels, sfreq, l_freq, h_freq)

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
            'record': ('tr03-0001/tr03-0001.mat', 'tr03-0001/tr03-0001.arousal'),
            'epochs_data': epochs_data,
            'epochs_label': epochs_label,
            'epochs_onsets': np.arange(num_epochs) * 30.,
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
        _record_info(record)
        epoch_samples = _settings(channels, sfreq, l_freq, h_freq)
        recording_file, scoring_file = record
        prefix = str(Path(recording_file).with_suffix(''))
        values, fields = wfdb.rdsamp(prefix, channel_names=list(channels))
        source_fs = fields['fs']
        if (isinstance(source_fs, (bool, np.bool_)) or not isinstance(source_fs, Real)
                or not np.isfinite(source_fs) or source_fs <= 0):
            raise ValueError('The WFDB header must supply a finite positive sampling rate.')
        values = np.asarray(values)
        returned = list(fields.get('sig_name', channels))
        if (values.ndim != 2 or values.shape[1] != len(returned) or not len(values)
                or len(set(returned)) != len(returned) or set(returned) != set(channels)
                or not np.isfinite(values).all()):
            raise ValueError('WFDB signals must be finite and match the requested channels.')
        order = [returned.index(name) for name in channels]
        values = values[:, order]
        units = fields.get('units')
        if units is not None and (not isinstance(units, (list, tuple, np.ndarray))
                or len(units) != len(returned)
                or any(unit is not None and not isinstance(unit, str) for unit in units)):
            raise ValueError('WFDB channel units must match the returned signals.')
        units = [units[index] for index in order] if units is not None else [None] * len(channels)
        annotation = wfdb.rdann(str(Path(scoring_file).with_suffix('')), 'arousal')
        annotation_fs = getattr(annotation, 'fs', None)
        if annotation_fs is not None and (isinstance(annotation_fs, (bool, np.bool_))
                or not isinstance(annotation_fs, Real) or not np.isfinite(annotation_fs)
                or not np.isclose(annotation_fs, source_fs, rtol=0, atol=1e-9)):
            raise ValueError('Annotation and signal sampling rates must match.')
        changes, stages = _stage_events(annotation, source_fs, len(values))
        # Retain the WFDB FFT resampling path, using the header rate. Labels
        # remain on the original annotation clock, separate from arousal events.
        values = np.column_stack([resample_sig(values[:, index], source_fs, sfreq)[0]
                                  for index in range(len(channels))])
        if l_freq is not None or h_freq is not None:
            cutoff = [l_freq, h_freq] if l_freq is not None and h_freq is not None else (l_freq or h_freq)
            kind = 'bandpass' if l_freq is not None and h_freq is not None else ('highpass' if l_freq is not None else 'lowpass')
            # SOS preserves the eighth-order Butterworth design while avoiding
            # poorly conditioned direct-form coefficients at low cutoffs.
            sos = signal.butter(8, cutoff, btype=kind, fs=sfreq, output='sos')
            values = signal.sosfiltfilt(sos, values, axis=0)
        if not np.isfinite(values).all():
            raise ValueError('Preprocessing produced nonfinite signal values.')
        first = int(np.ceil(changes[0] * sfreq - 1e-8))
        starts = np.arange(first, len(values) - epoch_samples + 1, epoch_samples, dtype=np.int64)
        times = starts / sfreq
        last_times = (starts + epoch_samples - 1) / sfreq
        intervals = np.searchsorted(changes, times + 1e-10, side='right') - 1
        ends = np.r_[changes[1:], np.inf]
        keep = np.asarray([stages[index] is not None and last < ends[index] - 1e-10
                           for index, last in zip(intervals, last_times)], dtype=bool)
        starts, times, intervals = starts[keep], times[keep], intervals[keep]
        if not len(starts):
            raise ValueError('No complete scored 30-second epochs were found.')
        epochs_data = np.stack([values[start:start + epoch_samples].T for start in starts])
        return {'epochs_data': epochs_data,
                'epochs_label': [stages[index] for index in intervals],
                'epochs_onsets': times, 'epochs_channels': list(channels), 'epochs_units': units}

    @staticmethod
    def process_record(record: Tuple,
                       epochs_data: np.ndarray,
                       epochs_label: List,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       epochs_onsets: Union[None, np.ndarray] = None,
                       epochs_channels: Union[None, List[str]] = None,
                       epochs_units: Union[None, List[Union[str, None]]] = None,
                       **kwargs):

        identity = _record_info(record)
        if np.ndim(epochs_data) != 3 or len(epochs_data) != len(epochs_label):
            raise ValueError('Epoch data must be [epochs, channels, samples] and align with labels.')
        if epochs_onsets is None:
            epochs_onsets = np.arange(len(epochs_data)) * 30.
        epochs_onsets = np.asarray(epochs_onsets, dtype=float)
        if (epochs_onsets.shape != (len(epochs_data),) or not np.isfinite(epochs_onsets).all()
                or np.any(epochs_onsets < 0) or np.any(np.diff(epochs_onsets) < 30. - 1e-6)):
            raise ValueError('Epoch onsets must be finite, nonnegative and at least 30 seconds apart.')
        if any(label not in _STAGES.values() for label in epochs_label):
            raise ValueError('Epoch labels must contain scored sleep stages only.')
        if epochs_channels is not None and (len(epochs_channels) != epochs_data.shape[1]
                or any(not isinstance(name, str) or not name for name in epochs_channels)
                or len(set(epochs_channels)) != len(epochs_channels)):
            raise ValueError('Channel names must identify each signal channel once.')
        if epochs_units is not None and (len(epochs_units) != epochs_data.shape[1]
                or any(unit is not None and not isinstance(unit, str) for unit in epochs_units)):
            raise ValueError('Channel units must identify each signal channel.')
        if before_trial is not None:
            epochs_data = before_trial(epochs_data)
            if len(epochs_data) != len(epochs_label):
                raise ValueError('before_trial must preserve epoch count and order.')
        for index, (data, label, onset) in enumerate(zip(epochs_data, epochs_label, epochs_onsets)):
            if offline_transform is not None:
                data = offline_transform(eeg=data)['eeg']
            clip_id = f"{identity['record_id']}_{index}"
            info = dict(identity, clip_id=clip_id, label=label, start_at=float(onset),
                        end_at=float(onset + 30.), trial_id=identity['record_id'])
            if epochs_channels is not None:
                info['channel_names'] = json.dumps(list(epochs_channels))
            if epochs_units is not None:
                info['channel_units'] = json.dumps(list(epochs_units))
            yield {'eeg': data, 'key': clip_id, 'info': info}

    def set_records(self, root_path, **kwargs):
        records = []
        for folder in sorted(Path(root_path).iterdir()):
            if not folder.is_dir() or folder.name.startswith('.'):
                continue
            if not re.fullmatch(r'(tr|te)\d{2}-\d{4}', folder.name):
                continue
            prefix = folder / folder.name
            required = [prefix.with_suffix(suffix) for suffix in ('.mat', '.hea', '.arousal')]
            if any(not path.is_file() for path in required):
                raise ValueError(f'Missing P2018 signal, header or scoring file for {folder.name}.')
            pair = (str(required[0]), str(required[2]))
            _record_info(pair)
            records.append(pair)
        return records

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
