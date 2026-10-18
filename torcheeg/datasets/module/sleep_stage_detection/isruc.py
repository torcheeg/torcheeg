import json
import os
import re
from numbers import Integral
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np
from mne.io import read_raw_edf

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


_DEFAULT_CHANNELS = (('F3-A2', 'F3-M2'), ('C3-A2', 'C3-M2'),
                     ('O1-A2', 'O1-M2'), ('F4-A1', 'F4-M1'),
                     ('C4-A1', 'C4-M1'), ('O2-A1', 'O2-M1'))


def interploate(raw: mne.io.BaseRaw, channels: List = _DEFAULT_CHANNELS,
                missing_channels: str = 'zero', derive_channels: bool = True):
    # Keep the historical helper name. Select actual named channels; when
    # enabled, compute missing bipolar derivations from recorded components.
    if missing_channels not in ('zero', 'raise'):
        raise ValueError('missing_channels must be "zero" or "raise".')
    if not isinstance(channels, (list, tuple)) or not channels:
        raise ValueError('channels must be a nonempty list of names or candidate tuples.')
    if not isinstance(derive_channels, bool):
        raise ValueError('derive_channels must be a boolean.')
    original = set(raw.ch_names)
    selected = []
    for entry in channels:
        candidates = [entry] if isinstance(entry, str) else entry
        if (derive_channels and isinstance(candidates, (list, tuple))
                and all(isinstance(name, str) for name in candidates)
                and not any(name in raw.ch_names for name in candidates)):
            for name in candidates:
                pair = name.rsplit('-', 1)
                if name not in raw.ch_names and len(pair) == 2 and set(pair) <= original:
                    mne.set_bipolar_reference(raw, pair[0], pair[1], ch_name=name,
                                             drop_refs=False, copy=False)
        if isinstance(entry, str) and entry:
            if entry not in raw.ch_names and missing_channels == 'raise':
                raise ValueError(f'Missing required channel: {entry}.')
            selected.append(entry)
        elif (isinstance(entry, tuple) and entry
              and all(isinstance(name, str) and name for name in entry)
              and len(set(entry)) == len(entry)):
            matches = [name for name in entry if name in raw.ch_names]
            if len(matches) > 1 or (not matches and missing_channels == 'raise'):
                raise ValueError(f'Expected exactly one channel from {entry}; found {matches}.')
            selected.append(matches[0] if matches else entry[0])
        else:
            raise ValueError('Each channel must be a name or a nonempty unique tuple of names.')
    if len(selected) != len(set(selected)):
        raise ValueError('Selected channels must be unique.')
    absent = [name for name in selected if name not in raw.ch_names]
    if absent:
        info = mne.create_info(absent, raw.info['sfreq'], 'eeg')
        raw.add_channels([mne.io.RawArray(np.zeros((len(absent), raw.n_times)), info)], force_update_info=True)
    return raw.pick_channels(selected, ordered=True)


def filter(raw: mne.io.BaseRaw, l_freq: float = 0.5, h_freq: float = 30):
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    return raw


def downsample(epochs: mne.Epochs, sfreq: int = 100):
    epochs = epochs.resample(sfreq)
    return epochs


def epoching(raw, duration: float = 30):
    events = mne.make_fixed_length_events(
        raw, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration-1. / raw.info['sfreq'],
                        baseline=None, preload=True, proj=False)
    return epochs


def _record_info(record: Tuple, root_path: Union[None, str] = None) -> Dict:
    recording_file, scoring_file, group = record
    if isinstance(group, bool) or not isinstance(group, Integral) or group not in (0, 1, 2):
        raise ValueError('group must identify Subgroup_1, Subgroup_2 or Subgroup_3.')
    recording_file = os.path.abspath(recording_file)
    parts = (Path(os.path.relpath(recording_file, root_path)).parts if root_path is not None
             else Path(recording_file).parts[-(4 if group == 1 else 3):])
    expected_length = 4 if group == 1 else 3
    if len(parts) != expected_length or parts[0] != f'Subgroup_{group + 1}':
        raise ValueError('Record path must follow the selected ISRUC subgroup layout.')
    subject = parts[1]
    session = parts[2] if group == 1 else '1'
    stem = session if group == 1 else subject
    if (not re.fullmatch(r'\d+(?:_2)?', subject) or session not in ('1', '2')
            or parts[-1] != f'{stem}.rec'):
        raise ValueError('Record path has an invalid subject, session or recording filename.')
    score = re.fullmatch(re.escape(stem) + r'_([12])\.txt', os.path.basename(scoring_file))
    if score is None or os.path.dirname(os.path.abspath(scoring_file)) != os.path.dirname(recording_file):
        raise ValueError('Scoring and recording files must belong to the same ISRUC session.')
    subject_id = f'Subgroup_{group + 1}_{subject}'
    return {'subject_id': subject_id, 'session_id': int(session),
            'record_id': f'{subject_id}_night{session}', 'group_id': int(group),
            'scorer_id': int(score.group(1))}


def format_subject_id(recording_file: str, root_path: str):
    parts = Path(os.path.relpath(recording_file, root_path)).parts
    if not parts or not re.fullmatch(r'Subgroup_[123]', parts[0]):
        raise ValueError('Record must be inside an ISRUC subgroup.')
    score = os.path.splitext(recording_file)[0] + '_1.txt'
    return _record_info((recording_file, score, int(parts[0][-1]) - 1), root_path)['subject_id']


class ISRUCDataset(BaseDataset):
    r'''
    A polysomnographic (PSG) dataset named ISRUC-Sleep that was created aiming to help sleep researchers in their studies. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Khalighi et al. 
    - Year: 2016
    - Download URL: https://sleeptight.isr.uc.pt/
    - Reference: Khalighi S, Sousa T, Santos J M, et al. ISRUC-Sleep: A comprehensive public dataset for sleep researchers[J]. Computer methods and programs in biomedicine, 2016, 124: 180-192.
    - Signals: Three groups of data. Group 1's data contains 100 subjects, with one recording session per subject; Group 2's data contains 8 subjects, with two recording sessions per subject; Group 3's data contains 10 healthy subjects. PSG recordings include electrophysiological signals, pneumological signals, and another contextual information of the subjects ('C3-A2', 'C4-A1', 'DC3', 'DC8', 'F3-A2', 'F4-A1', 'LOC-A2', 'O1-A2', 'O2-A1', 'ROC-A1', 'SaO2', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8').
    - Rating: One numeric score per 30-second epoch: 0 (W), 1 (N1), 2 (N2), 3 (N3), and 5 (R), independently scored by two experts.

    In order to use this dataset, the following file structure is required:

    .. code-block:: text

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
                                                    'Sleep stage R': 4})
                            ]),
                            online_transform=transforms.Compose([
                                transforms.MeanStdNormalize(),
                                transforms.ToTensor(),
                            ]),
                            )
        print(dataset[0])
        # EEG signal (torch.Tensor[2, 3000]),
        # label (int)

    The default channel candidates select recorded A1/A2 or M1/M2 derivations
    without renaming their reference. If individual electrode and reference
    signals are present instead, derive_channels computes their difference.
    Actual selected, derived and zero-filled channels are saved in metadata.
    Epoch events are exactly 30 seconds apart; rejected intervals retain their
    original time and corresponding scorer label. Two nights of one Subgroup_2
    subject share subject_id and have distinct session/record/clip identifiers.
    Use a new io_path after upgrading this reader. See :doc:`/user_guide/isruc`.

    Args:
        root_path (str): Root path of the ISRUC dataset. (default: :obj:`'./ISRUC-SLEEP'`)
        groups (list): List of groups to include in the dataset. 0 for Subgroup_1, 1 for Subgroup_2, and 2 for Subgroup_3. (default: :obj:`[0, 1, 2]`)
        channels (list): Ordered channel names or tuples of alternative measured names. Defaults select six EEG positions with A1/A2 or M1/M2 names; these references are not renamed or treated as physically identical. Available channels are 'C3-A2', 'C4-A1', 'DC3', 'DC8', 'F3-A2', 'F4-A1', 'LOC-A2', 'O1-A2', 'O2-A1', 'ROC-A1', 'SaO2', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'.
        l_freq (float): Low cut-off frequency in Hz. (default: :obj:`0.5`)
        h_freq (float): High cut-off frequency in Hz. (default: :obj:`30`)
        sfreq (int): The sampling frequency to resample the signal to in Hz. (default: :obj:`100`)
        missing_channels (str): Zero-fill missing explicit channels ('zero') or reject them ('raise'). Missing and derived channels are recorded in metadata. (default: :obj:`'zero'`)
        derive_channels (bool): Compute an absent bipolar channel from its two recorded components when available. This performs an actual subtraction, not an alias rename. (default: :obj:`True`)
        scorer (int): Expert annotation file to use, 1 or 2. (default: :obj:`1`)
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
                 channels: List = _DEFAULT_CHANNELS,
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
                 scorer: int = 1,
                 derive_channels: bool = True,
                 **kwargs):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        if (not groups or any(isinstance(g, bool) or not isinstance(g, Integral) or g not in (0, 1, 2)
                              for g in groups) or len(set(groups)) != len(groups)):
            raise ValueError('groups must contain unique subgroup codes from [0, 1, 2].')
        if isinstance(scorer, bool) or not isinstance(scorer, Integral) or scorer not in (1, 2):
            raise ValueError('scorer must be expert 1 or 2.')
        if missing_channels not in ('zero', 'raise'):
            raise ValueError('missing_channels must be "zero" or "raise".')

        if not isinstance(derive_channels, bool):
            raise ValueError('derive_channels must be a boolean.')

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
            'verbose': verbose,
            'missing_channels': missing_channels,
            'scorer': scorer,
            'derive_channels': derive_channels
        }
        params.update(kwargs)
        self.__dict__.update(params)

        super().__init__(**params)

    @staticmethod
    def read_record(record: Tuple, channels: List = _DEFAULT_CHANNELS,
                    l_freq: float = 0.5, h_freq: float = 30, sfreq: int = 100,
                    missing_channels: str = 'zero', root_path: Union[None, str] = None,
                    derive_channels: bool = True,
                    **kwargs) -> Dict:
        _record_info(record, root_path)
        recording_file, scoring_file, group = record
        stage_map = {'0': 'Sleep stage W', '1': 'Sleep stage N1',
                     '2': 'Sleep stage N2', '3': 'Sleep stage N3', '5': 'Sleep stage R'}
        with open(scoring_file) as file:
            codes = file.read().split()
        if not codes or any(code not in stage_map for code in codes):
            raise ValueError('Scoring must contain only ISRUC stage codes 0, 1, 2, 3 and 5.')
        # MNE requires an EDF suffix. Use a portable owned temporary directory,
        # keeping the source recording intact and closing EDF before cleanup.
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, 'record.edf')
            shutil.copyfile(recording_file, path)
            with read_raw_edf(path, preload=True) as raw:
                original = set(raw.ch_names)
                raw = interploate(raw, channels, missing_channels, derive_channels)
                derived = {name: name.rsplit('-', 1) for name in raw.ch_names
                           if name not in original and derive_channels
                           and len(name.rsplit('-', 1)) == 2
                           and set(name.rsplit('-', 1)) <= original}
                filled = [name for name in raw.ch_names if name not in original and name not in derived]
                raw = filter(raw, l_freq, h_freq)
                raw = downsample(raw, sfreq)
                events = mne.make_fixed_length_events(raw, duration=30.)
                if len(events) != len(codes):
                    raise ValueError('Scoring count must match complete 30-second recording epochs.')
                epochs = epoching(raw)
                epochs_data = epochs.get_data()
                if not len(epochs_data):
                    raise ValueError('No complete scored epochs remain after rejection.')
                epochs_label = [stage_map[codes[index]] for index in epochs.selection]
                epochs_onsets = (epochs.events[:, 0] - raw.first_samp) / raw.info['sfreq']
                epoch_channels = list(raw.ch_names)
        return {'epochs_data': epochs_data, 'epochs_label': epochs_label,
                'epochs_onsets': epochs_onsets, 'epochs_channels': epoch_channels,
                'derived_channels': derived, 'zero_filled_channels': filled}

    @staticmethod
    def fake_record(record: Tuple, **kwargs) -> Dict:
        num_epochs = 4
        num_channel = 6
        num_timepoint = 3000

        epochs_data = np.random.rand(num_epochs, num_channel, num_timepoint)
        epochs_label = ['Sleep stage W', 'Sleep stage N1',
                        'Sleep stage N2', 'Sleep stage N3']

        return {
            'record': ('./ISRUC-SLEEP/Subgroup_1/1/1.rec', './ISRUC-SLEEP/Subgroup_1/1/1_1.txt', 0),
            'epochs_data': epochs_data,
            'epochs_label': epochs_label,
            'epochs_onsets': np.arange(num_epochs) * 30.,
            'epochs_channels': [pair[0] for pair in _DEFAULT_CHANNELS],
        }

    @staticmethod
    def process_record(record: Tuple,
                       epochs_data: np.ndarray,
                       epochs_label: List[str],
                       root_path: str = './ISRUC-SLEEP',
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       epochs_onsets: Union[None, np.ndarray] = None,
                       epochs_channels: Union[None, List[str]] = None,
                       derived_channels: Union[None, Dict] = None,
                       zero_filled_channels: Union[None, List[str]] = None,
                       **kwargs):

        identity = _record_info(record, root_path)
        if np.ndim(epochs_data) != 3 or len(epochs_data) != len(epochs_label):
            raise ValueError('Epoch data must be [epochs, channels, samples] and align with labels.')
        if epochs_onsets is None:
            epochs_onsets = np.arange(len(epochs_data)) * 30.
        epochs_onsets = np.asarray(epochs_onsets, dtype=float)
        if (epochs_onsets.shape != (len(epochs_data),) or not np.isfinite(epochs_onsets).all()
                or np.any(epochs_onsets < 0) or np.any(np.diff(epochs_onsets) < 30. - 1e-6)):
            raise ValueError('Epoch times must be finite, nonnegative and at least 30 seconds apart.')
        stages = {'Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R'}
        if any(label not in stages for label in epochs_label):
            raise ValueError('Epoch labels must be scored sleep stages.')
        if epochs_channels is not None and (len(epochs_channels) != epochs_data.shape[1]
                or any(not isinstance(name, str) or not name for name in epochs_channels)
                or len(set(epochs_channels)) != len(epochs_channels)):
            raise ValueError('Channel names must identify every signal channel exactly once.')
        if derived_channels is not None:
            if (not isinstance(derived_channels, dict) or epochs_channels is None
                    or set(derived_channels) - set(epochs_channels)
                    or any(not isinstance(pair, (tuple, list)) or len(pair) != 2
                           or any(not isinstance(name, str) or not name for name in pair)
                           for pair in derived_channels.values())):
                raise ValueError('Derived-channel metadata must identify selected channels and their two source names.')
        if zero_filled_channels is not None:
            if (not isinstance(zero_filled_channels, (tuple, list)) or epochs_channels is None
                    or any(not isinstance(name, str) for name in zero_filled_channels)
                    or len(set(zero_filled_channels)) != len(zero_filled_channels)
                    or set(zero_filled_channels) - set(epochs_channels)
                    or set(zero_filled_channels) & set(derived_channels or {})):
                raise ValueError('Zero-filled channels must be unique selected names and cannot be derived.')
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
            if derived_channels is not None:
                info['derived_channels'] = json.dumps(derived_channels, sort_keys=True)
            if zero_filled_channels is not None:
                info['zero_filled_channels'] = json.dumps(list(zero_filled_channels))
            yield {'eeg': data, 'key': clip_id, 'info': info}

    def set_records(self, root_path, **kwargs):
        records = []
        for group in sorted(self.groups):
            group_dir = Path(root_path, f'Subgroup_{group + 1}')
            for subject in sorted(group_dir.iterdir()):
                if not subject.is_dir() or subject.name.startswith('.'):
                    continue
                sessions = [subject / '1', subject / '2'] if group == 1 else [subject]
                for session in sessions:
                    stem = session.name
                    recording, scoring = session / f'{stem}.rec', session / f'{stem}_{self.scorer}.txt'
                    if not recording.is_file() or not scoring.is_file():
                        raise ValueError(f'Missing ISRUC recording/scoring pair: {recording}, {scoring}.')
                    if len(list(session.glob('*.rec'))) != 1:
                        raise ValueError(f'Ambiguous ISRUC recordings in {session}.')
                    record = (str(recording), str(scoring), int(group))
                    _record_info(record, root_path)
                    records.append(record)
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
                'io_size': self.io_size,
                'scorer': self.scorer,
                'missing_channels': self.missing_channels,
                'derive_channels': self.derive_channels
            })
