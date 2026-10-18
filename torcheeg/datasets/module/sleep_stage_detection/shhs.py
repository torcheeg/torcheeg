import json
import re
import xml.etree.ElementTree as ET
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


_STAGES = {0: 'Sleep stage W', 1: 'Sleep stage N1', 2: 'Sleep stage N2',
           3: 'Sleep stage N3', 4: 'Sleep stage N3', 5: 'Sleep stage R',
           6: None, 9: None}


def _record_info(record):
    recording, scoring = map(Path, record)
    match = re.fullmatch(r'shhs([12])-(\d{6})', recording.stem)
    if (match is None or recording.suffix != '.edf'
            or scoring.name != recording.stem + '-nsrr.xml'):
        raise ValueError('Expected one SHHS EDF and its matching NSRR XML file.')
    return {'subject_id': match.group(2), 'session_id': int(match.group(1)),
            'record_id': recording.stem}


def _validate_visits(visits):
    if (not isinstance(visits, (list, tuple)) or not visits
            or any(type(visit) is not int or visit not in (1, 2) for visit in visits)
            or len(set(visits)) != len(visits)):
        raise ValueError('visits must contain unique visit numbers 1 and/or 2.')


def _settings(channels, sfreq, l_freq, h_freq, start_at, end_at):
    if (not isinstance(channels, (list, tuple)) or not channels
            or any(not isinstance(name, str) or not name for name in channels)
            or len(set(channels)) != len(channels)):
        raise ValueError('channels must contain unique channel names.')
    if (isinstance(sfreq, (bool, np.bool_)) or not isinstance(sfreq, Real)
            or not np.isfinite(sfreq) or sfreq <= 0
            or not np.isclose(30. * sfreq, round(30. * sfreq), rtol=0, atol=1e-6)):
        raise ValueError('sfreq must be positive and give an integer 30-second sample count.')
    if round(30. * sfreq) < 1:
        raise ValueError('sfreq must give at least one sample per epoch.')
    for name, value in [('l_freq', l_freq), ('h_freq', h_freq)]:
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 < value < sfreq / 2.):
            raise ValueError(f'{name} must be inside the output Nyquist frequency, or None.')
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError('l_freq must be below h_freq.')
    for name, value in [('start_at', start_at), ('end_at', end_at)]:
        if value is None and name == 'end_at':
            continue
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative.')
    if end_at is not None and end_at <= start_at:
        raise ValueError('end_at must be after start_at.')
    return int(round(30. * sfreq))


def _scored_epochs(path):
    root = ET.parse(path).getroot()
    if root.tag != 'PSGAnnotation' or root.findtext('EpochLength') is None:
        raise ValueError('Expected NSRR PSGAnnotation XML with EpochLength.')
    if float(root.findtext('EpochLength')) != 30.:
        raise ValueError('Only 30-second NSRR sleep-stage epochs are supported.')
    intervals = []
    for event in root.findall('./ScoredEvents/ScoredEvent'):
        if event.findtext('EventType') != 'Stages|Stages':
            continue
        try:
            code = int(event.findtext('EventConcept').rsplit('|', 1)[1])
            start = float(event.findtext('Start'))
            duration = float(event.findtext('Duration'))
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise ValueError('Invalid NSRR sleep-stage event.') from error
        if (code not in _STAGES or not np.isfinite(start) or start < 0
                or not np.isfinite(duration) or duration <= 0
                or not np.isclose(start / 30., round(start / 30.), rtol=0, atol=1e-8)
                or not np.isclose(duration / 30., round(duration / 30.), rtol=0, atol=1e-8)):
            raise ValueError('Sleep-stage events need known codes and complete 30-second grid intervals.')
        intervals.append((start, start + duration, code))
    intervals.sort()
    epochs = []
    previous_end = 0.
    for start, end, code in intervals:
        if start < previous_end - 1e-8:
            raise ValueError('Overlapping sleep-stage intervals are ambiguous.')
        previous_end = end
        if _STAGES[code] is not None:
            epochs.extend((start + 30. * i, _STAGES[code])
                          for i in range(int(round((end - start) / 30.))))
    if not epochs:
        raise ValueError('No complete scored sleep-stage epochs were found.')
    return epochs


class SHHSDataset(BaseDataset):
    r'''Sleep Heart Health Study (SHHS) sleep-stage signals and NSRR annotations.

    Each sample is one complete 30-second epoch. Stages 3 and 4 map to N3;
    movement (6) and unscored (9) intervals are omitted. Original recording
    times and gaps are retained. The same participant identifier is used
    across visits; record and session identifiers distinguish the visits.

    - Download URL: https://sleepdata.org/datasets/shhs
    - Reference: Quan S F, Howard B V, Iber C, et al. The Sleep Heart Health
      Study: design, rationale, and methods. Sleep, 1997, 20(12): 1077-1085.
    - Signals: the requested recorded EDF channels, in volts for EEG/EOG.
    - Labels: Sleep stage W, N1, N2, N3 and R.

    Expected structure::

        SHHS/polysomnography/
            edfs/shhs1/shhs1-200001.edf
            annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml

    The same structure applies to shhs2. Access to the original data must be
    obtained from NSRR. Profusion XML is a different format and is not read
    by this class. See :doc:`/user_guide/shhs` for the signal and split rules.

    Args:
        root_path (str): Dataset root, or its polysomnography directory.
            (default: :obj:`'./SHHS'`)
        channels (list): Exact EDF channel names, in output order. Missing
            channels raise an error. (default: :obj:`['EEG', 'EOG(L)']`)
        visits (list): Visit numbers to discover. (default: :obj:`[1, 2]`)
        l_freq (float, optional): FIR high-pass cutoff in Hz. None disables
            the high-pass. (default: :obj:`0.5`)
        h_freq (float, optional): FIR low-pass cutoff in Hz. None disables
            the low-pass. (default: :obj:`30`)
        sfreq (float): Output sampling rate in Hz. (default: :obj:`100`)
        start_at (float): Earliest retained complete epoch in recording
            seconds. (default: :obj:`0`)
        end_at (float, optional): Exclusive recording-time limit. None uses
            the whole recording. (default: :obj:`None`)
        online_transform (Callable, optional): Transform called with eeg
            when a cached sample is read. (default: :obj:`None`)
        offline_transform (Callable, optional): Transform called with eeg
            before a sample is cached. (default: :obj:`None`)
        label_transform (Callable, optional): Transform called with the
            metadata dictionary as y. (default: :obj:`None`)
        before_trial (Callable, optional): Transform the ordered epoch array
            without changing epoch count or order. (default: :obj:`None`)
        after_trial (Callable, optional): Trial hook. (default: :obj:`None`)
        after_session (Callable, optional): Session hook. (default: :obj:`None`)
        after_subject (Callable, optional): Subject hook. (default: :obj:`None`)
        io_path (str, optional): Cache directory. (default: :obj:`None`)
        io_size (int): Initial LMDB size. (default: :obj:`1048576`)
        io_mode (str): EEG cache mode. (default: :obj:`'lmdb'`)
        num_worker (int): Preprocessing workers. (default: :obj:`0`)
        verbose (bool): Show preprocessing progress. (default: :obj:`True`)
    '''

    def __init__(self, root_path: str = './SHHS', channels: List = ['EEG', 'EOG(L)'],
                 visits: List = [1, 2], l_freq: Union[float, None] = 0.5,
                 h_freq: Union[float, None] = 30., sfreq: float = 100,
                 start_at: float = 0., end_at: Union[float, None] = None,
                 online_transform: Union[Callable, None] = None,
                 offline_transform: Union[Callable, None] = None,
                 label_transform: Union[Callable, None] = None,
                 before_trial: Union[Callable, None] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[str, None] = None, io_size: int = 1048576,
                 io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True, **kwargs):
        _settings(channels, sfreq, l_freq, h_freq, start_at, end_at)
        _validate_visits(visits)
        params = dict(root_path=root_path, channels=list(channels), visits=list(visits),
                      l_freq=l_freq, h_freq=h_freq, sfreq=sfreq, start_at=start_at,
                      end_at=end_at, online_transform=online_transform,
                      offline_transform=offline_transform, label_transform=label_transform,
                      before_trial=before_trial, after_trial=after_trial,
                      after_session=after_session, after_subject=after_subject,
                      io_path=io_path if io_path is not None else get_random_dir_path(dir_prefix='datasets'),
                      io_size=io_size, io_mode=io_mode, num_worker=num_worker, verbose=verbose)
        params.update(kwargs)
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def read_record(record: Tuple, channels: List = ['EEG', 'EOG(L)'],
                    l_freq: Union[float, None] = 0.5, h_freq: Union[float, None] = 30.,
                    sfreq: float = 100, start_at: float = 0.,
                    end_at: Union[float, None] = None, **kwargs) -> Dict:
        _record_info(record)
        samples = _settings(channels, sfreq, l_freq, h_freq, start_at, end_at)
        scored = _scored_epochs(record[1])
        with mne.io.read_raw_edf(record[0], include=list(channels), preload=True, verbose=False) as raw:
            if set(raw.ch_names) != set(channels):
                raise ValueError(f'Missing required SHHS channels: {sorted(set(channels) - set(raw.ch_names))}.')
            raw.reorder_channels(list(channels))
            # MNE converts voltage channels to volts; other physical units
            # remain native. Keep the original unit for non-voltage channels.
            original_units = getattr(raw, '_orig_units', None) or {}
            units = ['V' if original_units.get(name) in ('V', 'mV', 'uV', 'µV', 'μV', 'nV')
                     else original_units.get(name) for name in channels]
            if not np.isfinite(raw.get_data()).all():
                raise ValueError('SHHS signals must be finite.')
            if l_freq is not None or h_freq is not None:
                raw.filter(l_freq=l_freq, h_freq=h_freq, picks='all',
                           method='fir', fir_design='firwin', verbose=False)
            if raw.info['sfreq'] != sfreq:
                raw.resample(sfreq, npad='auto', verbose=False)
            data = raw.get_data()
            if not np.isfinite(data).all():
                raise ValueError('SHHS preprocessing must produce finite signals.')
            duration = data.shape[1] / sfreq
            upper = duration if end_at is None else min(end_at, duration)
            selected = [(onset, label) for onset, label in scored
                        if onset >= start_at - 1e-8 and onset + 30. <= upper + 1e-8]
            if not selected:
                raise ValueError('The selected interval contains no complete scored epochs.')
            onsets, labels = zip(*selected)
            starts = np.rint(np.asarray(onsets) * sfreq).astype(int)
            epochs = np.stack([data[:, begin:begin + samples] for begin in starts])
        return dict(epochs_data=epochs, epochs_label=list(labels),
                    epochs_onsets=starts / sfreq, epochs_channels=list(channels), epochs_units=units)

    @staticmethod
    def fake_record(record: Tuple, **kwargs) -> Dict:
        return dict(record=('shhs1-200001.edf', 'shhs1-200001-nsrr.xml'),
                    epochs_data=np.zeros((4, 2, 3000)),
                    epochs_label=['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage R'],
                    epochs_onsets=np.arange(4) * 30., epochs_channels=['EEG', 'EOG(L)'],
                    epochs_units=['V', 'V'])

    @staticmethod
    def process_record(record: Tuple, epochs_data: np.ndarray, epochs_label: List,
                       epochs_onsets: np.ndarray, epochs_channels: List, epochs_units: List,
                       before_trial: Union[Callable, None] = None,
                       offline_transform: Union[Callable, None] = None, **kwargs):
        info = _record_info(record)
        data = np.asarray(epochs_data)
        onsets = np.asarray(epochs_onsets, dtype=float)
        if (data.ndim != 3 or len(data) != len(epochs_label)
                or onsets.shape != (len(data),) or not np.isfinite(onsets).all()
                or np.any(onsets < 0) or np.any(np.diff(onsets) < 30. - 1e-6)
                or len(epochs_channels) != data.shape[1] or len(epochs_units) != data.shape[1]
                or any(unit is not None and not isinstance(unit, str) for unit in epochs_units)
                or len(set(epochs_channels)) != len(epochs_channels)
                or any(not isinstance(name, str) or not name for name in epochs_channels)
                or any(label not in set(_STAGES.values()) - {None} for label in epochs_label)):
            raise ValueError('Invalid SHHS epoch data, labels, times or channels.')
        if before_trial is not None:
            data = before_trial(data)
            if len(data) != len(epochs_label):
                raise ValueError('before_trial must preserve the number and order of epochs.')
        for i, (eeg, label, onset) in enumerate(zip(data, epochs_label, onsets)):
            if offline_transform is not None:
                eeg = offline_transform(eeg=eeg)['eeg']
            clip_id = f"{info['record_id']}_{i}"
            metadata = dict(info, trial_id=info['record_id'], clip_id=clip_id,
                            start_at=float(onset), end_at=float(onset + 30.), label=label,
                            channel_names=json.dumps(epochs_channels), channel_units=json.dumps(epochs_units))
            yield dict(eeg=eeg, key=clip_id, info=metadata)

    def set_records(self, root_path, visits=(1, 2), **kwargs):
        _validate_visits(visits)
        root = Path(root_path)
        if (root / 'polysomnography').is_dir():
            root /= 'polysomnography'
        recordings, scores = {}, {}
        for visit in sorted(visits):
            folder = f'shhs{visit}'
            for base, suffix, output in [(root / 'edfs' / folder, '.edf', recordings),
                                         (root / 'annotations-events-nsrr' / folder, '-nsrr.xml', scores)]:
                for path in sorted(base.glob('*' + suffix)):
                    if not path.is_file():
                        continue
                    name = path.name[:-len(suffix)]
                    if re.fullmatch(rf'shhs{visit}-\d{{6}}', name) is None:
                        raise ValueError(f'Unrecognized SHHS filename: {path.name}.')
                    if name in output:
                        raise ValueError(f'Ambiguous SHHS files for {name}.')
                    output[name] = str(path.resolve())
        if recordings.keys() != scores.keys():
            raise ValueError('Unpaired SHHS files: missing annotations for '
                             f'{sorted(recordings.keys() - scores.keys())}; missing recordings for '
                             f'{sorted(scores.keys() - recordings.keys())}.')
        if not recordings:
            raise ValueError('No SHHS EDF/NSRR XML pairs were found.')
        return [(recordings[name], scores[name]) for name in sorted(recordings)]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        info = self.read_info(index)
        eeg = self.read_eeg(str(info['_record_id']), str(info['clip_id']))
        label = info
        if self.online_transform is not None:
            eeg = self.online_transform(eeg=eeg)['eeg']
        if self.label_transform is not None:
            label = self.label_transform(y=info)['y']
        return eeg, label

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, root_path=self.root_path, channels=self.channels,
                    visits=self.visits, sfreq=self.sfreq, l_freq=self.l_freq,
                    h_freq=self.h_freq, start_at=self.start_at, end_at=self.end_at)
