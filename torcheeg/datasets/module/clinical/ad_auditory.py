import csv
import json
import os
import re
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np

from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path


AD_AUDITORY_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'
]
ORIGINAL_SFREQ = 250
GROUP_TO_LABEL = {'Normal': 0, 'Mild AD': 1, 'Moderate AD': 1, 'MCI': 2}
CLASS_NAMES = ['HC', 'AD', 'MCI']
TRIAL_TYPES = ('Stimulus', 'Rest')


def _settings(trial_types, chunk_size, overlap, num_channel, sfreq,
              notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None):
    if trial_types is not None and (not isinstance(trial_types, (list, tuple))
            or not trial_types or any(value not in TRIAL_TYPES for value in trial_types)
            or len(set(trial_types)) != len(trial_types)):
        raise ValueError('trial_types must be unique Stimulus/Rest names, or None for the whole recording.')
    for value in (chunk_size, overlap, num_channel, sfreq):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError('Chunk, overlap, channel and sampling settings must be integers.')
    if not 1 <= num_channel <= 19 or sfreq <= 0 or overlap < 0 or (chunk_size > 0 and overlap >= chunk_size):
        raise ValueError('Require 1--19 channels, a positive rate and overlap smaller than a positive chunk.')
    for value in (notch_freqs, l_freq, h_freq):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 < value < ORIGINAL_SFREQ / 2):
            raise ValueError('Filter frequencies must be finite, positive and below source Nyquist.')
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError('l_freq must be smaller than h_freq.')
    if ref_channels not in (None, 'average'):
        raise ValueError('ref_channels must be None or average.')


def _read_header(set_path):
    try:
        import h5py
    except ImportError as error:
        raise ImportError('Install torcheeg[dataset] to read AD-Auditory MATLAB v7.3 headers.') from error
    with h5py.File(set_path, 'r') as file:
        header = file['EEG'] if 'EEG' in file else file
        dimensions = {}
        for name in ('nbchan', 'srate', 'pnts', 'trials'):
            array = np.asarray(header[name])
            if array.size != 1 or array.dtype.kind not in 'iuf':
                raise ValueError(f'EEGLAB {name} must be a scalar.')
            value = array.item()
            if not np.isfinite(value) or value <= 0 or value != int(value):
                raise ValueError(f'EEGLAB {name} must be a positive integer.')
            dimensions[name] = int(value)
        if dimensions['nbchan'] != 19 or dimensions['srate'] != 250 or dimensions['trials'] != 1:
            raise ValueError('Require the released continuous 19-channel, 250 Hz AD-Auditory data.')
        def text(dataset):
            values = np.asarray(dataset)
            if values.dtype.kind not in 'iu' or values.ndim > 2:
                raise ValueError('EEGLAB text fields must contain character codes.')
            return ''.join(chr(int(value)) for value in values.reshape(-1))
        filename = text(header['data'])
        labels = [text(file[ref]) for ref in np.asarray(header['chanlocs']['labels']).reshape(-1)]
    if len(labels) != 19 or set(labels) != set(AD_AUDITORY_CHANNELS):
        raise ValueError('EEGLAB channel labels must contain each of the 19 expected EEG channels once.')
    expected = Path(set_path).with_suffix('.fdt').name
    if filename != expected:
        raise ValueError('The SET data field must name its matching local FDT file.')
    return dimensions, labels, filename


class ADAuditoryDataset(BaseDataset):
    r'''
    The AD-Auditory BIDS release (OpenNeuro ds005048 v1.0.0) contains 19-channel,
    250 Hz EEG during 40 Hz auditory entrainment. The released EEGLAB SET/FDT
    recordings have already been cleaned by the provider. Additional filtering,
    average reference and resampling are explicit options, not an attempt to
    repeat the provider's ASR/ICA pipeline.

    - Author: Lahijanian et al.
    - Year: 2024
    - Download URL: https://openneuro.org/datasets/ds005048/versions/1.0.0
    - Reference: Lahijanian M, Aghajan H, Vahabi Z. Auditory gamma-band entrainment enhances default mode network connectivity in dementia patients. Scientific Reports, 2024, 14: 13153.
    - Signals: 19 EEG channels. Output units follow the EEGLAB microvolt convention.
    - Labels: HC (0), AD (1, Mild AD and Moderate AD combined), MCI (2).

    Read participants.tsv, paired MATLAB v7.3 SET/FDT files and matching BIDS
    events.tsv. Unknown diagnosis rows are excluded. The event table defines
    40-second Stimulus and 20-second Rest intervals, with a final 5-second Rest
    interval. All samples from a participant must stay in one evaluation group.
    See :doc:`/user_guide/ad_auditory` for source units and preprocessing limits.

    Args:
        root_path (str): Downloaded BIDS root. (default: :obj:`'./AD-Auditory'`)
        trial_types (list or tuple, optional): Stimulus and/or Rest. None explicitly selects the whole recording, including time outside events. (default: :obj:`('Stimulus', 'Rest')`)
        chunk_size (int): Output samples per chunk. Nonpositive values select each complete chosen interval. (default: :obj:`250`)
        overlap (int): Output samples shared by adjacent chunks. (default: :obj:`0`)
        num_channel (int): First EEG channels in the documented canonical order. (default: :obj:`19`)
        sfreq (int): Output sampling rate in Hz. (default: :obj:`250`)
        notch_freqs (float, optional): Additional notch frequency, or None. (default: :obj:`None`)
        l_freq (float, optional): Additional high-pass cutoff, or None. (default: :obj:`None`)
        h_freq (float, optional): Additional low-pass cutoff, or None. (default: :obj:`None`)
        ref_channels (str, optional): Additional average EEG reference, or None. (default: :obj:`None`)
        online_transform (Callable, optional): Transform applied to a sample when read. (default: :obj:`None`)
        offline_transform (Callable, optional): Transform applied before caching a sample. (default: :obj:`None`)
        label_transform (Callable, optional): Transform applied to the metadata label. (default: :obj:`None`)
        before_trial (Callable, optional): Hook applied to a copy of each selected interval. It must preserve finite signal geometry. (default: :obj:`None`)
        after_trial (Callable, optional): Hook applied after a trial is cached. (default: :obj:`None`)
        after_subject (Callable, optional): Hook applied after a subject is cached. (default: :obj:`None`)
        io_path (str, optional): Cache path; None generates a path. (default: :obj:`None`)
        io_size (int): Maximum LMDB map size in bytes. (default: :obj:`10485760`)
        io_mode (str): lmdb, pickle or memory storage. (default: :obj:`'lmdb'`)
        num_worker (int): Preprocessing worker count. (default: :obj:`0`)
        verbose (bool): Display processing progress. (default: :obj:`True`)
    '''

    def __init__(self, root_path: str = './AD-Auditory',
                 trial_types: Union[List[str], Tuple[str, ...], None] = TRIAL_TYPES,
                 chunk_size: int = 250, overlap: int = 0, num_channel: int = 19,
                 sfreq: int = 250, notch_freqs=None, l_freq=None, h_freq=None,
                 ref_channels=None, online_transform: Union[Callable, None] = None,
                 offline_transform: Union[Callable, None] = None,
                 label_transform: Union[Callable, None] = None,
                 before_trial: Union[Callable, None] = None,
                 after_trial: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[str, None] = None, io_size: int = 10485760,
                 io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        _settings(trial_types, chunk_size, overlap, num_channel, sfreq,
                  notch_freqs, l_freq, h_freq, ref_channels)
        params = dict(root_path=root_path, trial_types=trial_types, chunk_size=chunk_size,
                      overlap=overlap, num_channel=num_channel, sfreq=sfreq,
                      notch_freqs=notch_freqs, l_freq=l_freq, h_freq=h_freq,
                      ref_channels=ref_channels, online_transform=online_transform,
                      offline_transform=offline_transform, label_transform=label_transform,
                      before_trial=before_trial, after_trial=after_trial,
                      after_subject=after_subject,
                      io_path=io_path if io_path is not None else get_random_dir_path(dir_prefix='datasets'),
                      io_size=io_size, io_mode=io_mode, num_worker=num_worker, verbose=verbose)
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def _read_participants(participants_path: str) -> Dict[str, str]:
        with open(participants_path, newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file, delimiter='\t')
            if not {'participant_id', 'Group'} <= set(reader.fieldnames or []):
                raise ValueError('participants.tsv requires participant_id and Group columns.')
            groups = {}
            for row in reader:
                subject, group = row['participant_id'], row['Group']
                if not subject or not re.fullmatch(r'sub-[A-Za-z0-9]+', subject) or subject in groups:
                    raise ValueError('Participant IDs must be valid and unique.')
                if group not in (*GROUP_TO_LABEL, '-', 'n/a'):
                    raise ValueError('Unknown diagnosis group in participants.tsv.')
                groups[subject] = group
        return groups

    def set_records(self, root_path='./AD-Auditory', trial_types=TRIAL_TYPES, **kwargs):
        root = Path(root_path)
        groups = self._read_participants(root / 'participants.tsv')
        records = []
        for subject, group in sorted(groups.items()):
            if group not in GROUP_TO_LABEL:
                continue
            paths = sorted(path for path in (root / subject).rglob('*_eeg.set')
                           if path.parent.name == 'eeg' and 'task-40HzAuditoryEntrainment' in path.stem.split('_'))
            if not paths:
                raise FileNotFoundError(f'No matching AD-Auditory SET record for {subject}.')
            for path in paths:
                if not path.name.startswith(subject + '_'):
                    raise ValueError('SET filename subject must match its BIDS subject directory.')
                events = path.with_name(path.name.replace('_eeg.set', '_events.tsv'))
                if not path.with_suffix('.fdt').is_file() or (trial_types is not None and not events.is_file()):
                    raise FileNotFoundError('Each selected SET needs its paired FDT and events table.')
                session = next((part[4:] for part in path.stem.split('_') if part.startswith('ses-')), 'n/a')
                records.append(dict(subject_id=subject, session_id=session, group=group, label=GROUP_TO_LABEL[group],
                                    record_id=path.stem[:-4], set_path=str(path),
                                    fdt_path=str(path.with_suffix('.fdt')),
                                    events_path=str(events) if events.is_file() else None))
        if not records:
            raise ValueError('No labeled AD-Auditory records were found.')
        return records

    @staticmethod
    def _read_events(events_path, duration=None):
        if events_path is None:
            raise ValueError('Event selection requires a matching events.tsv.')
        events = []
        with open(events_path, newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file, delimiter='\t')
            if not {'onset', 'duration', 'trial_type'} <= set(reader.fieldnames or []):
                raise ValueError('Events require onset, duration and trial_type columns.')
            last_end = 0.
            for index, row in enumerate(reader):
                onset, length = float(row['onset']), float(row['duration'])
                kind = row['trial_type']
                if (not np.isfinite(onset) or not np.isfinite(length) or onset < last_end
                        or length <= 0 or kind not in TRIAL_TYPES
                        or (duration is not None and onset + length > duration + 1e-9)):
                    raise ValueError('Events must be finite, ordered, nonoverlapping and inside the recording.')
                value = 2 if kind == 'Stimulus' else 1
                if row.get('value') not in (None, '') and float(row['value']) != value:
                    raise ValueError('Event value and trial_type must agree.')
                if row.get('sample') not in (None, '') and float(row['sample']) != round(onset * 250) + 1:
                    raise ValueError('The one-based event sample must agree with onset seconds.')
                events.append(dict(onset=onset, duration=length, trial_type=kind,
                                   value=value, event_index=index))
                last_end = onset + length
        if not events:
            raise ValueError('Event selection requires a nonempty events table.')
        return events

    @staticmethod
    def read_record(record: Dict, num_channel=19, sfreq=250, notch_freqs=None,
                    l_freq=None, h_freq=None, ref_channels=None, trial_types=TRIAL_TYPES,
                    **kwargs) -> Dict:
        _settings(trial_types, -1, 0, num_channel, sfreq,
                  notch_freqs, l_freq, h_freq, ref_channels)
        if record.get('label') != GROUP_TO_LABEL.get(record.get('group')) or record.get('group') not in GROUP_TO_LABEL:
            raise ValueError('Record group and diagnosis label must agree.')
        set_path = Path(record['set_path'])
        if (not set_path.name.startswith(record['subject_id'] + '_')
                or record.get('record_id') != set_path.stem[:-4]):
            raise ValueError('Record identity must match its SET filename.')
        dimensions, labels, filename = _read_header(record['set_path'])
        path = Path(record['set_path']).parent / filename
        if path.resolve() != Path(record['fdt_path']).resolve() or path.stat().st_size != dimensions['pnts'] * 19 * 4:
            raise ValueError('FDT path and byte count must agree with the SET header.')
        values = np.fromfile(path, dtype='<f4').reshape(-1, 19).T
        if not np.isfinite(values).all():
            raise ValueError('FDT signals must be finite.')
        values = values[[labels.index(name) for name in AD_AUDITORY_CHANNELS]].copy()
        if sfreq != 250 or any(value is not None for value in (notch_freqs, l_freq, h_freq, ref_channels)):
            raw = mne.io.RawArray(values.astype(np.float64) * 1e-6,
                  mne.create_info(AD_AUDITORY_CHANNELS, 250, 'eeg'), verbose=False)
            if notch_freqs is not None:
                raw.notch_filter(notch_freqs, verbose=False)
            if l_freq is not None or h_freq is not None:
                raw.filter(l_freq, h_freq, verbose=False)
            if ref_channels is not None:
                raw.set_eeg_reference(ref_channels, verbose=False)
            if sfreq != 250:
                raw.resample(sfreq, verbose=False)
            values = raw.get_data() * 1e6
        events = [] if trial_types is None else ADAuditoryDataset._read_events(
            record['events_path'], duration=dimensions['pnts'] / 250)
        return dict(eeg_data=values[:num_channel], events=events, label=record['label'],
                    group=record['group'], subject_id=record['subject_id'], sfreq=sfreq)

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        rate = kwargs.get('sfreq', 250)
        record = dict(subject_id='sub-fake', record_id='sub-fake_task-40HzAuditoryEntrainment')
        events = []
        for index in range(6):
            events.extend([dict(onset=5. + index * 60, duration=40., trial_type='Stimulus', value=2, event_index=2 * index),
                           dict(onset=45. + index * 60, duration=20. if index < 5 else 5., trial_type='Rest', value=1, event_index=2 * index + 1)])
        return dict(record=record, eeg_data=np.random.default_rng(42).standard_normal((19, 350 * rate)),
                    events=events, label=0, group='Normal', subject_id='sub-fake', sfreq=rate)

    @staticmethod
    def process_record(record: Dict, eeg_data: np.ndarray, events: List[Dict], label: int,
                       group: str, subject_id: str, sfreq: int, trial_types=TRIAL_TYPES,
                       chunk_size=250, overlap=0, num_channel=19,
                       before_trial: Union[Callable, None] = None,
                       offline_transform: Union[Callable, None] = None, **kwargs):
        _settings(trial_types, chunk_size, overlap, num_channel, sfreq)
        eeg_data = np.asarray(eeg_data)
        if (eeg_data.ndim != 2 or eeg_data.shape[0] < num_channel or eeg_data.dtype.kind not in 'iuf'
                or not np.isfinite(eeg_data).all() or group not in GROUP_TO_LABEL or GROUP_TO_LABEL.get(group) != label):
            raise ValueError('Require finite EEG channels and a matching diagnosis label.')
        identity = record.get('record_id', subject_id)
        if trial_types is None:
            selected = [dict(onset=0., duration=eeg_data.shape[1] / sfreq,
                             trial_type='Whole', value=0, event_index='whole')]
        else:
            selected = [event for event in events if event['trial_type'] in trial_types]
            if not selected:
                raise ValueError('No events match the requested trial types; whole-record mode requires None.')
        for event in selected:
            first = round(event['onset'] * sfreq)
            last = round((event['onset'] + event['duration']) * sfreq)
            if not 0 <= first < last <= eeg_data.shape[1]:
                raise ValueError('Selected event bounds must fit the actual output recording.')
            segment = eeg_data[:num_channel, first:last].copy()
            if before_trial is not None:
                segment = np.asarray(before_trial(segment))
                if (segment.shape != (num_channel, last - first) or segment.dtype.kind not in 'iuf'
                        or not np.isfinite(segment).all()):
                    raise ValueError('before_trial must preserve finite selected-interval geometry.')
            size = segment.shape[1] if chunk_size <= 0 else chunk_size
            if overlap >= size:
                raise ValueError('Overlap must be smaller than each selected chunk.')
            trial_id = f"{identity}_event-{event['event_index']}"
            for start in range(0, segment.shape[1] - size + 1, size - overlap):
                end = start + size
                clip = segment[:, start:end].copy()
                if offline_transform is not None:
                    clip = offline_transform(eeg=clip)['eeg']
                key = f'{trial_id}_{start}'
                info = dict(subject_id=subject_id, session_id=record.get('session_id', 'n/a'),
                            record_id=identity, trial_id=trial_id,
                            trial_index=event['event_index'], label=label, group=group,
                            trial_type=event['trial_type'], value=event['value'], clip_id=key,
                            start_at=first + start, end_at=first + end, sfreq=sfreq,
                            source_sfreq=250, signal_unit='uV',
                            source_start_seconds=(first + start) / sfreq,
                            source_end_seconds=(first + end) / sfreq,
                            event_onset_seconds=event['onset'], event_duration_seconds=event['duration'],
                            channel_names=json.dumps(AD_AUDITORY_CHANNELS[:num_channel]))
                yield dict(eeg=clip, key=key, info=info)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        info = self.read_info(index)
        eeg = self.read_eeg(str(info['_record_id']), str(info['clip_id']))
        signal = self.online_transform(eeg=eeg)['eeg'] if self.online_transform else eeg
        label = self.label_transform(y=info)['y'] if self.label_transform else info
        return signal, label

    @property
    def repr_body(self) -> Dict:
        names = ('root_path', 'trial_types', 'chunk_size', 'overlap', 'num_channel', 'sfreq',
                 'notch_freqs', 'l_freq', 'h_freq', 'ref_channels', 'online_transform',
                 'offline_transform', 'label_transform', 'before_trial', 'after_trial', 'after_subject')
        return dict(super().repr_body, **{name: getattr(self, name) for name in names})
