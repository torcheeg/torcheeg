import json
import re
import warnings
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np

from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path


ADFSU_CHANNEL_LIST = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2',
]

# Mapping from the classic ADFSU channel names to the standard ``standard_1020``
# montage names, used when attaching electrode positions in MNE.
ADFSU_CHANNEL_NAME_TO_1020 = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8',
}

# The two diagnostic groups and their integer labels (HC: 0, AD: 1).
ADFSU_GROUPS = {'Healthy': 0, 'AD': 1}

# The two resting-state recording conditions.
ADFSU_STATES = ['Eyes_closed', 'Eyes_open']

ADFSU_ORIGINAL_SFREQ = 128



def _settings(chunk_size, overlap, num_channel, sfreq, states,
              notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None):
    if (not isinstance(states, (list, tuple)) or not states
            or any(state not in ADFSU_STATES for state in states)
            or len(set(states)) != len(states)):
        raise ValueError('states must contain unique Eyes_closed/Eyes_open names.')
    for value in (chunk_size, overlap, num_channel, sfreq):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError('Chunk, overlap, channel and sampling settings must be integers.')
    if (not 1 <= num_channel <= 19 or sfreq <= 0 or overlap < 0
            or (chunk_size > 0 and overlap >= chunk_size)):
        raise ValueError('Require 1--19 channels, a positive rate and overlap smaller than a positive chunk.')
    for value in (notch_freqs, l_freq, h_freq):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 < value < ADFSU_ORIGINAL_SFREQ / 2):
            raise ValueError('Filter frequencies must be finite, positive and below source Nyquist.')
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError('l_freq must be smaller than h_freq.')
    if ref_channels not in (None, 'average'):
        raise ValueError('ref_channels must be None or average.')


class ADFSUDataset(BaseDataset):
    r'''
    The Florida State University resting EEG release contains plain-text
    channel arrays for healthy controls and probable Alzheimer's disease (AD).
    Each complete condition record contains 19 channels and 1024 samples at
    128 Hz. Keep the provider's already cleaned, band-limited signal by default.

    - Author: Vicchietti et al.
    - Year: 2023
    - Download URL: https://osf.io/2v5md/
    - Reference: Vicchietti M L, Ramos F M, Betting L E, Campanharo A S L O. Computational methods of EEG signals analysis for Alzheimer's disease classification. Scientific Reports, 2023, 13: 8184.
    - Labels: Healthy (0), AD (1).
    - Signal units: unchanged source numerical units; the archive does not specify a calibrated physical unit.

    Expected files are ``EEG_data/AD/Eyes_closed/Paciente1/Fp1.txt`` and the
    other canonical channel names under AD/Healthy and Eyes_closed/Eyes_open.
    See :doc:`/user_guide/adfsu` for the archive checksum, missing-condition
    policy, channel order, preprocessing and evaluation identity limits.

    Args:
        root_path (str): Extracted EEG_data directory. (default: :obj:`'./EEG_data'`)
        chunk_size (int): Output samples per window. Nonpositive values select the complete condition. (default: :obj:`128`)
        overlap (int): Output samples shared by adjacent windows. (default: :obj:`0`)
        num_channel (int): First 1--19 channels in canonical order. Preprocessing uses all 19 before selection. (default: :obj:`19`)
        sfreq (int): Output sampling rate in Hz. (default: :obj:`128`)
        notch_freqs (float, optional): Additional notch frequency. (default: :obj:`None`)
        l_freq (float, optional): Additional high-pass cutoff. (default: :obj:`None`)
        h_freq (float, optional): Additional low-pass cutoff. (default: :obj:`None`)
        ref_channels (str, optional): Additional average reference over all 19 recorded EEG channels. (default: :obj:`None`)
        online_transform (Callable, optional): Transform applied when a sample is read. (default: :obj:`None`)
        offline_transform (Callable, optional): Transform applied before caching a sample. (default: :obj:`None`)
        label_transform (Callable, optional): Transform applied to metadata. (default: :obj:`None`)
        before_trial (Callable, optional): Hook on an independent condition array; preserve finite shape. (default: :obj:`None`)
        after_trial (Callable, optional): Hook on cached windows of a condition. (default: :obj:`None`)
        after_subject (Callable, optional): Hook on cached windows of a conservative subject group. (default: :obj:`None`)
        io_path (str, optional): Cache path; None generates a path. (default: :obj:`None`)
        io_size (int): Maximum LMDB map size in bytes. (default: :obj:`10485760`)
        io_mode (str): lmdb, pickle or memory storage. (default: :obj:`'lmdb'`)
        num_worker (int): Preprocessing worker count. (default: :obj:`0`)
        verbose (bool): Display processing progress. (default: :obj:`True`)
        states (tuple or list): Select resting conditions. (default: :obj:`('Eyes_closed', 'Eyes_open')`)
    '''

    def __init__(self, root_path: str = './EEG_data', chunk_size: int = 128,
                 overlap: int = 0, num_channel: int = 19, sfreq: int = 128,
                 notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None,
                 online_transform: Union[Callable, None] = None,
                 offline_transform: Union[Callable, None] = None,
                 label_transform: Union[Callable, None] = None,
                 before_trial: Union[Callable, None] = None,
                 after_trial: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[str, None] = None, io_size: int = 10485760,
                 io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True,
                 states: Union[List[str], Tuple[str, ...]] = tuple(ADFSU_STATES)):
        _settings(chunk_size, overlap, num_channel, sfreq, states,
                  notch_freqs, l_freq, h_freq, ref_channels)
        params = dict(root_path=root_path, chunk_size=chunk_size, overlap=overlap,
                      num_channel=num_channel, sfreq=sfreq, notch_freqs=notch_freqs,
                      l_freq=l_freq, h_freq=h_freq, ref_channels=ref_channels,
                      online_transform=online_transform, offline_transform=offline_transform,
                      label_transform=label_transform, before_trial=before_trial,
                      after_trial=after_trial, after_subject=after_subject,
                      io_path=io_path if io_path is not None else get_random_dir_path(dir_prefix='datasets'),
                      io_size=io_size, io_mode=io_mode, num_worker=num_worker,
                      verbose=verbose, states=tuple(states))
        super().__init__(**params)
        self.__dict__.update(params)

    def set_records(self, root_path='./EEG_data', states=tuple(ADFSU_STATES), **kwargs):
        _settings(128, 0, 19, 128, states)
        root = Path(root_path)
        if not root.is_dir():
            raise FileNotFoundError(f'ADFSU root is not a directory: {root}')
        records = []
        for group in ADFSU_GROUPS:
            subjects = {}
            for state in states:
                folder = root / group / state
                if not folder.exists():
                    continue
                if not folder.is_dir():
                    raise ValueError(f'Expected an ADFSU condition directory: {folder}')
                for subject in sorted(folder.iterdir()):
                    if subject.name.startswith('.') or not subject.is_dir():
                        continue
                    if not re.fullmatch(r'Paciente[1-9][0-9]*', subject.name):
                        raise ValueError(f'Invalid ADFSU subject directory: {subject}')
                    paths = [subject / (channel + '.txt') for channel in ADFSU_CHANNEL_LIST]
                    present = [path.is_file() for path in paths]
                    if not any(present):
                        if any(subject.iterdir()):
                            raise ValueError(f'Condition has files but no canonical EEG channels: {subject}')
                        warnings.warn(f'Empty ADFSU condition omitted: {subject}', UserWarning)
                        continue
                    if not all(present):
                        raise FileNotFoundError(f'Incomplete 19-channel ADFSU condition: {subject}')
                    subjects.setdefault(subject.name, []).append(state)
            for subject in sorted(subjects, key=lambda name: int(name[8:])):
                records.append(dict(group=group, subject=subject,
                                    subject_id=f'{group}_{subject}', states=tuple(subjects[subject])))
        if not records:
            raise ValueError('No complete ADFSU records match the requested conditions.')
        return records

    @staticmethod
    def read_record(record: Dict, root_path='./EEG_data', num_channel=19, sfreq=128,
                    notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None,
                    states=tuple(ADFSU_STATES), **kwargs):
        _settings(128, 0, num_channel, sfreq, states,
                  notch_freqs, l_freq, h_freq, ref_channels)
        group, subject = record['group'], record['subject']
        if (group not in ADFSU_GROUPS or not re.fullmatch(r'Paciente[1-9][0-9]*', subject)
                or record['subject_id'] != f'{group}_{subject}'):
            raise ValueError('Invalid ADFSU group or conservative subject identity.')
        selected = record['states']
        if not selected or len(set(selected)) != len(selected) or any(state not in states for state in selected):
            raise ValueError('Record conditions must match the selected ADFSU states.')
        samples = []
        for state in selected:
            folder = Path(root_path) / group / state / subject
            arrays = [np.loadtxt(folder / (channel + '.txt')) for channel in ADFSU_CHANNEL_LIST]
            if any(array.shape != (1024,) or not np.isfinite(array).all() for array in arrays):
                raise ValueError(f'Each released ADFSU channel needs 1024 finite samples: {folder}')
            eeg = np.stack(arrays)
            if any(value is not None for value in (notch_freqs, l_freq, h_freq, ref_channels)) or sfreq != 128:
                # These operations are linear. Preserve the source numerical scale;
                # the archive supplies no voltage calibration to convert to SI units.
                raw = mne.io.RawArray(eeg, mne.create_info(list(ADFSU_CHANNEL_LIST), 128, 'eeg'), verbose=False)
                if notch_freqs is not None:
                    raw.notch_filter(notch_freqs, verbose=False)
                if l_freq is not None or h_freq is not None:
                    raw.filter(l_freq, h_freq, verbose=False)
                if ref_channels is not None:
                    raw.set_eeg_reference(ref_channels=ref_channels, verbose=False)
                if sfreq != 128:
                    raw.resample(sfreq, verbose=False)
                eeg = raw.get_data()
            samples.append(eeg[:num_channel].copy())
        return dict(samples=samples, states=list(selected), group=group,
                    subject_id=record['subject_id'], sfreq=sfreq)

    @staticmethod
    def fake_record(num_channel=19, sfreq=128, states=tuple(ADFSU_STATES), **kwargs):
        _settings(128, 0, num_channel, sfreq, states)
        rng = np.random.default_rng(42)
        record = dict(group='AD', subject='Paciente1', subject_id='AD_Paciente1', states=tuple(states))
        return dict(record=record, samples=[rng.standard_normal((num_channel, 8 * sfreq)) for _ in states],
                    states=list(states), group='AD', subject_id='AD_Paciente1', sfreq=sfreq)

    @staticmethod
    def process_record(record: Dict, samples: List[np.ndarray], states: List[str],
                       group: str, subject_id: str, sfreq=128, chunk_size=128,
                       overlap=0, num_channel=19, before_trial=None, offline_transform=None, **kwargs):
        _settings(chunk_size, overlap, num_channel, sfreq, states)
        if (group not in ADFSU_GROUPS or record['group'] != group or record['subject_id'] != subject_id
                or subject_id != f"{group}_{record['subject']}" or len(samples) != len(states)
                or tuple(states) != tuple(record['states'])):
            raise ValueError('Signal, condition and ADFSU identity metadata must agree.')
        for state, values in zip(states, samples):
            values = np.asarray(values)
            if (values.ndim != 2 or values.shape[0] < num_channel or values.shape[1] == 0
                    or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()):
                raise ValueError('ADFSU requires a finite nonempty EEG array.')
            segment = values[:num_channel].copy()
            if before_trial is not None:
                segment = np.asarray(before_trial(segment))
                if segment.shape != (num_channel, values.shape[1]) or segment.dtype.kind not in 'iuf' or not np.isfinite(segment).all():
                    raise ValueError('before_trial must preserve finite condition geometry.')
            size = segment.shape[1] if chunk_size <= 0 else chunk_size
            if overlap >= size:
                raise ValueError('Overlap must be smaller than each selected chunk.')
            identity = f'{subject_id}_{state}'
            for start in range(0, segment.shape[1] - size + 1, size - overlap):
                end = start + size
                clip = segment[:, start:end].copy()
                if offline_transform is not None:
                    clip = offline_transform(eeg=clip)['eeg']
                key = f'{identity}_{start}'
                info = dict(subject_id=subject_id, source_subject=record['subject'],
                            session_id=state, record_id=identity, trial_id=identity,
                            group=group, state=state, label=ADFSU_GROUPS[group],
                            clip_id=key, start_at=start, end_at=end, sfreq=sfreq,
                            source_sfreq=128, source_start_seconds=start / sfreq,
                            source_end_seconds=end / sfreq, signal_unit='source',
                            channel_names=json.dumps(ADFSU_CHANNEL_LIST[:num_channel]))
                yield dict(eeg=clip, key=key, info=info)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        info = self.read_info(index)
        eeg = self.read_eeg(str(info['_record_id']), str(info['clip_id']))
        signal = self.online_transform(eeg=eeg)['eeg'] if self.online_transform else eeg
        label = self.label_transform(y=info)['y'] if self.label_transform else info
        return signal, label

    @property
    def repr_body(self) -> Dict:
        names = ('root_path', 'chunk_size', 'overlap', 'num_channel', 'sfreq',
                 'notch_freqs', 'l_freq', 'h_freq', 'ref_channels', 'states',
                 'online_transform', 'offline_transform', 'label_transform',
                 'before_trial', 'after_trial', 'after_subject')
        return dict(super().repr_body, **{name: getattr(self, name) for name in names})
