import json
import re
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np

from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path
from ...constants.clinical.adsz import ADSZ_CHANNEL_LIST, ADSZ_SCHIZOPHRENIA_CHANNEL_LIST


_SUBSETS = {
    'alzheimer': dict(channels=ADSZ_CHANNEL_LIST, groups={'Healthy': ('HC', 0), 'AD': ('AD', 1)}, extension='.out'),
    'schizophrenia': dict(channels=ADSZ_SCHIZOPHRENIA_CHANNEL_LIST, groups={'Healthy': ('HC', 0), 'SCH': ('SCH', 1)}, extension='.eea'),
}


def _settings(subset, chunk_size, overlap, num_channel, sfreq, notch_freqs=None,
              l_freq=None, h_freq=None, ref_channels=None, source_unit=None):
    if not isinstance(subset, str) or subset not in _SUBSETS:
        raise ValueError('subset must be alzheimer or schizophrenia.')
    channels = _SUBSETS[subset]['channels']
    num_channel = len(channels) if num_channel is None else num_channel
    for value in (chunk_size, overlap, num_channel, sfreq):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError('Chunk, overlap, channel and sampling settings must be integers.')
    if (not 1 <= num_channel <= len(channels) or sfreq <= 0 or overlap < 0
            or (chunk_size > 0 and overlap >= chunk_size)):
        raise ValueError('Require valid channels, a positive rate and overlap smaller than a positive chunk.')
    for value in (notch_freqs, l_freq, h_freq):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 < value < 64):
            raise ValueError('Filter frequencies must be finite, positive and below source Nyquist (64 Hz).')
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError('l_freq must be smaller than h_freq.')
    if ref_channels not in (None, 'average'):
        raise ValueError('ref_channels must be None or average.')
    if source_unit not in (None, 'source', 'uV', 'V'):
        raise ValueError('source_unit must be None, source, uV or V.')
    return int(num_channel)


def _metadata(path, subset):
    config = _SUBSETS[subset]
    if path.parent.name not in config['groups'] or path.suffix != config['extension']:
        raise ValueError(f'Unexpected ADSZ group or file extension: {path}')
    group, label = config['groups'][path.parent.name]
    stem = path.stem
    if subset == 'alzheimer':
        pattern = r'(ec|eo)([0-9]+)' if group == 'AD' else r'eeg([0-9]+)([co])([0-9]+)'
        match = re.fullmatch(pattern, stem)
        if match is None:
            raise ValueError(f'Unrecognized Alzheimer source filename: {path.name}')
        code, number = match.groups() if group == 'AD' else (match[2], match[1])
        subject = f'{group}_{number}'
        basis = 'conservative_filename_group_not_clinically_verified'
    else:
        if not re.fullmatch(r'[A-Za-z0-9-]+', stem):
            raise ValueError(f'Unrecognized schizophrenia source filename: {path.name}')
        subject, code = f'{group}_{stem.lower()}', 'n/a'
        basis = 'provider_one_file_per_subject'
    return dict(subset=subset, class_name=group, label=label, subject_id=f'ADSZ_{subset}_{subject}',
                record_id=f'ADSZ_{subset}_{group}_{stem}', file_name=path.name,
                filename_condition_code=code, identity_basis=basis)


class ADSZDataset(BaseDataset):
    r'''
    Read the two independent cohorts in the Figshare Alzheimer's disease and
    schizophrenia release. The default selects Alzheimer/healthy-control
    matrices; ``subset='schizophrenia'`` selects the original MSU column files.

    - Author: Alves et al.; schizophrenia data obtained/prepared by Gorbachevskaya and Borisov.
    - Year: 2022
    - Download URL: https://doi.org/10.6084/m9.figshare.19091771.v1
    - Reference: Alves C L, Pineda A M, Roster K, et al. EEG functional connectivity and deep learning for automatic diagnosis of brain disorders: Alzheimer's disease and schizophrenia. Journal of Physics: Complexity, 2022, 3(2): 025001.
    - Signals: Alzheimer 19 channels; schizophrenia 16 channels, both 128 Hz.
    - Classes: HC (0), AD (1) for Alzheimer; HC (0), SCH (1) for schizophrenia.

    Accept the full extraction root (``ADSZ/dataset/alzheimer/AD/*.out`` and
    ``ADSZ/dataset/schizophrenia/SCH/*.eea``), the ``dataset`` directory or the
    selected cohort directory. These are separate binary tasks, not one
    three-class clinical cohort. Read every native AD row, including records
    longer than eight seconds. MSU columns contain 16 consecutive blocks of
    7680 microvolt samples in the provider's channel order.

    AD source units are unspecified: preserve numerical values by default.
    MSU documents microvolts: convert that subset to volts by default.
    ``source_unit`` explicitly overrides the source-unit interpretation. The
    earlier unreviewed AD implementation assumed microvolts; select ``'uV'``
    to reproduce that scale. Do not infer physical calibration from amplitude.

    AD ``subject_id`` conservatively groups matching ``ec/eo`` suffixes and
    matching Healthy ``eeg<number>c/o`` prefixes. This is a filename grouping,
    not a verified clinical identity table or verification of eye state.
    MSU states that each file represents one subject; keep its complete name.
    See :doc:`/user_guide/adsz` for provenance and evaluation limits.

    .. code-block:: python

        from torcheeg import transforms
        from torcheeg.datasets import ADSZDataset

        dataset = ADSZDataset(root_path='./ADSZ', subset='alzheimer',
                             io_path='./cache/adsz-alzheimer-native',
                             online_transform=transforms.ToTensor(),
                             label_transform=transforms.Select('label'))
        eeg, label = dataset[0]  # [19, 128], unchanged source units

    Args:
        root_path (str): Full extraction or selected cohort directory. (default: :obj:`'./ADSZ'`)
        chunk_size (int): Output samples per window; nonpositive selects the entire record. (default: :obj:`128`)
        overlap (int): Shared samples between adjacent windows. (default: :obj:`0`)
        num_channel (int, optional): First channels after full-channel preprocessing. None selects all 19 or 16. (default: :obj:`None`)
        sfreq (int): Output rate in Hz. (default: :obj:`128`)
        notch_freqs (float, optional): Additional notch frequency. (default: :obj:`None`)
        l_freq (float, optional): Additional high-pass cutoff. (default: :obj:`None`)
        h_freq (float, optional): Additional low-pass cutoff. (default: :obj:`None`)
        ref_channels (str, optional): Additional average reference over all recorded channels. (default: :obj:`None`)
        online_transform (Callable, optional): Transform when reading a sample. (default: :obj:`None`)
        offline_transform (Callable, optional): Transform before caching a window. (default: :obj:`None`)
        label_transform (Callable, optional): Transform metadata into the target. (default: :obj:`None`)
        before_trial (Callable, optional): Hook on an independent recording array; preserve finite shape. (default: :obj:`None`)
        after_trial (Callable, optional): Hook on cached windows of a record. (default: :obj:`None`)
        after_subject (Callable, optional): Hook on cached windows of the documented identity group. (default: :obj:`None`)
        io_path (str, optional): Cache path; None generates a path. Use a fresh path for each subset, unit or preprocessing setting. (default: :obj:`None`)
        io_size (int): Maximum LMDB map size in bytes. (default: :obj:`10485760`)
        io_mode (str): lmdb, pickle or memory storage. (default: :obj:`'lmdb'`)
        num_worker (int): Preprocessing workers. (default: :obj:`0`)
        verbose (bool): Display preprocessing progress. (default: :obj:`True`)
        subset (str): alzheimer or schizophrenia. (default: :obj:`'alzheimer'`)
        source_unit (str, optional): None follows the subset source evidence; source preserves numbers, uV converts to volts, V preserves volts. An override is caller-supplied, not verified calibration. (default: :obj:`None`)
    '''

    def __init__(self, root_path: str = './ADSZ', chunk_size: int = 128,
                 overlap: int = 0, num_channel: Union[int, None] = None, sfreq: int = 128,
                 notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None,
                 online_transform: Union[Callable, None] = None,
                 offline_transform: Union[Callable, None] = None,
                 label_transform: Union[Callable, None] = None,
                 before_trial: Union[Callable, None] = None,
                 after_trial: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[str, None] = None, io_size: int = 10485760,
                 io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True,
                 subset: str = 'alzheimer', source_unit: Union[str, None] = None):
        num_channel = _settings(subset, chunk_size, overlap, num_channel, sfreq,
                                notch_freqs, l_freq, h_freq, ref_channels, source_unit)
        params = dict(root_path=root_path, chunk_size=chunk_size, overlap=overlap,
                      num_channel=num_channel, sfreq=sfreq, notch_freqs=notch_freqs,
                      l_freq=l_freq, h_freq=h_freq, ref_channels=ref_channels,
                      online_transform=online_transform, offline_transform=offline_transform,
                      label_transform=label_transform, before_trial=before_trial,
                      after_trial=after_trial, after_subject=after_subject,
                      io_path=io_path if io_path is not None else get_random_dir_path(dir_prefix='datasets'),
                      io_size=io_size, io_mode=io_mode, num_worker=num_worker,
                      verbose=verbose, subset=subset, source_unit=source_unit)
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def set_records(root_path='./ADSZ', subset='alzheimer', **kwargs) -> List[Dict]:
        _settings(subset, 128, 0, None, 128)
        root = Path(root_path)
        folders = _SUBSETS[subset]['groups']
        candidates = {path.resolve() for path in (root, root / subset, root / 'dataset' / subset)
                      if all((path / name).is_dir() for name in folders)}
        if not candidates:
            raise FileNotFoundError(f'No complete {subset} class directories under {root}.')
        if len(candidates) != 1:
            raise ValueError('Multiple ADSZ cohort roots found; select one unambiguous directory.')
        selected = candidates.pop()
        records = []
        for folder in folders:
            paths = sorted((selected / folder).glob('*' + _SUBSETS[subset]['extension']))
            if not paths:
                raise ValueError(f'No {subset} records in {selected / folder}.')
            for path in paths:
                if not path.is_file():
                    raise ValueError(f'Expected an ADSZ signal file: {path}')
                records.append(dict(file_path=str(path), **_metadata(path, subset)))
        keys = [record['record_id'].lower() for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate case-insensitive ADSZ record identities.')
        return records

    @staticmethod
    def read_record(record: Dict, num_channel=None, sfreq=128, notch_freqs=None,
                    l_freq=None, h_freq=None, ref_channels=None, subset=None,
                    source_unit=None, **kwargs) -> Dict:
        subset = record['subset'] if subset is None else subset
        num_channel = _settings(subset, 128, 0, num_channel, sfreq,
                                notch_freqs, l_freq, h_freq, ref_channels, source_unit)
        path = Path(record['file_path'])
        metadata = _metadata(path, subset)
        if any(record.get(key) != value for key, value in metadata.items()):
            raise ValueError('ADSZ record metadata do not match its source file.')
        source = np.loadtxt(path, ndmin=2)
        channels = _SUBSETS[subset]['channels']
        if subset == 'alzheimer':
            if source.ndim != 2 or source.shape[1] != 19 or source.shape[0] == 0:
                raise ValueError('Alzheimer .out data must be a nonempty samples-by-19 matrix.')
            signal = source.T.copy()
        else:
            if source.shape != (16 * 7680, 1):
                raise ValueError('Schizophrenia .eea data must contain one column of 16 x 7680 samples.')
            signal = source[:, 0].reshape(16, 7680).copy()
        if not np.isfinite(signal).all():
            raise ValueError('ADSZ data must contain only finite values.')
        unit = source_unit if source_unit is not None else ('source' if subset == 'alzheimer' else 'uV')
        if unit == 'uV':
            signal *= 1e-6
        basis = ('caller_supplied_unit' if source_unit is not None else
                 'unspecified_source_unit' if subset == 'alzheimer' else 'provider_microvolts')
        if any(value is not None for value in (notch_freqs, l_freq, h_freq, ref_channels)) or sfreq != 128:
            raw = mne.io.RawArray(signal, mne.create_info(channels, 128, 'eeg'), verbose=False)
            if notch_freqs is not None:
                raw.notch_filter(notch_freqs, verbose=False)
            if l_freq is not None or h_freq is not None:
                raw.filter(l_freq, h_freq, verbose=False)
            if ref_channels is not None:
                raw.set_eeg_reference(ref_channels, verbose=False)
            if sfreq != 128:
                raw.resample(sfreq, verbose=False)
            signal = raw.get_data()
        signal = signal[:num_channel].copy()
        if not signal.shape[1] or not np.isfinite(signal).all():
            raise ValueError('Preprocessing must produce a nonempty finite signal.')
        return dict(signal=signal, sfreq=sfreq, num_channel=num_channel, signal_unit='source' if unit == 'source' else 'V',
                    units_basis=basis, source_unit=unit, channel_names=json.dumps(channels[:num_channel]),
                    **metadata)

    @staticmethod
    def fake_record(subset='alzheimer', num_channel=None, sfreq=128, **kwargs) -> Dict:
        num_channel = _settings(subset, 128, 0, num_channel, sfreq)
        group = 'AD' if subset == 'alzheimer' else 'SCH'
        name = 'ec0101.out' if subset == 'alzheimer' else '022w1.eea'
        folder = group
        metadata = _metadata(Path(folder) / name, subset)
        unit = 'source' if subset == 'alzheimer' else 'V'
        return dict(signal=np.random.default_rng(42).normal(size=(num_channel, 8 * sfreq)),
                    sfreq=sfreq, num_channel=num_channel, signal_unit=unit, units_basis='synthetic', source_unit=unit,
                    channel_names=json.dumps(_SUBSETS[subset]['channels'][:num_channel]), **metadata)

    @staticmethod
    def process_record(signal: np.ndarray, label: int, class_name: str,
                       subject_id: str, file_name: str, sfreq: int,
                       chunk_size: int = 128, overlap: int = 0, num_channel=None,
                       before_trial: Union[Callable, None] = None,
                       offline_transform: Union[Callable, None] = None,
                       subset='alzheimer', record_id=None, signal_unit='source',
                       units_basis='unspecified_source_unit', source_unit='source',
                       channel_names=None, identity_basis='source_record',
                       filename_condition_code='n/a', **kwargs):
        num_channel = _settings(subset, chunk_size, overlap, num_channel, sfreq)
        data = np.asarray(signal)
        if data.ndim != 2 or data.shape[0] < num_channel or data.shape[1] == 0 or not np.isfinite(data).all():
            raise ValueError('Signal must have enough channels and finite nonempty samples.')
        trial = data[:num_channel].copy()
        if before_trial is not None:
            trial = np.asarray(before_trial(trial))
            if trial.shape != (num_channel, data.shape[1]) or not np.isfinite(trial).all():
                raise ValueError('before_trial must preserve finite signal geometry.')
        size = trial.shape[1] if chunk_size <= 0 else chunk_size
        if overlap >= size:
            raise ValueError('overlap must be smaller than the actual window.')
        record_id = record_id if record_id is not None else f'ADSZ_{subset}_{class_name}_{Path(file_name).stem}'
        metadata = dict(subject_id=subject_id, class_name=class_name, label=label, file_name=file_name,
                        sfreq=sfreq, subset=subset, record_id=record_id, trial_id=record_id,
                        signal_unit=signal_unit, source_unit=source_unit, units_basis=units_basis,
                        channel_names=channel_names or json.dumps(_SUBSETS[subset]['channels'][:num_channel]),
                        identity_basis=identity_basis, filename_condition_code=filename_condition_code)
        for index, start in enumerate(range(0, trial.shape[1] - size + 1, size - overlap)):
            clip = trial[:, start:start + size].copy()
            value = offline_transform(eeg=clip)['eeg'] if offline_transform is not None else clip
            key = f'{record_id}_{index}'
            info = dict(metadata, start_at=start, end_at=start + size, clip_id=key,
                        source_start_seconds=start / sfreq, source_end_seconds=(start + size) / sfreq)
            yield dict(eeg=value, key=key, info=info)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        info = self.read_info(index)
        eeg = self.read_eeg(str(info['_record_id']), str(info['clip_id']))
        signal = self.online_transform(eeg=eeg)['eeg'] if self.online_transform is not None else eeg
        label = self.label_transform(y=info)['y'] if self.label_transform is not None else info
        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{name: getattr(self, name) for name in
                    ('root_path', 'chunk_size', 'overlap', 'num_channel', 'sfreq', 'notch_freqs',
                     'l_freq', 'h_freq', 'ref_channels', 'subset', 'source_unit')})
