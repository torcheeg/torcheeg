import os
import json
import re
from numbers import Integral, Real
from typing import Callable, Dict, List, Tuple, Union

import mne
import numpy as np
import scipy.io as scio

from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path




CHANNELS = ['C3', 'Cz', 'C4', 'EOG1', 'EOG2', 'EOG3']


def _identity(record):
    if isinstance(record, dict):
        name = record['file_name']
    else:
        name = str(record)
    match = re.fullmatch(r'B(0[1-9])([TE])\.mat', name)
    if match is None:
        raise ValueError('Expected a BNCI MAT file named B01T.mat through B09E.mat.')
    identity = dict(file_name=name, subject_id=int(match.group(1)),
                    phase='train' if match.group(2) == 'T' else 'test')
    if isinstance(record, dict) and any(key in record and record[key] != identity[key]
                                        for key in ('subject_id', 'phase')):
        raise ValueError('Record metadata must match its subject and phase filename.')
    return identity


def _settings(chunk_size, overlap, num_channel, sfreq, offset_start, offset_end,
              l_freq=None, h_freq=None, notch_freqs=None, ref_channels=None):
    for value in (chunk_size, overlap, num_channel, sfreq, offset_start, offset_end):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError('Window, channel and sampling settings must be integers.')
    if not 1 <= num_channel <= 6 or sfreq <= 0 or not 0 <= offset_start < offset_end <= 1750:
        raise ValueError('Require 1--6 channels (three EEG, then three EOG), a positive output rate and source offsets within seven seconds.')
    first, last = int(round(offset_start * sfreq / 250)), int(round(offset_end * sfreq / 250))
    size = last - first if chunk_size <= 0 else chunk_size
    if not 0 < size <= last - first or not 0 <= overlap < size:
        raise ValueError('Chunk size must fit the resampled window and overlap must be smaller than it.')
    for name, value in [('l_freq', l_freq), ('h_freq', h_freq), ('notch_freqs', notch_freqs)]:
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 < value < 125):
            raise ValueError(f'{name} must be a finite frequency below the source Nyquist, or None.')
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError('l_freq must be smaller than h_freq.')
    if ref_channels not in (None, 'average'):
        raise ValueError('ref_channels must be None or average.')
    return first, last, size


def _vector(value, length, allowed, name):
    array = np.asarray(value)
    if (array.ndim > 2 or (array.ndim == 2 and 1 not in array.shape) or array.size != length
            or array.dtype.kind not in 'iuf' or not np.isin(array, allowed).all()):
        raise ValueError(f'{name} must contain one valid value per trial.')
    return array.reshape(-1).astype(int)


class BCICIV2bDataset(BaseDataset):
    r'''
    A dataset for motor imagery, BCI Competition 2008 Graz data set B (BCICIV_2b). This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Tangermann et al.
    - Year: 2012
    - Download URL: http://bnci-horizon-2020.eu/database/data-sets
    - Reference: Tangermann M, Müller K R, Aertsen A, et al. Review of the BCI competition IV[J]. Frontiers in neuroscience, 2012: 55.
    - Signals: Electroencephalogram (3 channels, i.e., C3, Cz, and C4 at 250Hz) and electrooculogram (3 channels, i.e., right, center, and left) of 9 subjects. Each subject participated in two screening sessions without feedback recorded on two different days within two weeks. Each session consisted of six runs with ten trials each and two classes of imagery. This resulted in 20 trials per run and 120 trials per session. Each trial started with a fixation cross and an additional short acoustic warning tone. Some seconds later a visual cue was presented for 1.25 seconds. Afterwards the subjects had to imagine the corresponding hand movement over a period of 4 seconds.
    - Category: The imagination of movement of the left hand (1), right hand (2).

    In order to use this dataset, the download folder :obj:`BCICIV_2b_mat` is required, containing the following files:

    .. code-block:: text

        BCICIV_2b_mat/
        ├── B01T.mat
        ├── B01E.mat
        ├── B02T.mat
        ├── B02E.mat
        ├── ...
        ├── B09T.mat
        └── B09E.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import BCICIV2bDataset
        from torcheeg import transforms

        dataset = BCICIV2bDataset(root_path='./BCICIV_2b_mat',
                                  online_transform=transforms.Compose([
                                      transforms.To2d(),
                                      transforms.ToTensor()
                                  ]),
                                  label_transform=transforms.Compose([
                                      transforms.Select('label'),
                                      transforms.Lambda(lambda x: x - 1)
                                  ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 3, 1125])
        # label (int)

    An example dataset with custom time window:

    .. code-block:: python

        from torcheeg.datasets import BCICIV2bDataset
        from torcheeg import transforms

        dataset = BCICIV2bDataset(root_path='./BCICIV_2b_mat',
                                  offset_start=500,  # 2s after trial start
                                  offset_end=1500,   # 6s after trial start
                                  chunk_size=1000,
                                  overlap=0,
                                  online_transform=transforms.ToTensor(),
                                  label_transform=transforms.Select('label'))
        print(dataset[0])

    An example dataset with preprocessing:

    .. code-block:: python

        from torcheeg.datasets import BCICIV2bDataset
        from torcheeg import transforms

        dataset = BCICIV2bDataset(root_path='./BCICIV_2b_mat',
                                  sfreq=128,
                                  l_freq=4.0,
                                  h_freq=40.0,
                                  notch_freqs=50.0,
                                  ref_channels='average',
                                  online_transform=transforms.ToTensor(),
                                  label_transform=transforms.Select('label'))
        print(dataset[0])

    The direct BNCI MATLAB reader retains labels 1/2 and expert artifact
    flags. Native outputs are microvolts, not MNE volts. Offsets are source
    samples at 250 Hz and are converted to the output grid. The default
    375:1500 window preserves the earlier draft's 1.5--6 second selection;
    use 750:1750 for an explicit 3--7 second imagery window. Whole-session
    processing precedes trial extraction. See :doc:`/user_guide/bciciv2b`.

    Args:
        root_path (str): Downloaded data files in matlab (unzipped BCICIV_2b_mat.zip) formats (default: :obj:`'./BCICIV_2b_mat'`)
        skip_trial_with_artifacts (bool): Omit trials flagged by the provider when True; otherwise retain flags in metadata. (default: :obj:`False`)
        chunk_size (int): Output samples per chunk; nonpositive values use the complete selected window, giving 1125 samples at the default 250 Hz. (default: :obj:`-1`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 3 channels are EEG signals (C3, Cz, C4). (default: :obj:`3`)
        sfreq (int): Target sampling rate in Hz for the output EEG signals. The original sampling rate is 250Hz. (default: :obj:`250`)
        notch_freqs (float, optional): Frequency to remove with notch filter (e.g., 50 or 60 Hz for line noise). If None, notch filter is not applied. (default: :obj:`None`)
        l_freq (float, optional): Low cutoff frequency for bandpass filter (high-pass). If None, no high-pass filter is applied. (default: :obj:`None`)
        h_freq (float, optional): High cutoff frequency for bandpass filter (low-pass). If None, no low-pass filter is applied. (default: :obj:`None`)
        ref_channels (str, optional): Reference channels for re-referencing. Can be 'average' for average reference, or None for no re-referencing. (default: :obj:`None`)
        offset_start (int): The start offset in data points from trial onset for extracting EEG segment. (default: :obj:`375`, i.e., 1.5s * 250Hz)
        offset_end (int): The end offset in data points from trial onset for extracting EEG segment. (default: :obj:`1500`, i.e., 6s * 250Hz)
        online_transform (Callable, optional): The transformation of the EEG signals. The input is a :obj:`np.ndarray`, and the output is used as the first value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the output is used as the second value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        after_session (Callable, optional): The hook performed on the session to which the sample belongs. Similar to after_trial but operates on session level.
        after_subject (Callable, optional): The hook performed on the subject to which the sample belongs. Similar to after_trial but operates on subject level.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    '''

    def __init__(self,
                 root_path: str = './BCICIV_2b_mat',
                 chunk_size: int = -1,
                 overlap: int = 0,
                 num_channel: int = 3,
                 sfreq: int = 250,
                 notch_freqs: Union[float, None] = None,
                 l_freq: Union[float, None] = None,
                 h_freq: Union[float, None] = None,
                 ref_channels: Union[str, None] = None,
                 offset_start: int = 375,
                 offset_end: int = 1500,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 skip_trial_with_artifacts: bool = False):

        _settings(chunk_size, overlap, num_channel, sfreq, offset_start, offset_end,
                  l_freq, h_freq, notch_freqs, ref_channels)
        if not isinstance(skip_trial_with_artifacts, bool):
            raise ValueError('skip_trial_with_artifacts must be boolean.')
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'sfreq': sfreq,
            'notch_freqs': notch_freqs,
            'l_freq': l_freq,
            'h_freq': h_freq,
            'ref_channels': ref_channels,
            'offset_start': offset_start,
            'offset_end': offset_end,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
            'after_session': after_session,
            'after_subject': after_subject,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose,
            'skip_trial_with_artifacts': skip_trial_with_artifacts
        }
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def read_record(record: Dict,
                    root_path: str = './BCICIV_2b_mat',
                    num_channel: int = 3,
                    sfreq: int = 250,
                    notch_freqs: Union[float, None] = None,
                    l_freq: Union[float, None] = None,
                    h_freq: Union[float, None] = None,
                    ref_channels: Union[str, None] = None,
                    offset_start: int = 375,
                    offset_end: int = 1500,
                    **kwargs) -> Dict:

        identity = _identity(record)
        _settings(-1, 0, num_channel, sfreq, offset_start, offset_end,
                  l_freq, h_freq, notch_freqs, ref_channels)
        data = scio.loadmat(os.path.join(root_path, identity['file_name']), simplify_cells=True)
        if 'data' not in data:
            raise ValueError('The MAT file must contain session data.')
        sessions = data['data']
        sessions = [sessions] if isinstance(sessions, dict) else list(np.asarray(sessions, dtype=object).reshape(-1))
        expected = 3 if identity['phase'] == 'train' else 2
        if len(sessions) != expected:
            raise ValueError('BNCI training files need three sessions and evaluation files need two.')
        result = dict(samples=[], labels=[], trial_ids=[], session_ids=[], artifact_flags=[],
                      trial_onsets=[], trial_indices=[], **{k: identity[k] for k in ('subject_id', 'phase')},
                      sfreq=sfreq)
        for session_index, session in enumerate(sessions):
            if not isinstance(session, dict) or not {'X', 'trial', 'y', 'fs', 'artifacts'} <= session.keys():
                raise ValueError('Each session needs X, trial, y, fs and artifacts.')
            if session['fs'] != 250:
                raise ValueError('The BNCI source header must specify 250 Hz.')
            values = np.asarray(session['X'])
            onsets = np.asarray(session['trial'])
            if (values.ndim != 2 or values.shape[1] != 6 or values.dtype.kind not in 'iuf'
                    or not np.isfinite(values).all() or onsets.size == 0
                    or onsets.dtype.kind not in 'iu' or onsets.ndim > 2
                    or (onsets.ndim == 2 and 1 not in onsets.shape)):
                raise ValueError('Require finite six-channel sessions and integer trial onset vectors.')
            onsets = onsets.reshape(-1).astype(int) - 1
            if np.any(onsets < 0) or np.any(np.diff(onsets) < 1750) or onsets[-1] + 1750 > len(values):
                raise ValueError('Seven-second trials must be ordered, complete and nonoverlapping.')
            labels = _vector(session['y'], len(onsets), [1, 2], 'y')
            artifacts = _vector(session['artifacts'], len(onsets), [0, 1], 'artifacts')
            selected = values[:, :num_channel].T.copy()
            # Process one continuous session in volts for MNE, then restore
            # the native microvolt output. Explicit EOG channels keep their type.
            if any(value is not None for value in (notch_freqs, l_freq, h_freq, ref_channels)) or sfreq != 250:
                raw = mne.io.RawArray(selected * 1e-6,
                      mne.create_info(CHANNELS[:num_channel], 250, (['eeg'] * 3 + ['eog'] * 3)[:num_channel]), verbose=False)
                if notch_freqs is not None:
                    raw.notch_filter(notch_freqs, picks=list(range(num_channel)), verbose=False)
                if l_freq is not None or h_freq is not None:
                    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=list(range(num_channel)), verbose=False)
                if ref_channels is not None:
                    raw.set_eeg_reference(ref_channels, verbose=False)
                if sfreq != 250:
                    raw.resample(sfreq, verbose=False)
                selected = raw.get_data() * 1e6
            if not np.isfinite(selected).all():
                raise ValueError('Preprocessing returned nonfinite values.')
            session_id = session_index + (1 if identity['phase'] == 'train' else 4)
            for trial_index, onset in enumerate(onsets):
                first = int(round(onset * sfreq / 250))
                last = first + 7 * sfreq
                if last > selected.shape[1]:
                    raise ValueError('Resampled seven-second trial is incomplete.')
                result['samples'].append(selected[:, first:last].copy())
                result['labels'].append(int(labels[trial_index]))
                result['trial_ids'].append(trial_index)
                result['session_ids'].append(session_id)
                result['artifact_flags'].append(int(artifacts[trial_index]))
                result['trial_onsets'].append(int(onset))
                result['trial_indices'].append(trial_index)
        return result

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        sfreq = kwargs.get('sfreq', 250)
        num_channel = kwargs.get('num_channel', 3)
        window_length = int(7 * sfreq)

        num_trials = 120  # Typical number of trials per session in 2b
        samples = [np.random.randn(num_channel, window_length)
                   for _ in range(num_trials)]
        labels = [((i % 2) + 1)
                  for i in range(num_trials)]  # 2 classes: 1 and 2
        trial_ids = list(range(num_trials))
        # 6 runs with 20 trials each
        session_ids = [1] * num_trials
        artifact_flags = [0] * num_trials

        return {
            'samples': samples,
            'labels': labels,
            'trial_ids': trial_ids,
            'session_ids': session_ids,
            'artifact_flags': artifact_flags,
            'subject_id': 1,
            'phase': 'train',
            'sfreq': sfreq
        }

    @staticmethod
    def process_record(record: Dict,
                       samples: List[np.ndarray],
                       labels: List[int],
                       trial_ids: List[int],
                       session_ids: List[int],
                       artifact_flags: List[int],
                       subject_id: int,
                       phase: str,
                       chunk_size: int = -1,
                       overlap: int = 0,
                       offset_start: int = 375,
                       offset_end: int = 1500,
                       sfreq: int = 250,
                       skip_trial_with_artifacts: bool = False,
                       trial_onsets=None,
                       trial_indices=None,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        channels = samples[0].shape[0] if samples else 3
        first, last, size = _settings(chunk_size, overlap, channels, sfreq, offset_start, offset_end)
        count = len(samples)
        if not count or any(len(values) != count for values in (labels, trial_ids, session_ids, artifact_flags)):
            raise ValueError('Trial signals and all metadata must have matching nonzero lengths.')
        if (isinstance(subject_id, (bool, np.bool_)) or not isinstance(subject_id, Integral)
                or not 1 <= subject_id <= 9 or phase not in ('train', 'test')
                or not isinstance(skip_trial_with_artifacts, bool)):
            raise ValueError('Require a valid subject, phase and boolean artifact policy.')
        allowed_sessions = [1, 2, 3] if phase == 'train' else [4, 5]
        session_ids = _vector(session_ids, count, allowed_sessions, 'session_ids')
        for optional in (trial_onsets, trial_indices):
            if optional is not None and len(optional) != count:
                raise ValueError('Optional trial metadata must align with the signals.')
        labels = _vector(labels, count, [1, 2], 'labels')
        artifacts = _vector(artifact_flags, count, [0, 1], 'artifacts')
        for index, trial in enumerate(samples):
            if skip_trial_with_artifacts and artifacts[index]:
                continue
            trial = np.asarray(trial)
            if (trial.shape != (channels, 7 * sfreq) or trial.dtype.kind not in 'iuf'
                    or not np.isfinite(trial).all()):
                raise ValueError('Trial signals must have the selected channels and seven output seconds.')
            segment = trial[:, first:last].copy()
            if before_trial is not None:
                segment = np.asarray(before_trial(segment))
                if (segment.shape != (channels, last - first) or segment.dtype.kind not in 'iuf'
                        or not np.isfinite(segment).all()):
                    raise ValueError('before_trial must preserve finite selected-window geometry.')
            trial_id = f"B{subject_id:02d}_{phase}_s{session_ids[index]}_t{trial_ids[index]}"
            for start in range(0, segment.shape[1] - size + 1, size - overlap):
                end = start + size
                clip = segment[:, start:end].copy()
                if offline_transform is not None:
                    clip = offline_transform(eeg=clip)['eeg']
                key = f'{trial_id}_{start}'
                info = dict(subject_id=subject_id, session_id=session_ids[index], phase=phase,
                            record_id=f'B{subject_id:02d}_{phase}_s{session_ids[index]}',
                            trial_id=trial_id, trial_index=trial_indices[index] if trial_indices is not None else index,
                            label=int(labels[index]), artifact=int(artifacts[index]), clip_id=key,
                            start_at=first + start, end_at=first + end, sfreq=sfreq,
                            signal_unit='uV', channel_names=json.dumps(CHANNELS[:channels]))
                if trial_onsets is not None:
                    info['source_trial_onset'] = trial_onsets[index]
                    output_onset = int(round(trial_onsets[index] * sfreq / 250))
                    info['source_start_seconds'] = (output_onset + first + start) / sfreq
                    info['source_end_seconds'] = (output_onset + first + end) / sfreq
                yield {'eeg': clip, 'key': key, 'info': info}

    def set_records(self, root_path: str = './BCICIV_2b_mat', **kwargs) -> List[Dict]:
        if not os.path.isdir(root_path):
            raise ValueError('root_path must be a downloaded BNCI MAT directory.')
        records = [_identity(name) for name in sorted(os.listdir(root_path))
                   if name.startswith('B') and name.endswith('.mat')]
        if not records:
            raise ValueError('No B01T.mat through B09E.mat files were found.')
        return records

    def __getitem__(self, index: int) -> Tuple:
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
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'sfreq': self.sfreq,
                'notch_freqs': self.notch_freqs,
                'l_freq': self.l_freq,
                'h_freq': self.h_freq,
                'ref_channels': self.ref_channels,
                'offset_start': self.offset_start,
                'offset_end': self.offset_end,
                'skip_trial_with_artifacts': self.skip_trial_with_artifacts,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'after_session': self.after_session,
                'after_subject': self.after_subject,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
