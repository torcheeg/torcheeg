import json
import os
import re
from numbers import Integral
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import scipy.io
from scipy import signal as scipy_signal

from ...constants.motor_imagery.shu_mi import SHU_MI_CHANNEL_LIST
from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path


def _settings(chunk_size, overlap, num_channel, sfreq):
    values = (chunk_size, overlap, num_channel, sfreq)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in values):
        raise ValueError('chunk_size, overlap, num_channel and sfreq must be integers.')
    if sfreq < 1 or not 1 <= num_channel <= 32:
        raise ValueError('sfreq must be positive and num_channel must be between 1 and 32.')
    size = 4 * sfreq if chunk_size <= 0 else chunk_size
    if size > 4 * sfreq or not 0 <= overlap < size:
        raise ValueError('Chunk length must fit a four-second trial and overlap must be smaller than it.')
    return size


def _identity(record):
    name = os.path.basename(record)
    match = re.fullmatch(r'(sub-\d{3})_(ses-\d{2})_task_motorimagery_eeg\.mat', name)
    if match is None:
        raise ValueError('Expected sub-XXX_ses-YY_task_motorimagery_eeg.mat.')
    return dict(subject_id=match.group(1), session_id=match.group(2), record_id=name[:-4])


def _trials(data, labels, samples):
    data, labels = np.asarray(data), np.asarray(labels)
    if (data.ndim != 3 or data.shape[1:] != (32, samples) or not len(data)
            or data.dtype.kind not in 'iuf' or not np.isfinite(data).all()):
        raise ValueError('data must contain finite real trials with 32 channels and four seconds of samples.')
    if (labels.ndim not in (1, 2) or (labels.ndim == 2 and 1 not in labels.shape)
            or labels.size != len(data) or labels.dtype.kind not in 'iuf'
            or not np.isin(labels, [1, 2]).all()):
        raise ValueError('labels must be a trial-aligned row, column or vector containing only 1 and 2.')
    return data, labels.reshape(-1)


class SHUMIDataset(BaseDataset):
    r'''
    The SHU Motor Imagery dataset was collected from 25 subjects at Shanghai University using a 32-channel EEG system. Each subject performed 5 sessions of left-hand and right-hand motor imagery tasks, with up to 100 retained trials per session. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The provider has already removed bad segments and baselines and applied a 0.5--40 Hz FIR filter. Four-second MAT trials at 250 Hz are resampled with SciPy to :obj:`sfreq` (default 200 Hz), then chunked. Signals remain in the published microvolt units; no amplitude scaling or additional filter is applied. The relevant information of the dataset is as follows:

    - Author: Ma et al.
    - Year: 2022
    - Download URL: https://figshare.com/articles/code/shu_dataset/19228725
    - Reference: Ma J, Yang B, Qiu W, et al. A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface[J]. Scientific Data, 2022, 9: 531.
    - Stimulus: Visual cues instructing subjects to perform left-hand or right-hand motor imagery.
    - Signals: Electroencephalogram (32 channels at 250Hz) from 25 subjects across 5 sessions each. The ``.mat`` files contain pre-segmented 4-second trials.
    - Rating: left hand (0), right hand (1).

    In order to use this dataset, the download folder is required, containing ``.mat`` files:

    .. code-block:: text

        SHU-MI/
        ├── sub-001_ses-01_task_motorimagery_eeg.mat
        ├── sub-001_ses-02_task_motorimagery_eeg.mat
        ├── ...
        └── sub-025_ses-05_task_motorimagery_eeg.mat

    An example dataset with 200Hz output, 32 channels and 1s chunks:

    .. code-block:: python

        from torcheeg.datasets import SHUMIDataset
        from torcheeg import transforms

        dataset = SHUMIDataset(root_path='./SHU-MI',
                               online_transform=transforms.ToTensor(),
                               label_transform=transforms.Compose([
                                   transforms.Select('label'),
                               ]))

        print(dataset[0])
        # EEG signal (torch.Tensor[32, 200]),
        # label (int)

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import SHUMIDataset
        from torcheeg import transforms

        dataset = SHUMIDataset(root_path='./SHU-MI',
                               online_transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.To2d()
                               ]),
                               label_transform=transforms.Compose([
                                   transforms.Select('label'),
                               ]))

        print(dataset[0])
        # EEG signal (torch.Tensor[1, 32, 200]),
        # label (int)

    The published trial arrays can contain fewer than 100 trials after artifact
    rejection. Channel order follows :obj:`SHU_MI_CHANNEL_LIST` in
    :mod:`torcheeg.datasets.constants`. The first :obj:`num_channel` channels
    are retained. ``start_at`` and ``end_at`` are output sample indices within
    the trial, with an exclusive end. ``trial_id`` includes the session record
    so windows from different sessions do not share a trial ID. Keep all
    windows of a trial together in a partition. See :doc:`/user_guide/shu_mi`.

    Args:
        root_path (str): Downloaded data directory containing ``.mat`` files. (default: :obj:`'./SHU-MI'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`200`, i.e. 1s at 200Hz)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. (default: :obj:`32`)
        sfreq (int): Target sampling rate in Hz for the output EEG signals. The original sampling rate is 250Hz. (default: :obj:`200`)
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

    ORIGINAL_SFREQ = 250
    TRIAL_DURATION = 4
    NUM_CLASSES = 2

    def __init__(self,
                 root_path: str = './SHU-MI',
                 chunk_size: int = 200,
                 overlap: int = 0,
                 num_channel: int = 32,
                 sfreq: int = 200,
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
                 verbose: bool = True):
        _settings(chunk_size, overlap, num_channel, sfreq)
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'sfreq': sfreq,
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
        }
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def read_record(record: str,
                    root_path: str = './SHU-MI',
                    sfreq: int = 200,
                    **kwargs) -> Dict:
        r'''
        Read a single ``.mat`` file for one session of one subject and resample
        to the target frequency. Each ``.mat`` file contains pre-segmented
        trials with keys ``'data'`` and ``'labels'``.

        Args:
            record (str): File name of the ``.mat`` file, e.g.
                ``'sub-001_ses-01_task_motorimagery_eeg.mat'``.
            root_path (str): Root directory of the dataset.
            sfreq (int): Target sampling rate in Hz.

        Returns:
            dict: A dictionary containing ``'data'`` with shape
            ``(n_trials, 32, resampled_points)`` and ``'labels'`` with shape
            ``(n_trials,)``.
        '''
        _identity(record)
        _settings(0, 0, 32, sfreq)
        mat = scipy.io.loadmat(os.path.join(root_path, record))
        if 'data' not in mat or 'labels' not in mat:
            raise ValueError('The trial MAT file must contain data and labels.')
        data, labels = _trials(mat['data'], mat['labels'], 1000)
        target_points = 4 * sfreq
        if target_points != 1000:
            data = scipy_signal.resample(data, target_points, axis=2)
        if not np.isfinite(data).all():
            raise ValueError('Resampling produced nonfinite trial values.')
        return {'data': data, 'labels': labels}

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        r'''
        Generate a fake record for testing purposes.

        Returns:
            dict: A dictionary containing random ``'data'`` with shape
            ``(100, 32, 800)``, dummy ``'labels'``, and a dummy ``'record'``
            key.
        '''
        sfreq = kwargs.get('sfreq', 200)
        num_points = SHUMIDataset.TRIAL_DURATION * sfreq
        n_trials = 100

        data = np.random.randn(n_trials, 32, num_points).astype(np.float32)
        labels = np.random.randint(1, 3, size=(n_trials,))

        return {
            'data': data,
            'labels': labels,
            'record': 'sub-001_ses-01_task_motorimagery_eeg.mat',
        }

    @staticmethod
    def process_record(record: str,
                       data: np.ndarray,
                       labels: np.ndarray,
                       chunk_size: int = 200,
                       overlap: int = 0,
                       num_channel: int = 32,
                       sfreq: int = 200,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        r'''
        Process resampled session trials into chunks in native microvolts.
        Source labels 1 (left hand) and 2 (right hand) become 0 and 1.

        Args:
            record (str): File name, e.g.
                ``'sub-001_ses-01_task_motorimagery_eeg.mat'``.
            data (np.ndarray): Trial array of shape
                ``(n_trials, 32, resampled_points)``, already resampled.
            labels (np.ndarray): Label array of shape ``(n_trials,)`` with
                values 1 (left hand) or 2 (right hand).
            chunk_size (int): Number of data points per chunk.
            overlap (int): Overlap between consecutive chunks.
            num_channel (int): Number of channels to keep.
            sfreq (int): Sampling rate (used for documentation).
            before_trial (Callable, optional): Pre-processing hook per trial.
            offline_transform (Callable, optional): Offline transform per chunk.

        Yields:
            dict: ``{'eeg': np.ndarray, 'key': str, 'info': dict}``
        '''
        size = _settings(chunk_size, overlap, num_channel, sfreq)
        identity = _identity(record)
        data, labels = _trials(data, labels, 4 * sfreq)
        write_pointer = 0
        for trial_index in range(len(data)):
            trial = data[trial_index, :num_channel].copy()
            if before_trial is not None:
                trial = np.asarray(before_trial(trial))
                if (trial.shape != (num_channel, 4 * sfreq) or trial.dtype.kind not in 'iuf'
                        or not np.isfinite(trial).all()):
                    raise ValueError('before_trial must preserve finite trial channels, samples and order.')
            for start_at in range(0, trial.shape[1] - size + 1, size - overlap):
                end_at = start_at + size
                clip_sample = trial[:, start_at:end_at].copy()
                if offline_transform is not None:
                    clip_sample = offline_transform(eeg=clip_sample)['eeg']
                clip_id = f"{identity['record_id']}_{write_pointer}"
                write_pointer += 1
                info = dict(identity, trial_id=f"{identity['record_id']}_trial-{trial_index}",
                            trial_index=trial_index, label=int(labels[trial_index]) - 1,
                            start_at=start_at, end_at=end_at, clip_id=clip_id,
                            sfreq=sfreq, signal_unit='uV',
                            channel_names=json.dumps(SHU_MI_CHANNEL_LIST[:num_channel]))
                yield {'eeg': clip_sample, 'key': clip_id, 'info': info}

    def set_records(self,
                    root_path: str = './SHU-MI',
                    **kwargs) -> List[str]:
        r'''
        Collect all ``.mat`` files from :obj:`root_path`.

        Args:
            root_path (str): Root directory of the dataset.

        Returns:
            list: Sorted list of ``.mat`` file names, e.g.
            ``['sub-001_ses-01_task_motorimagery_eeg.mat', ...]``.
        '''
        if not os.path.isdir(root_path):
            raise ValueError(f'root_path ({root_path}) must be a downloaded SHU-MI directory.')
        records = []
        for name in sorted(os.listdir(root_path)):
            if name.startswith('sub-') and name.endswith('_eeg.mat'):
                _identity(name)
                if not os.path.isfile(os.path.join(root_path, name)):
                    raise ValueError(f'Expected a trial MAT file: {name}.')
                records.append(name)
        if not records:
            raise ValueError('No SHU-MI trial MAT records were found.')
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
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'sfreq': self.sfreq,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size,
            })
