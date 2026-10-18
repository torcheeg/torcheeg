import os
import json
import re
from numbers import Integral, Real
from typing import Callable, Dict, Tuple, Union
import mne
import numpy as np
import scipy.io as scio

from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path

CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
            'P2', 'POz', 'EOG1', 'EOG2', 'EOG3']


def _identity(record):
    name = os.path.basename(os.fspath(record))
    match = re.fullmatch(r'A(0[1-9])([TE])\.mat', name)
    if match is None:
        raise ValueError('Expected A01T.mat through A09E.mat.')
    return name[:-4], 'A' + match.group(1), match.group(2)


def _settings(offset, chunk_size, overlap, num_channel, sfreq, offset_start,
              offset_end, notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None):
    start = offset if offset_start is None else offset_start
    end = 1750 if offset_end is None else offset_end
    for value in (offset, chunk_size, overlap, num_channel, sfreq, start, end):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError('Offsets, window sizes, channels and sampling rate must be integers.')
    if offset_start is not None and offset not in (0, offset_start):
        raise ValueError('offset and offset_start must agree when both are provided.')
    if not 0 <= offset <= 1750 or not 0 <= start < end <= 1750 or not 1 <= num_channel <= 25 or sfreq <= 0:
        raise ValueError('Require source offsets within seven seconds, 1--25 channels and a positive output rate.')
    first, last = round(start * sfreq / 250), round(end * sfreq / 250)
    size = last - first if chunk_size <= 0 else chunk_size
    if not 0 < size <= last - first or not 0 <= overlap < size:
        raise ValueError('Chunks must fit the selected output window and overlap must be smaller than a chunk.')
    for value in (notch_freqs, l_freq, h_freq):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or not 0 < value < 125):
            raise ValueError('Filter frequencies must be finite, positive and below the source Nyquist.')
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError('l_freq must be smaller than h_freq.')
    if ref_channels not in (None, 'average'):
        raise ValueError('ref_channels must be None or average.')
    return first, last, size


def _vector(value, count, allowed, name):
    array = np.asarray(value)
    if (array.dtype.kind not in 'iuf' or array.ndim > 2
            or (array.ndim == 2 and 1 not in array.shape)
            or array.size != count or not np.isin(array, allowed).all()):
        raise ValueError(f'{name} must contain one valid value per trial.')
    return array.reshape(-1).astype(int)


class BCICIV2aDataset(BaseDataset):
    r'''
    A dataset for motor imagery, BCI Competition 2008 Graz data set A (BCICIV_2a). This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Tangermann et al.
    - Year: 2012
    - Download URL: http://bnci-horizon-2020.eu/database/data-sets
    - Reference: Tangermann M, Müller K R, Aertsen A, et al. Review of the BCI competition IV[J]. Frontiers in neuroscience, 2012: 55.
    - Signals: Electroencephalogram (22 channels at 250Hz) and electrooculogram (3 channels at 250Hz) of 9 subjects. Two sessions on different days were recorded for each subject. Each session is comprised of 6 runs separated by short breaks. One run consists of 48 trials (12 for each of the four possible classes), yielding a total of 288 trials per session. For more detail, please refer to https://www.bbci.de/competition/iv/desc_2a.pdf
    - Category: The imagination of movement of the left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4).

    In order to use this dataset, the download folder :obj:`BCICIV_2a_mat` is required, containing the following files:

    .. code-block:: text

        BCICIV_2a_mat/
        ├── A01T.mat
        ├── A01E.mat
        ├── ...
        └── A09E.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import BCICIV2aDataset
        from torcheeg import transforms

        dataset = BCICIV2aDataset(root_path='./BCICIV_2a_mat',
                                  online_transform=transforms.Compose([
                                      transforms.To2d(),
                                      transforms.ToTensor()
                                  ]),
                                  label_transform=transforms.Compose([
                                      transforms.Select('label'),
                                      transforms.Lambda(lambda x: x - 1)
                                  ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 22, 1750])
        # label (int)

    Native outputs are microvolts. Trial onsets in the BNCI MAT files are
    one-based and are converted once to zero-based source samples. Default
    samples retain 22 EEG channels and 1750 samples at 250 Hz. See
    :doc:`/user_guide/bciciv2a` for preprocessing and cache migration.

    Args:
        sfreq (int): Output sampling rate in Hz. (default: :obj:`250`)
        notch_freqs (float, optional): One notch frequency in Hz, or None.
        l_freq (float, optional): High-pass cutoff in Hz, or None.
        h_freq (float, optional): Low-pass cutoff in Hz, or None.
        ref_channels (str, optional): None or 'average'; only EEG channels enter the reference.
        offset_start (int, optional): Source start offset at 250 Hz. Defaults to offset.
        offset_end (int, optional): Exclusive source end offset at 250 Hz. Defaults to 1750.
        root_path (str): Downloaded data files in matlab (unzipped BCICIV_2a_mat.zip) formats (default: :obj:`'./BCICIV_2a_mat'`)
        offset (int): Source samples at 250 Hz to discard from each trial. Alias for offset_start. (default: :obj:`0`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. Nonpositive values use the complete selected window; positive values count output samples. (default: :obj:`7 * 250`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 22 channels are EEG signals. (default: :obj:`22`)
        skip_trial_with_artifacts (bool): Whether to skip trials with artifacts. (default: :obj:`False`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): For compatibility, this hook operates once on each continuous motor-imagery run before trial extraction. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    '''

    def __init__(self,
                 root_path: str = './BCICIV_2a_mat',
                 offset: int = 0,
                 chunk_size: int = 7 * 250,
                 overlap: int = 0,
                 num_channel: int = 22,
                 skip_trial_with_artifacts: bool = False,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 sfreq: int = 250,
                 notch_freqs: Union[float, None] = None,
                 l_freq: Union[float, None] = None,
                 h_freq: Union[float, None] = None,
                 ref_channels: Union[str, None] = None,
                 offset_start: Union[int, None] = None,
                 offset_end: Union[int, None] = None):
        _settings(offset, chunk_size, overlap, num_channel, sfreq, offset_start,
                  offset_end, notch_freqs, l_freq, h_freq, ref_channels)
        if not isinstance(skip_trial_with_artifacts, bool):
            raise ValueError('skip_trial_with_artifacts must be boolean.')
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'offset': offset,
            'sfreq': sfreq,
            'notch_freqs': notch_freqs,
            'l_freq': l_freq,
            'h_freq': h_freq,
            'ref_channels': ref_channels,
            'offset_start': offset_start,
            'offset_end': offset_end,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'skip_trial_with_artifacts': skip_trial_with_artifacts,
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
            'verbose': verbose
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def read_record(record: str, sfreq: int = 250, num_channel: int = 22,
                    notch_freqs=None, l_freq=None, h_freq=None, ref_channels=None,
                    **kwargs) -> Dict:
        _identity(record)
        _settings(0, -1, 0, num_channel, sfreq, None, None,
                  notch_freqs, l_freq, h_freq, ref_channels)
        data = scio.loadmat(record, simplify_cells=True)
        if 'data' not in data:
            raise ValueError('The BNCI MAT file must contain data runs.')
        runs = data['data']
        runs = [runs] if isinstance(runs, dict) else list(np.asarray(runs, dtype=object).reshape(-1))
        result = []
        for run in runs:
            if not isinstance(run, dict) or not {'X', 'trial', 'y', 'fs', 'artifacts'} <= run.keys():
                raise ValueError('Each run needs X, trial, y, fs and artifacts.')
            values = np.asarray(run['X'])
            if (run['fs'] != 250 or values.ndim != 2 or values.shape[1] != 25
                    or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()):
                raise ValueError('Require finite 25-channel BNCI runs sampled at 250 Hz.')
            onsets = np.asarray(run['trial'])
            if (onsets.dtype.kind not in 'iuf' or onsets.ndim > 2
                    or (onsets.ndim == 2 and 1 not in onsets.shape)
                    or not np.isfinite(onsets).all() or np.any(onsets != np.floor(onsets))):
                raise ValueError('Trial onsets must be integer-valued vectors.')
            onsets = onsets.reshape(-1).astype(int) - 1
            labels = _vector(run['y'], len(onsets), [1, 2, 3, 4], 'y')
            artifacts = _vector(run['artifacts'], len(onsets), [0, 1], 'artifacts')
            if len(onsets) and (np.any(onsets < 0) or np.any(np.diff(onsets) < 1750)
                                or onsets[-1] + 1750 > len(values)):
                raise ValueError('Seven-second trials must be ordered, complete and nonoverlapping.')
            selected = values[:, :num_channel].T.copy()
            # EOG calibration runs contain no motor-imagery trials. Keep their
            # position in the run list without applying trial preprocessing.
            if len(onsets) and (sfreq != 250 or any(value is not None for value in
                    (notch_freqs, l_freq, h_freq, ref_channels))):
                raw = mne.io.RawArray(selected.astype(np.float64) * 1e-6,
                    mne.create_info(CHANNELS[:num_channel], 250,
                                    (['eeg'] * 22 + ['eog'] * 3)[:num_channel]), verbose=False)
                if notch_freqs is not None:
                    raw.notch_filter(notch_freqs, picks=list(range(num_channel)), verbose=False)
                if l_freq is not None or h_freq is not None:
                    raw.filter(l_freq, h_freq, picks=list(range(num_channel)), verbose=False)
                if ref_channels is not None:
                    raw.set_eeg_reference(ref_channels, verbose=False)
                if sfreq != 250:
                    raw.resample(sfreq, verbose=False)
                selected = raw.get_data() * 1e6
            result.append(dict(X=selected, trial=onsets, y=labels, artifacts=artifacts))
        if not any(len(run['trial']) for run in result):
            raise ValueError('The MAT file contains no motor-imagery trials.')
        return dict(a_data=result)

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        rate = kwargs.get('sfreq', 250)
        channels = kwargs.get('num_channel', 22)
        generator = np.random.default_rng(42)
        return dict(record='A01T.mat', a_data=[dict(X=generator.standard_normal((channels, 24 * rate)),
                    trial=np.array([0, 8 * 250, 16 * 250]), y=np.array([1, 2, 4]),
                    artifacts=np.array([0, 1, 0]))])

    @staticmethod
    def process_record(record: str, a_data, offset: int = 0,
                       chunk_size: int = 7 * 250, overlap: int = 0,
                       num_channel: int = 22, skip_trial_with_artifacts: bool = False,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       sfreq: int = 250, offset_start=None, offset_end=None, **kwargs):
        first, last, size = _settings(offset, chunk_size, overlap, num_channel, sfreq,
                                     offset_start, offset_end)
        if not isinstance(skip_trial_with_artifacts, bool):
            raise ValueError('skip_trial_with_artifacts must be boolean.')
        filename, subject, session = _identity(record)
        pointer = 0
        for run_id, run in enumerate(a_data):
            if not len(run['trial']):
                continue
            values = np.asarray(run['X']).copy()
            if before_trial is not None:
                values = np.asarray(before_trial(values))
            if (values.shape != np.shape(run['X']) or values.ndim != 2
                    or values.shape[0] != num_channel or values.dtype.kind not in 'iuf'
                    or not np.isfinite(values).all()):
                raise ValueError('before_trial must preserve finite continuous-run geometry.')
            for trial_index, onset in enumerate(run['trial']):
                if skip_trial_with_artifacts and run['artifacts'][trial_index]:
                    continue
                output_onset = int(round(int(onset) * sfreq / 250))
                if output_onset + 7 * sfreq > values.shape[1]:
                    raise ValueError('Resampled seven-second trial is incomplete.')
                trial_id = f'{filename}_r{run_id}_t{trial_index}'
                for start in range(first, last - size + 1, size - overlap):
                    end = start + size
                    clip = values[:, output_onset + start:output_onset + end].copy()
                    if offline_transform is not None:
                        clip = offline_transform(eeg=clip)['eeg']
                    key = f'{filename}_{pointer}'
                    pointer += 1
                    info = dict(subject_id=subject, subject=subject, session=session,
                                phase='train' if session == 'T' else 'test', run=run_id,
                                record_id=filename, trial_id=trial_id, trial_index=trial_index,
                                label=int(run['y'][trial_index]), artifact=int(run['artifacts'][trial_index]),
                                clip_id=key, start_at=output_onset + start, end_at=output_onset + end,
                                source_trial_onset=int(onset), sfreq=sfreq, signal_unit='uV',
                                source_start_seconds=(output_onset + start) / sfreq,
                                source_end_seconds=(output_onset + end) / sfreq,
                                channel_names=json.dumps(CHANNELS[:num_channel]))
                    yield dict(eeg=clip, key=key, info=info)

    def set_records(self, root_path: str = './BCICIV_2a_mat', **kwargs):
        if not os.path.isdir(root_path):
            raise ValueError('root_path must be a downloaded BNCI MAT directory.')
        records = []
        for name in sorted(os.listdir(root_path)):
            if name.startswith('A') and name.endswith('.mat'):
                _identity(name)
                records.append(os.path.join(root_path, name))
        if not records:
            raise ValueError('No A01T.mat through A09E.mat files were found.')
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
                'offset': self.offset,
                'offset_start': self.offset_start,
                'offset_end': self.offset_end,
                'sfreq': self.sfreq,
                'notch_freqs': self.notch_freqs,
                'l_freq': self.l_freq,
                'h_freq': self.h_freq,
                'ref_channels': self.ref_channels,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'skip_trial_with_artifacts': self.skip_trial_with_artifacts,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'after_session': self.after_session,
                'after_subject': self.after_subject,
                'io_path': self.io_path,
                'io_size': self.io_size,
                'io_mode': self.io_mode,
                'num_worker': self.num_worker,
                'verbose': self.verbose
            })
