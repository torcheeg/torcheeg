import os
import re
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np

from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path

SELECTED_CHANNELS = [
    'EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8',
    'EEG T3', 'EEG T4', 'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6',
    'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG Fz', 'EEG Cz',
    'EEG Pz', 'EEG A2-A1',
]

DEFAULT_SAMPLING_RATE = 200


class MentalArithmeticDataset(BaseDataset):
    r'''
    The EEG During Mental Arithmetic Tasks (EEGMat) dataset contains resting
    and mental arithmetic recordings from 36 subjects. This reader selects
    scalp EEG channels, resamples to the requested rate, and caches windows
    using the standard TorchEEG dataset interface.

    - Author: Zyma et al.
    - Year: 2019
    - Download URL: https://physionet.org/content/eegmat/1.0.0/
    - Reference: Zyma I, Tukaev S, Seleznov I, et al. Electroencephalograms
      during Mental Arithmetic Task Performance. Data, 2019, 4(1): 14.
    - Data license: Open Data Commons Attribution License v1.0. Cite the
      dataset and PhysioNet when using the recordings.
    - Labels: 0 for the background recording (suffix ``_1``), and 1 for
      mental arithmetic (suffix ``_2``). These are recording conditions,
      not the count-quality labels in ``subject-info.csv``.

    Output signals are in microvolts. No additional filtering, amplitude
    scaling or rejection is applied. Use explicit transforms for a model's
    preprocessing protocol. The default uses 19 scalp channels at 200 Hz;
    ``num_channel=20`` also includes the recorded A2-A1 reference difference
    as the final channel. The reference channel has no scalp-grid position.
    Windows span each full recording in order. An incomplete final window
    is discarded; ``chunk_size=-1`` uses the whole recording. The
    ``before_trial`` hook runs once per recording and must preserve its shape.
    Overlapping windows may be correlated, so split subjects before fitting
    preprocessing or training for a cross-subject evaluation.
    Use a new :obj:`io_path` when changing source files, channels, sampling
    rate, window parameters or offline preprocessing. Regenerate caches
    from the earlier experimental implementation because channel defaults,
    window boundaries and recording IDs have changed.

    In order to use this dataset, the following file structure is required:

    .. code-block:: text

        MentalArithmetic/
        ├── Subject00_1.edf
        ├── Subject00_2.edf
        ├── Subject01_1.edf
        ├── Subject01_2.edf
        └── ...

    An example dataset with explicit signal and label transforms:

    .. code-block:: python

        from torcheeg.datasets import MentalArithmeticDataset
        from torcheeg import transforms

        dataset = MentalArithmeticDataset(
            root_path='./MentalArithmetic',
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Compose([
                transforms.Select('label'),
            ]))

        print(dataset[0])
        # EEG signal (torch.Tensor[19, 1000]),
        # label (int)

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import MentalArithmeticDataset
        from torcheeg import transforms

        dataset = MentalArithmeticDataset(
            root_path='./MentalArithmetic',
            online_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.To2d()
            ]),
            label_transform=transforms.Compose([
                transforms.Select('label'),
            ]))

        print(dataset[0])
        # EEG signal (torch.Tensor[1, 19, 1000]),
        # label (int)

    Args:
        root_path (str): Root path of the MentalArithmetic dataset.
            (default: :obj:`'./MentalArithmetic'`)
        chunk_size (int): Number of data points included in each EEG chunk
            as training or test samples. If set to -1, the EEG signal of a
            trial is used as a sample of a chunk. (default: :obj:`1000`,
            i.e. 5s at the default 200 Hz)
        overlap (int): The number of overlapping data points between
            different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels to keep in the documented order,
            from 1 to 20. The first 19 are scalp EEG; the twentieth is the
            reference difference. (default: :obj:`19`)
        sampling_rate (int): Output sampling rate in Hz.
            (default: :obj:`200`)
        online_transform (Callable, optional): The transformation of the
            EEG signals. The input is a :obj:`np.ndarray`, and the output
            is used as the first value of each element in the dataset.
            (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as
            :obj:`online_transform`, but executed before generating IO
            intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the
            label. The input is an information dictionary, and the output
            is used as the second value of each element in the dataset.
            (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial
            to which the sample belongs. It is performed before the offline
            transformation and thus typically used to implement
            context-dependent sample transformations, such as moving
            averages, etc. The input of this hook function is a 2D EEG
            signal with shape (number of electrodes, number of data
            points), whose ideal output shape is also (number of
            electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial
            to which the sample belongs. It is performed after the offline
            transformation and thus typically used to implement
            context-dependent sample transformations, such as moving
            averages, etc. The input and output of this hook function
            should be a sequence of dictionaries representing a sequence of
            EEG samples. Each dictionary contains two key-value pairs,
            indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key`
            (the index in the database) respectively.
        after_subject (Callable, optional): The hook performed on the
            subject to which the sample belongs. Similar to after_trial
            but operates on subject level.
        io_path (str): The path to generated unified data IO, cached as an
            intermediate result. If set to None, a random path will be
            generated. (default: :obj:`None`)
        io_size (int): Initial LMDB map size in bytes. The IO backend
            increases it when required.
            (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to
            :obj:`lmdb`, TorchEEG provides an efficient database (LMDB)
            for storing EEG signals. LMDB may not perform well on limited
            operating systems, where a file system based EEG signal
            storage is also provided. When io_mode is set to
            :obj:`pickle`, pickle-based persistence files are used. When
            io_mode is set to :obj:`memory`, memory are used.
            (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
            (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as
            progress bars, etc. (default: :obj:`True`)
    '''

    def __init__(self,
                 root_path: str = './MentalArithmetic',
                 chunk_size: int = 1000,
                 overlap: int = 0,
                 num_channel: int = 19,
                 sampling_rate: int = DEFAULT_SAMPLING_RATE,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True):
        self._validate_parameters(chunk_size, overlap, num_channel, sampling_rate)
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'sampling_rate': sampling_rate,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
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
    def _validate_parameters(chunk_size, overlap, num_channel, sampling_rate):
        for name, value in (('chunk_size', chunk_size), ('overlap', overlap),
                            ('num_channel', num_channel), ('sampling_rate', sampling_rate)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{name} must be an integer.')
        if chunk_size != -1 and chunk_size <= 0:
            raise ValueError('chunk_size must be positive or -1 for a whole recording.')
        if overlap < 0 or (chunk_size == -1 and overlap != 0) or (chunk_size > 0 and overlap >= chunk_size):
            raise ValueError('overlap must be nonnegative and smaller than chunk_size; use zero for a whole recording.')
        if not 1 <= num_channel <= len(SELECTED_CHANNELS):
            raise ValueError('num_channel must be between 1 and 20.')
        if sampling_rate <= 0:
            raise ValueError('sampling_rate must be positive.')

    @staticmethod
    def read_record(record: Tuple, num_channel: int = 19,
                    sampling_rate: int = DEFAULT_SAMPLING_RATE, **kwargs) -> Dict:
        r'''Read ordered EEG channels from EDF and return microvolt signals.'''
        MentalArithmeticDataset._validate_parameters(1, 0, num_channel, sampling_rate)
        edf_path, label, _ = record
        channels = SELECTED_CHANNELS[:num_channel]
        with mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR') as raw:
            missing = set(channels) - set(raw.ch_names)
            if missing:
                raise ValueError(f'Missing MentalArithmetic channels: {sorted(missing)}.')
            raw.pick(channels)
            raw.reorder_channels(channels)
            if raw.info['sfreq'] != sampling_rate:
                raw.resample(sampling_rate)
            eeg_data = raw.get_data(units='uV')
        return {'eeg_data': eeg_data, 'label': label}

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        r'''Generate a deterministic synthetic record without changing global RNG.'''
        return {'eeg_data': np.random.RandomState(42).normal(size=(20, 2400)),
                'label': 0, 'record': ('/fake/Subject00_1.edf', 0, 'Subject00')}

    @staticmethod
    def process_record(record: Tuple,
                       eeg_data: np.ndarray,
                       label: int,
                       chunk_size: int = 1000,
                       overlap: int = 0,
                       num_channel: int = 19,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        r'''Yield windows from one complete recording, without implicit scaling.'''
        MentalArithmeticDataset._validate_parameters(chunk_size, overlap, num_channel, 1)
        edf_path, record_label, subject_id = record
        if label not in (0, 1) or label != record_label:
            raise ValueError('Recording labels must match the session condition, 0 or 1.')
        eeg_data = np.asarray(eeg_data)
        if (eeg_data.ndim != 2 or eeg_data.shape[0] < num_channel
                or eeg_data.shape[1] == 0 or not np.isrealobj(eeg_data)
                or not np.issubdtype(eeg_data.dtype, np.number)
                or not np.isfinite(eeg_data).all()):
            raise ValueError('eeg_data must contain finite real channels and time samples.')
        trial_samples = eeg_data[:num_channel]
        if before_trial is not None:
            transformed = np.asarray(before_trial(trial_samples.copy()))
            if transformed.shape != trial_samples.shape:
                raise ValueError('before_trial must preserve the recording shape.')
            if (not np.isrealobj(transformed)
                    or not np.issubdtype(transformed.dtype, np.number)
                    or not np.isfinite(transformed).all()):
                raise ValueError('before_trial must return finite real signals.')
            trial_samples = transformed
        dynamic_chunk_size = trial_samples.shape[1] if chunk_size == -1 else chunk_size
        step = dynamic_chunk_size - overlap
        record_name = os.path.splitext(os.path.basename(edf_path))[0]
        for write_pointer, start_at in enumerate(range(
                0, trial_samples.shape[1] - dynamic_chunk_size + 1, step)):
            end_at = start_at + dynamic_chunk_size
            clip_sample = trial_samples[:, start_at:end_at].copy()
            if offline_transform is not None:
                clip_sample = offline_transform(eeg=clip_sample)['eeg']
            clip_id = f'{record_name}_{write_pointer}'
            info = {'subject_id': subject_id, 'trial_id': record_label + 1,
                    'label': label, 'clip_id': clip_id,
                    'start_at': start_at, 'end_at': end_at}
            yield {'eeg': clip_sample, 'key': clip_id, 'info': info}

    def set_records(self, root_path: str = './MentalArithmetic', **kwargs) -> List[Tuple]:
        r'''Collect session EDF files in deterministic filename order.'''
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f'MentalArithmetic root_path does not exist: {root_path}.')
        records = []
        for filename in sorted(os.listdir(root_path)):
            path = os.path.join(root_path, filename)
            if not os.path.isfile(path) or not filename.lower().endswith('.edf'):
                continue
            match = re.fullmatch(r'(Subject[0-9]+)_([12])\.edf', filename, flags=re.IGNORECASE)
            if match is None:
                raise ValueError(f'Unrecognized MentalArithmetic EDF filename: {filename}.')
            records.append((path, int(match.group(2)) - 1, match.group(1)))
        if not records:
            raise ValueError('No MentalArithmetic session EDF files were found.')
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
                'sampling_rate': self.sampling_rate,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size,
            })
