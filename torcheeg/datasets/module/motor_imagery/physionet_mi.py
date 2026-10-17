import os
import re
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np

from ...constants.motor_imagery.physionet_mi import PHYSIONET_MI_EDF_CHANNEL_LIST
from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path

mne.set_log_level('CRITICAL')

MI_TASKS = ['04', '06', '08', '10', '12', '14']

LEFT_RIGHT_TASKS = {'04', '08', '12'}
BOTH_TASKS = {'06', '10', '14'}

TASK_LABEL_MAP = {
    'four_class': {0: 0, 1: 1, 2: 2, 3: 3},
    'left_right': {0: 0, 1: 1},
    'hands_feet': {0: 0, 1: 0, 2: 0, 3: 1},
}


class PhysioNetMIDataset(BaseDataset):
    r'''
    The PhysioNet EEG Motor Movement/Imagery Dataset was collected using the BCI2000 system. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). Raw signals are average-referenced, high-pass filtered at 0.3Hz, notch filtered at 60Hz, and resampled from 160Hz to :obj:`sfreq` (default 200Hz) via MNE. Four-second epochs start at each motor imagery cue and are segmented into chunks. Returned signals use microvolts in the published EDF channel order. This fixed-window protocol is not a claim of reproducing a specific benchmark. The relevant information of the dataset is as follows:

    - Author: Schalk et al.
    - Year: 2009
    - Download URL: https://physionet.org/content/eegmmidb/1.0.0/
    - Reference: Schalk G, McFarland D J, Hinterberger T, et al. BCI2000: a general-purpose brain-computer interface (BCI) system[J]. IEEE Transactions on biomedical engineering, 2004, 51(6): 1034-1043.
    - Stimulus: Subjects performed motor movement and motor imagery tasks with visual cues.
    - Signals: Electroencephalogram (64 channels at 160Hz) from 109 subjects. Each subject completed 14 runs: 2 baseline runs (eyes open/closed) and 12 task runs. Only the 6 motor imagery runs (R04, R06, R08, R10, R12, R14) are used.
    - Rating: left fist (0), right fist (1), both fists (2), both feet (3).

    In order to use this dataset, the download folder is required, containing subject directories:

    .. code-block:: text

        PhysioNet-MI/
        ├── S001/
        │   ├── S001R01.edf
        │   ├── S001R02.edf
        │   ├── ...
        │   └── S001R14.edf
        ├── S002/
        └── ...

    An example dataset with the default preprocessing (200Hz, 64 channels, 1s chunks):

    .. code-block:: python

        from torcheeg.datasets import PhysioNetMIDataset
        from torcheeg import transforms

        dataset = PhysioNetMIDataset(root_path='./PhysioNet-MI',
                                     online_transform=transforms.ToTensor(),
                                     label_transform=transforms.Compose([
                                         transforms.Select('label'),
                                     ]))

        print(dataset[0])
        # EEG signal (torch.Tensor[64, 200]),
        # label (int)

    Channel names follow :obj:`PHYSIONET_MI_CHANNEL_LIST`; the matching EDF
    labels, grid coordinates and standard-distance graph are available from
    :mod:`torcheeg.datasets.constants` as :obj:`PHYSIONET_MI_EDF_CHANNEL_LIST`,
    :obj:`PHYSIONET_MI_CHANNEL_LOCATION_DICT` and
    :obj:`PHYSIONET_MI_ADJACENCY_MATRIX`. The first :obj:`num_channel` channels
    are retained in that order. ``trial_id`` includes the run ID so windows
    from different runs cannot share a trial identifier within one subject.
    Select a new :obj:`io_path` when changing :obj:`task`; a cache from another
    task is rejected. Legacy caches without a task column are accepted only
    for :obj:`four_class`.
    Keep all windows of a trial together when partitioning; use subject-wise
    partitions for cross-subject evaluation.

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import PhysioNetMIDataset
        from torcheeg import transforms

        dataset = PhysioNetMIDataset(root_path='./PhysioNet-MI',
                                     online_transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.To2d()
                                     ]),
                                     label_transform=transforms.Compose([
                                         transforms.Select('label'),
                                     ]))

        print(dataset[0])
        # EEG signal (torch.Tensor[1, 64, 200]),
        # label (int)

    Args:
        task (str): Task label mapping. Use :obj:`four_class` for left fist (0), right fist (1), both fists (2) and feet (3); :obj:`left_right` for left fist (0) and right fist (1), excluding other runs; or :obj:`hands_feet` for any fist imagery (0) versus feet (1). The instance's :obj:`num_classes` is 4 or 2 accordingly. (default: :obj:`four_class`)
        root_path (str): Downloaded data directory containing subject folders (e.g., S001, S002, ...). (default: :obj:`'./PhysioNet-MI'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`200`, i.e. 1s at 200Hz)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. (default: :obj:`64`)
        sfreq (int): Target sampling rate in Hz for the output EEG signals. The original sampling rate is 160Hz. (default: :obj:`200`)
        l_freq (float): Low cutoff frequency for high-pass filter. (default: :obj:`0.3`)
        notch_freq (float): Notch filter frequency for power-line noise removal. (default: :obj:`60.0`)
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

    ORIGINAL_SFREQ = 160
    NUM_CLASSES = 4

    def __init__(self,
                 root_path: str = './PhysioNet-MI',
                 chunk_size: int = 200,
                 overlap: int = 0,
                 num_channel: int = 64,
                 sfreq: int = 200,
                 l_freq: float = 0.3,
                 notch_freq: float = 60.0,
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
                 task: str = 'four_class'):
        self._validate_task(task)
        self.num_classes = len(set(TASK_LABEL_MAP[task].values()))
        self._validate_parameters(chunk_size, overlap, num_channel, sfreq)
        if not np.isfinite(l_freq) or not 0 < l_freq < 80:
            raise ValueError('l_freq must be between 0 and 80 Hz.')
        if not np.isfinite(notch_freq) or not 0 < notch_freq < 80:
            raise ValueError('notch_freq must be between 0 and 80 Hz.')
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        params = {
            'root_path': root_path,
            'task': task,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'sfreq': sfreq,
            'l_freq': l_freq,
            'notch_freq': notch_freq,
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
        if 'task' not in self.info:
            if task != 'four_class':
                raise ValueError('This legacy cache uses four_class; select a new io_path for another task.')
        elif set(self.info['task']) != {task}:
            raise ValueError('The cached imagery task differs; select a new io_path.')

    @staticmethod
    def _validate_task(task):
        if not isinstance(task, str) or task not in TASK_LABEL_MAP:
            raise ValueError('task must be four_class, left_right or hands_feet.')

    @staticmethod
    def _validate_parameters(chunk_size, overlap, num_channel, sfreq):
        for name, value in [('chunk_size', chunk_size), ('overlap', overlap),
                            ('num_channel', num_channel), ('sfreq', sfreq)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{name} must be an integer.')
        if sfreq <= 0 or not 1 <= num_channel <= 64:
            raise ValueError('sfreq must be positive and num_channel in [1, 64].')
        window = chunk_size if chunk_size > 0 else 4 * sfreq
        if window > 4 * sfreq:
            raise ValueError('chunk_size cannot exceed the four-second epoch.')
        if not 0 <= overlap < window:
            raise ValueError('overlap must be nonnegative and smaller than the window.')

    @staticmethod
    def read_record(record: str,
                    root_path: str = './PhysioNet-MI',
                    sfreq: int = 200,
                    l_freq: float = 0.3,
                    notch_freq: float = 60.0,
                    **kwargs) -> Dict:
        r'''
        Read a single EDF file for one run of one subject, select channels,
        apply average reference, high-pass and notch filters, resample, and
        extract motor imagery epochs.

        Args:
            record (str): Relative path from :obj:`root_path` to the ``.edf``
                file, e.g. ``'S001/S001R04.edf'``.
            root_path (str): Root directory of the dataset.
            sfreq (int): Target sampling rate in Hz.
            l_freq (float): Low cutoff frequency for high-pass filter.
            notch_freq (float): Notch filter frequency.

        Returns:
            dict: A dictionary containing ``'epochs_data'`` in µV with shape
            ``(n_epochs, 64, epoch_samples)``, ``'events'`` with event codes,
            and ``'task_id'`` as a string.
        '''
        file_name = os.path.basename(record)
        match = re.fullmatch(r'(S[0-9]{3})R([0-9]{2})\.edf', file_name)
        if match is None or match.group(2) not in MI_TASKS:
            raise ValueError('record must identify a motor imagery run.')
        file_path = os.path.join(root_path, record)

        raw = mne.io.read_raw_edf(file_path, preload=True)
        raw.pick(PHYSIONET_MI_EDF_CHANNEL_LIST)

        if len(raw.info['bads']) > 0:
            raw.interpolate_bads()

        raw.set_eeg_reference(ref_channels='average')
        raw.filter(l_freq=l_freq, h_freq=None)
        raw.notch_filter(notch_freq)
        raw.resample(sfreq)

        events_from_annot, _ = mne.events_from_annotations(
            raw, event_id={'T0': 1, 'T1': 2, 'T2': 3})
        epochs = mne.Epochs(
            raw, events_from_annot, event_id={'T1': 2, 'T2': 3},
            tmin=0, tmax=4.0 - 1.0 / raw.info['sfreq'],
            baseline=None, preload=True, on_missing='ignore',
        )

        epochs_data = epochs.get_data(units='uV')
        if not len(epochs_data):
            raise ValueError('No complete four-second motor imagery epochs were found.')
        events = epochs.events[:, 2]

        file_name = os.path.basename(record)
        task_id = file_name[-6:-4]

        return {
            'epochs_data': epochs_data,
            'events': events,
            'task_id': task_id,
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        r'''
        Generate a fake record for testing purposes.

        Returns:
            dict: A dictionary containing random ``'epochs_data'`` with shape
            ``(15, 64, 800)``, dummy ``'events'``, and ``'task_id'``.
        '''
        sfreq = kwargs.get('sfreq', 200)
        n_epochs = 15
        epoch_samples = 4 * sfreq

        epochs_data = np.random.randn(n_epochs, 64, epoch_samples)
        events = np.array([1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1])

        return {
            'epochs_data': epochs_data,
            'events': events,
            'task_id': '04',
            'record': 'S001/S001R04.edf',
        }

    @staticmethod
    def process_record(record: str,
                       epochs_data: np.ndarray,
                       events: np.ndarray,
                       task_id: str,
                       chunk_size: int = 200,
                       overlap: int = 0,
                       num_channel: int = 64,
                       sfreq: int = 200,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       task: str = 'four_class',
                       **kwargs):
        r'''
        Process a run's epoch data into chunks and yield sample dictionaries.
        Rest epochs (T0, event=1) are excluded. For left/right fist runs
        (R04, R08, R12), labels are T1→0, T2→1. For both fists/feet runs
        (R06, R10, R14), labels are T1→2, T2→3.

        :obj:`left_right` excludes both-fists/feet runs. :obj:`hands_feet`
        merges left, right and both-fists labels into 0 and maps feet to 1.
        These mappings match the corresponding EEG-Arena registry tasks;
        preprocessing and evaluation protocols must still be specified.

        Only the last ``4 * sfreq`` samples of each
        epoch are used, then segmented into non-overlapping chunks.

        Args:
            record (str): Relative path to the ``.edf`` file.
            epochs_data (np.ndarray): Epoch array of shape
                ``(n_epochs, 64, epoch_samples)``, already preprocessed.
            events (np.ndarray): Event codes for each epoch (1=T0, 2=T1, 3=T2).
            task_id (str): Run identifier (``'04'``, ``'06'``, etc.).
            chunk_size (int): Number of data points per chunk.
            overlap (int): Overlap between consecutive chunks.
            num_channel (int): Number of channels to keep.
            sfreq (int): Sampling rate (used for trimming).
            before_trial (Callable, optional): Pre-processing hook per trial.
            offline_transform (Callable, optional): Offline transform per chunk.
            task (str): One of :obj:`four_class`, :obj:`left_right` or :obj:`hands_feet`.

        Yields:
            dict: ``{'eeg': np.ndarray, 'key': str, 'info': dict}``
        '''
        PhysioNetMIDataset._validate_task(task)
        PhysioNetMIDataset._validate_parameters(
            chunk_size, overlap, num_channel, sfreq)
        if task_id not in MI_TASKS:
            raise ValueError('task_id must identify a motor imagery run.')
        epochs_data = np.asarray(epochs_data)
        events = np.asarray(events)
        if epochs_data.ndim != 3 or epochs_data.shape[1] < num_channel:
            raise ValueError('epochs_data must contain the requested channels.')
        if events.ndim != 1 or len(events) != len(epochs_data):
            raise ValueError('events must contain one code per epoch.')
        if not np.isin(events, [1, 2, 3]).all():
            raise ValueError('events must use T0=1, T1=2 and T2=3.')
        if epochs_data.shape[2] < 4 * sfreq:
            raise ValueError('epochs_data must contain four seconds per epoch.')
        file_name = os.path.basename(record)
        subject_id = file_name[:4]

        epoch_len = 4 * sfreq

        write_pointer = 0

        for epoch_idx in range(epochs_data.shape[0]):
            event = int(events[epoch_idx])

            if event == 1:
                continue

            if task_id in LEFT_RIGHT_TASKS:
                label = event - 2
            else:
                label = event

            if label not in TASK_LABEL_MAP[task]:
                continue
            label = TASK_LABEL_MAP[task][label]

            trial_samples = epochs_data[
                epoch_idx, :num_channel, -epoch_len:
            ]

            if before_trial is not None:
                trial_samples = before_trial(trial_samples)
                if (not isinstance(trial_samples, np.ndarray) or
                        trial_samples.shape != (num_channel, epoch_len)):
                    raise ValueError('before_trial must preserve the trial shape.')

            trial_meta_info = {
                'subject_id': subject_id,
                'task': task,
                'task_id': task_id,
                'session_id': task_id,
                'trial_id': f'{task_id}_{epoch_idx}',
                'label': label,
            }

            if chunk_size <= 0:
                dynamic_chunk_size = trial_samples.shape[1]
            else:
                dynamic_chunk_size = chunk_size

            start_at = 0
            end_at = start_at + dynamic_chunk_size
            step = dynamic_chunk_size - overlap

            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:, start_at:end_at]

                t_eeg = clip_sample
                if offline_transform is not None:
                    t_eeg = offline_transform(eeg=clip_sample)['eeg']

                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1

                record_info = {
                    'start_at': start_at,
                    'end_at': end_at,
                    'clip_id': clip_id,
                }
                record_info.update(trial_meta_info)

                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size

    def set_records(self,
                    root_path: str = './PhysioNet-MI',
                    task: str = 'four_class',
                    **kwargs) -> List[str]:
        r'''
        Collect all motor imagery EDF files from :obj:`root_path`. Only the
        6 motor imagery runs (R04, R06, R08, R10, R12, R14) are included.
        For :obj:`left_right`, only R04, R08 and R12 are selected.

        Args:
            root_path (str): Root directory of the dataset.
            task (str): The imagery task to include.

        Returns:
            list: Sorted list of relative paths to ``.edf`` files,
            e.g. ``['S001/S001R04.edf', 'S001/S001R06.edf', ...]``.
        '''
        assert os.path.exists(root_path), \
            (f'root_path ({root_path}) does not exist. '
             f'Please download the dataset and set the root_path to the '
             f'downloaded path.')

        self._validate_task(task)
        runs = sorted(LEFT_RIGHT_TASKS) if task == 'left_right' else MI_TASKS
        file_list = []
        for subject_dir in sorted(os.listdir(root_path)):
            subject_path = os.path.join(root_path, subject_dir)
            if (re.fullmatch(r'S[0-9]{3}', subject_dir) is None or
                    not os.path.isdir(subject_path)):
                continue
            for run in runs:
                edf_name = f'{subject_dir}R{run}.edf'
                edf_path = os.path.join(subject_path, edf_name)
                if os.path.exists(edf_path):
                    file_list.append(os.path.join(subject_dir, edf_name))

        if not file_list:
            raise ValueError(
                'No motor imagery EDF runs were found under root_path.')
        return file_list

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
                'task': self.task,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'sfreq': self.sfreq,
                'l_freq': self.l_freq,
                'notch_freq': self.notch_freq,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size,
            })
