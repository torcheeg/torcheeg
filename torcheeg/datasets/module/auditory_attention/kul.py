import os
from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy.io import loadmat
from scipy.signal import filtfilt, firwin

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


def apply_fir_filter(eeg_data: np.ndarray, lowpass: float, highpass: float, sampling_rate: int) -> np.ndarray:
    """
    Apply bandpass FIR filter to EEG data.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_samples, n_channels)
        lowpass (float): Low-pass cutoff frequency in Hz
        highpass (float): High-pass cutoff frequency in Hz
        sampling_rate (int): Sampling rate in Hz

    Returns:
        np.ndarray: Filtered EEG data
    """
    numtaps = int(3 * sampling_rate / highpass)
    if numtaps % 2 == 0:
        numtaps += 1

    fir_coeff = firwin(
        numtaps,
        [highpass, lowpass],
        pass_zero=False,
        fs=sampling_rate,
        window='hamming'
    )

    filtered_data = filtfilt(fir_coeff, 1, eeg_data.astype(np.float64), axis=0)

    return filtered_data


def apply_rereference(eeg_data: np.ndarray, method: Union[None, str] = 'Cz') -> np.ndarray:
    """
    Rereference EEG data.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_samples, n_channels)
        method (str, optional): Rereference method ('Cz', 'mean', or None)

    Returns:
        np.ndarray: Rereferenced EEG data
    """
    if method is None:
        return eeg_data

    eeg_data = eeg_data.copy()

    if method.lower() == 'cz':
        cz_idx = 47
        reference = eeg_data[:, cz_idx:cz_idx+1]
        eeg_data = eeg_data - reference

    elif method.lower() == 'mean':
        reference = np.mean(eeg_data, axis=1, keepdims=True)
        eeg_data = eeg_data - reference

    return eeg_data


def apply_downsample(eeg_data: np.ndarray, original_sr: float, target_sr: float) -> np.ndarray:
    """
    Downsample EEG data using simple downsampling (no filtering).

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_samples, n_channels)
        original_sr (float): Original sampling rate
        target_sr (float): Target sampling rate

    Returns:
        np.ndarray: Downsampled EEG data
    """
    downsample_factor = int(original_sr / target_sr)

    if downsample_factor != original_sr / target_sr:
        raise ValueError(
            f"Downsample factor must be integer, got {original_sr / target_sr}")

    downsampled_data = eeg_data[::downsample_factor, :]

    return downsampled_data


class KULDataset(BaseDataset):
    r'''
    This dataset contains EEG recordings from 16 subjects listening to one of two competing speech audio streams. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Biesmans et al.
    - Year: 2016
    - Download URL: https://zenodo.org/records/4004271
    - Reference: Biesmans W, Das N, Francart T, et al. Auditory-inspired speech envelope extraction methods for improved EEG-based auditory attention detection in a cocktail party scenario[J]. IEEE transactions on neural systems and rehabilitation engineering, 2016, 25(5): 402-412.
    - Stimulus: Four Dutch short stories, lasting approximately six minutes each. Either dry speech was offered, i.e., each speaker was presented to a different ear, or speech signals were processed by (dead room) head-related transfer functions, simulating a more realistic listening scenario in which the speakers are spatially located 90 degrees to the left and the right of the subject.
    - Signals: The BioSemi ActiveTwo system was used to record 64-channel EEG signals at 8196 Hz sample rate (downsampled to 64 Hz). Each subject completed 8 trials, each lasting 6 minutes.
    - Labels: Attended direction (attended_ear='L' for left, attended_ear='R' for right), stimulus condition (condition='dry' or 'hrtf').

    In order to use this dataset, the download file :obj:`4004271.zip` is required. After unzipping, the folder should contain the following files:

    .. code-block:: text

        4004271/
        ├── S1.mat
        ├── S2.mat
        ├── S3.mat
        ├── ...
        └── S16.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import KULDataset
        from torcheeg import transforms

        dataset = KULDataset(root_path='./4004271',
                             offline_transform=transforms.To2d(),
                             online_transform=transforms.ToTensor(),
                             label_transform=transforms.Compose([
                                 transforms.Select('attended_ear'),
                                 transforms.Lambda(lambd=lambda x: 1 if x == 'L' else 0)
                             ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 64, 64]),
        # label (int)

    An example dataset with custom preprocessing parameters:

    .. code-block:: python

        from torcheeg.datasets import KULDataset
        from torcheeg import transforms

        dataset = KULDataset(root_path='./4004271',
                            chunk_size=64,
                            overlap=64,
                            lowpass=10,
                            highpass=0.5,
                            sampling_rate=64,
                            rereference='mean',
                            offline_transform=transforms.To2d(),
                            online_transform=transforms.ToTensor(),
                            label_transform=transforms.Compose([
                                transforms.Select('attended_ear'),
                                transforms.Lambda(lambd=lambda x: 1 if x == 'L' else 0)
                            ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 64, 64]),
        # label (int)

    Args:
        root_path (str): Path to the raw data files in MATLAB format. (default: :obj:`'./4004271'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of an entire trial is used as a single chunk. (default: :obj:`64`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG data. (default: :obj:`0`)
        num_channel (int): Number of channels to use. The dataset contains 64 channels. (default: :obj:`64`)
        lowpass (float): Low-pass filter cutoff frequency in Hz. (default: :obj:`9`)
        highpass (float): High-pass filter cutoff frequency in Hz. (default: :obj:`1`)
        sampling_rate (int): Target sampling rate in Hz for the output EEG signals. (default: :obj:`64`)
        rereference (str, optional): Rereference method. Options are 'Cz' (reference to Cz channel, index 47), 'mean' (average reference), or None (no rereference). (default: :obj:`'Cz'`)
        online_transform (Callable, optional): The transformation applied to the EEG signals. The input is a :obj:`np.ndarray`, and the output is used as the first value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation applied to the label. The input is an information dictionary, and the output is used as the second value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): A hook function performed on the trial to which the sample belongs. It is executed before the offline transformation and is typically used to implement context-dependent sample transformations, such as moving averages. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), and the ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): A hook function performed on the trial to which the sample belongs. It is executed after the offline transformation and is typically used to implement context-dependent sample transformations, such as moving averages. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        after_subject (Callable, optional): A hook function performed on the subject to which the sample belongs. It is executed after the offline transformation and is typically used to implement context-dependent sample transformations, such as moving averages. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to the generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size the database may grow to; used to size the memory mapping. If the database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signals. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system-based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory is used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars. (default: :obj:`True`)
    '''

    def __init__(self,
                 root_path: str = './4004271',
                 chunk_size: int = 64,
                 overlap: int = 0,
                 num_channel: int = 64,
                 lowpass: float = 9,
                 highpass: float = 1,
                 sampling_rate: int = 64,
                 rereference: Union[None, str] = 'Cz',
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'lowpass': lowpass,
            'highpass': highpass,
            'sampling_rate': sampling_rate,
            'rereference': rereference,
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
            'verbose': verbose
        }
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def read_record(record: str,
                    root_path: str = './4004271',
                    num_channel: int = 64,
                    lowpass: float = 9,
                    highpass: float = 1,
                    sampling_rate: int = 64,
                    rereference: Union[None, str] = 'Cz',
                    **kwargs) -> Dict:
        mat_path = os.path.join(root_path, record)
        matstruct_contents = loadmat(mat_path)

        trials = matstruct_contents['trials'][0]
        n_trials = 8

        eeg_data = []
        attended_ear_data = []
        condition_data = []
        stimuli_data = []

        for trial_id in range(n_trials):
            trial = trials[trial_id]
            raw_eeg = trial[0][0][0][0][0][1]
            original_sr = trial[0][0][1][0][0][10][0][0]
            attended_ear = trial[0][0][3][0]
            condition = trial[0][0][5][0]

            try:
                stimuli_left = trial[0][0][4][0][0][0]
                stimuli_right = trial[0][0][4][1][0][0]
                stimuli = (stimuli_left, stimuli_right)
            except:
                stimuli = (None, None)

            raw_eeg = raw_eeg[:, :num_channel]
            if rereference:
                raw_eeg = apply_rereference(raw_eeg, rereference)

            raw_eeg = apply_fir_filter(raw_eeg, lowpass, highpass, original_sr)

            if original_sr != sampling_rate:
                raw_eeg = apply_downsample(raw_eeg, original_sr, sampling_rate)

            eeg_data.append(raw_eeg)
            attended_ear_data.append(attended_ear)
            condition_data.append(condition)
            stimuli_data.append(stimuli)

        return {
            'samples': eeg_data,
            'attended_ear': attended_ear_data,
            'condition': condition_data,
            'stimuli': stimuli_data,
            'sampling_rate': sampling_rate
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        """
        Generate fake data for testing purposes.
        """
        num_trials = 8
        num_channels = 64
        sampling_rate = kwargs.get('sampling_rate', 64)
        trial_length = int(6 * 60 * sampling_rate)

        samples = [np.random.randn(trial_length, num_channels)
                   for _ in range(num_trials)]
        attended_ear = [np.random.choice(['L', 'R'])
                        for _ in range(num_trials)]
        condition = [np.random.choice(['dry', 'hrtf'])
                     for _ in range(num_trials)]
        stimuli = [(f'part{i}_track1_dry.wav', f'part{i}_track2_dry.wav')
                   for i in range(1, num_trials + 1)]

        return {
            'samples': samples,
            'attended_ear': attended_ear,
            'condition': condition,
            'stimuli': stimuli,
            'sampling_rate': sampling_rate
        }

    @staticmethod
    def process_record(record: str,
                       samples: list,
                       attended_ear: list,
                       condition: list,
                       stimuli: list,
                       sampling_rate: int,
                       chunk_size: int = 64,
                       overlap: int = 0,
                       num_channel: int = 64,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        """
        Process the preprocessed EEG data into chunks.

        This method chunks the preprocessed EEG data according to the specified
        chunk_size and overlap parameters.
        """
        subject_id = record.split('.')[0]

        write_pointer = 0

        for trial_id in range(len(samples)):

            trial_samples = samples[trial_id].T

            if before_trial:
                trial_samples = before_trial(trial_samples)

            trial_meta_info = {
                'subject_id': subject_id,
                'trial_id': trial_id,
                'attended_ear': attended_ear[trial_id],
                'condition': condition[trial_id],
                'stimuli_left': stimuli[trial_id][0] if stimuli[trial_id][0] else 'unknown',
                'stimuli_right': stimuli[trial_id][1] if stimuli[trial_id][1] else 'unknown',
                'sampling_rate': sampling_rate
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
                    t = offline_transform(eeg=clip_sample)
                    t_eeg = t['eeg']

                clip_id = f'{record}_{write_pointer}'
                write_pointer += 1

                record_info = {
                    'start_at': start_at,
                    'end_at': end_at,
                    'clip_id': clip_id
                }
                record_info.update(trial_meta_info)

                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size

    def set_records(self,
                    root_path: str = './4004271',
                    **kwargs):
        """
        Set up the list of subject files to be processed.
        """
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        all_files = os.listdir(root_path)
        records = [f for f in all_files if f.endswith(
            '.mat') and f.startswith('S') and not f.startswith('._')]

        records = [r for r in records if 'preproc' not in r.lower()]

        return sorted(records)

    def __getitem__(self, index: int) -> Tuple:
        """
        Get an item from the dataset.

        Returns:
            Tuple: (signal, label) where signal is the EEG data and label contains metadata
        """
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
                'lowpass': self.lowpass,
                'highpass': self.highpass,
                'sampling_rate': self.sampling_rate,
                'rereference': self.rereference,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'after_subject': self.after_subject,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
