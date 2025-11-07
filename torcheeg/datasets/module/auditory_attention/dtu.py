import os
from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy import signal
from scipy.io import loadmat

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


def apply_notch_filter(eeg_data: np.ndarray, fs: float, notch_freq: float = 50.0) -> np.ndarray:
    """
    Apply notch filter to remove line noise.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_samples, n_channels)
        fs (float): Sampling frequency
        notch_freq (float): Line noise frequency to remove (default: 50 Hz)

    Returns:
        np.ndarray: Filtered EEG data
    """
    quality_factor = 30.0
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)

    filtered_data = signal.filtfilt(b, a, eeg_data, axis=0)

    return filtered_data


def apply_highpass_filter(eeg_data: np.ndarray, fs: float, hp_freq: float = 0.1) -> np.ndarray:
    """
    Apply high-pass Butterworth filter.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_samples, n_channels)
        fs (float): Sampling frequency
        hp_freq (float): High-pass cutoff frequency (default: 0.1 Hz)

    Returns:
        np.ndarray: Filtered EEG data
    """
    detrended_data = signal.detrend(eeg_data, axis=0, type='constant')

    sos = signal.butter(2, hp_freq, btype='high', fs=fs, output='sos')

    filtered_data = signal.sosfiltfilt(sos, detrended_data, axis=0)

    return filtered_data


def apply_downsample(eeg_data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """
    Downsample EEG data.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_samples, n_channels)
        original_fs (float): Original sampling rate
        target_fs (float): Target sampling rate

    Returns:
        np.ndarray: Downsampled EEG data
    """
    decimation_factor = int(original_fs / target_fs)

    downsampled_data = signal.decimate(
        eeg_data, decimation_factor, axis=0, zero_phase=True)

    return downsampled_data


def create_eog_bipolar(eeg_data: np.ndarray, ch_names: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create bipolar EOG channels from the DTU dataset.

    VEOG: EXG3 - EXG5 (vertical eye movement)
    HEOG: EXG4 - EXG7 (horizontal eye movement)

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_samples, n_channels)
        ch_names (list): List of channel names

    Returns:
        Tuple of (veog_data, heog_data)
    """
    exg3_idx = ch_names.index('EXG3')
    exg5_idx = ch_names.index('EXG5')
    exg4_idx = ch_names.index('EXG4')
    exg7_idx = ch_names.index('EXG7')

    veog = eeg_data[:, exg3_idx] - eeg_data[:, exg5_idx]
    heog = eeg_data[:, exg4_idx] - eeg_data[:, exg7_idx]

    return veog, heog


def apply_eog_removal(eeg_data: np.ndarray, veog: np.ndarray, heog: np.ndarray) -> np.ndarray:
    """
    Remove EOG artifacts using regression.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_samples, n_channels)
        veog (np.ndarray): Vertical EOG signal with shape (n_samples,)
        heog (np.ndarray): Horizontal EOG signal with shape (n_samples,)

    Returns:
        np.ndarray: Denoised EEG data
    """
    denoised_data = eeg_data.copy()

    eog_data = np.vstack([np.ones(len(veog)), veog, heog]).T

    for i in range(eeg_data.shape[1]):
        y = eeg_data[:, i]

        beta, _, _, _ = np.linalg.lstsq(eog_data, y, rcond=None)

        eeg_clean = y - (beta[1] * veog + beta[2] * heog)
        denoised_data[:, i] = eeg_clean

    return denoised_data


def apply_average_reference(eeg_data: np.ndarray) -> np.ndarray:
    """
    Apply average reference to EEG data.

    Args:
        eeg_data (np.ndarray): EEG data with shape (n_samples, n_channels)

    Returns:
        np.ndarray: Average-referenced EEG data
    """
    reference = np.mean(eeg_data, axis=1, keepdims=True)
    return eeg_data - reference


class DTUDataset(BaseDataset):
    r'''
    This dataset contains EEG recordings from 18 subjects listening to one of two competing speech audio streams. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Fuglsang et al.
    - Year: 2017
    - Download URL: https://zenodo.org/records/1199011
    - Reference: Fuglsang S A, Dau T, Hjortkjær J. Noise-robust cortical tracking of attended speech in real-world acoustic scenes[J]. NeuroImage, 2017, 156: 435-444.
    - Stimulus: Continuous speech in trials of ~50 seconds with target speakers positioned from -60° to 60°.
    - Signals: Electroencephalogram recorded in a double-walled soundproof booth at the Technical University of Denmark (DTU) using a 64-channel Biosemi system (channels 1-64: scalp EEG electrodes, channel 65: right mastoid electrode, channel 66: left mastoid electrode) and digitized at a sampling rate of 512 Hz (downsampled to 64 Hz).
    - Labels: Attended speaker (attend_mf=1 for male, attend_mf=2 for female).

    In order to use this dataset, the download file :obj:`EEG.zip` is required. After unzipping, the folder should contain the following files:

    .. code-block:: text

        EEG/
        ├── S1.mat
        ├── S2.mat
        ├── S3.mat
        ├── ...
        └── S18.mat

    You need to create an :obj:`expinfo.m` file in the folder:

    .. code-block:: matlab

        for ss = 1:18
            load(['S' num2str(ss) '.mat'])
            writetable(expinfo,['S' num2str(ss) '.csv'])
        end

    This extracts experiment information. After running, the folder should contain the following files:

    .. code-block:: text

        EEG/
        ├── S1.mat
        ├── S1.csv
        ├── S2.mat
        ├── S2.csv
        ├── ...
        └── S18.csv

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import DTUDataset
        from torcheeg import transforms

        dataset = DTUDataset(root_path='./EEG',
                            offline_transform=transforms.To2d(),
                            online_transform=transforms.ToTensor(),
                            label_transform=transforms.Compose([
                                transforms.Select('attend_mf'),
                                transforms.Lambda(lambd=lambda x: x - 1)
                            ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 64, 64]),
        # label (int)

    An example dataset with custom preprocessing parameters:

    .. code-block:: python

        from torcheeg.datasets import DTUDataset
        from torcheeg import transforms

        dataset = DTUDataset(root_path='./EEG',
                            chunk_size=64,
                            overlap=0,
                            sampling_rate=64,
                            num_channel=64,
                            eog_removal=True,
                            offline_transform=transforms.To2d(),
                            online_transform=transforms.ToTensor(),
                            label_transform=transforms.Compose([
                                transforms.Select('attend_mf'),
                                transforms.Lambda(lambd=lambda x: x - 1)
                            ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 64, 64]),
        # label (int)

    Args:
        root_path (str): Path to the raw data files in MATLAB format. (default: :obj:`'./EEG'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of an entire trial is used as a single chunk. (default: :obj:`64`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG data. (default: :obj:`0`)
        num_channel (int): Number of channels to use. The dataset contains 73 channels total (64 scalp EEG + EOG channels + reference). We use the first 64 channels by default. (default: :obj:`64`)
        sampling_rate (int): Target sampling rate in Hz for the output EEG signals. (default: :obj:`64`)
        eog_removal (bool): Whether to apply EOG artifact removal using regression. If True, EOG channels will be used to remove eye movement artifacts from the EEG data. (default: :obj:`True`)
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
                 root_path: str = './EEG',
                 chunk_size: int = 64,
                 overlap: int = 0,
                 num_channel: int = 64,
                 sampling_rate: int = 64,
                 eog_removal: bool = True,
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
            'sampling_rate': sampling_rate,
            'eog_removal': eog_removal,
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
                    root_path: str = './EEG',
                    num_channel: int = 64,
                    sampling_rate: int = 64,
                    eog_removal: bool = True,
                    **kwargs) -> Dict:

        mat_path = os.path.join(root_path, record)
        matstruct_contents = loadmat(
            mat_path, struct_as_record=False, squeeze_me=True)

        data_struct = matstruct_contents['data']

        eeg_data = data_struct.eeg
        original_fs = float(data_struct.fsample.eeg)
        ch_names = [str(name) for name in data_struct.dim.chan.eeg]

        event_samples = data_struct.event.eeg.sample
        event_values = data_struct.event.eeg.value

        subject_id = record.split('.')[0]
        csv_path = os.path.join(root_path, f'{subject_id}.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}. Please create CSV files using the expinfo.m script as described in the documentation."
            )

        import pandas as pd
        expinfo = pd.read_csv(csv_path)

        has_wav_file = expinfo.wavfile_female.astype(str) != 'nan'
        expinfo = expinfo[has_wav_file].reset_index(drop=True)

        eeg_data = apply_notch_filter(eeg_data, original_fs, notch_freq=50.0)

        if original_fs != sampling_rate:
            eeg_data = apply_downsample(eeg_data, original_fs, sampling_rate)
            event_samples = (event_samples * sampling_rate /
                             original_fs).astype(int)

        eeg_data = apply_highpass_filter(eeg_data, sampling_rate, hp_freq=0.1)

        if eog_removal:
            veog, heog = create_eog_bipolar(eeg_data, ch_names)
            channels_to_remove = ['EXG3', 'EXG4',
                                  'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
            keep_indices = [i for i, ch in enumerate(
                ch_names) if ch not in channels_to_remove]
            eeg_data = eeg_data[:, keep_indices]
            ch_names_clean = [ch_names[i] for i in keep_indices]

            eeg_data = apply_eog_removal(eeg_data, veog, heog)
        else:
            channels_to_remove = ['EXG3', 'EXG4',
                                  'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
            keep_indices = [i for i, ch in enumerate(
                ch_names) if ch not in channels_to_remove]
            eeg_data = eeg_data[:, keep_indices]
            ch_names_clean = [ch_names[i] for i in keep_indices]

        eeg_data = apply_average_reference(eeg_data)

        trial_data = []
        trial_info = []

        if isinstance(event_values, np.ndarray):
            event_values = event_values.tolist()
        if isinstance(event_samples, np.ndarray):
            event_samples = event_samples.tolist()

        triggers = expinfo['trigger'].tolist()

        event_ptr = 0

        for trial_idx, current_trigger in enumerate(triggers):
            start_sample = None
            end_sample = None

            while event_ptr < len(event_values):
                if event_values[event_ptr] == current_trigger:
                    start_sample = event_samples[event_ptr]
                    event_ptr += 1

                    while event_ptr < len(event_values):
                        # 191 is the end trigger, 192 is a fix for subject 8
                        if event_values[event_ptr] == 191 or event_values[event_ptr] == 192:
                            end_sample = event_samples[event_ptr]
                            event_ptr += 1
                            break
                        event_ptr += 1
                    break
                event_ptr += 1

            if start_sample is None:
                print(
                    f"Warning: Subject {subject_id} has no start sample for trial {trial_idx}")
            elif end_sample is None:
                print(
                    f"Warning: Subject {subject_id} has no end sample for trial {trial_idx}")
            else:
                trial_eeg = eeg_data[start_sample:end_sample, :num_channel]

                trial_metadata = {
                    'attend_mf': int(expinfo.iloc[trial_idx]['attend_mf']),
                    'attend_lr': int(expinfo.iloc[trial_idx]['attend_lr']),
                    'acoustic_condition': int(expinfo.iloc[trial_idx]['acoustic_condition']),
                    'n_speakers': int(expinfo.iloc[trial_idx]['n_speakers']),
                    'wavfile_male': str(expinfo.iloc[trial_idx]['wavfile_male']),
                    'wavfile_female': str(expinfo.iloc[trial_idx]['wavfile_female']),
                    'trigger': int(expinfo.iloc[trial_idx]['trigger'])
                }

                trial_data.append(trial_eeg)
                trial_info.append(trial_metadata)

        return {
            'samples': trial_data,
            'trial_info': trial_info,
            'sampling_rate': sampling_rate,
            'ch_names': ch_names_clean[:num_channel]
        }

    @staticmethod
    def fake_record(eog_removal: bool = True, **kwargs) -> Dict:

        num_trials = 60
        num_channels = 64
        sampling_rate = kwargs.get('sampling_rate', 64)
        trial_length = int(50 * sampling_rate)

        samples = [np.random.randn(trial_length, num_channels)
                   for _ in range(num_trials)]

        trial_info = []
        for i in range(num_trials):
            trial_metadata = {
                'attend_mf': np.random.randint(1, 3),
                'attend_lr': np.random.randint(1, 3),
                'acoustic_condition': np.random.randint(1, 4),
                'n_speakers': 2,
                'wavfile_male': f'male_story{i+1}.wav',
                'wavfile_female': f'female_story{i+1}.wav',
                'trigger': np.random.randint(100, 200)
            }
            trial_info.append(trial_metadata)

        ch_names = [f'Ch{i+1}' for i in range(num_channels)]

        return {
            'samples': samples,
            'trial_info': trial_info,
            'sampling_rate': sampling_rate,
            'ch_names': ch_names
        }

    @staticmethod
    def process_record(record: str,
                       samples: list,
                       trial_info: list,
                       sampling_rate: int,
                       ch_names: list,
                       chunk_size: int = 64,
                       overlap: int = 0,
                       num_channel: int = 64,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        subject_id = record.split('.')[0]

        write_pointer = 0

        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id].T

            if before_trial:
                trial_samples = before_trial(trial_samples)

            trial_meta_info = {
                'subject_id': subject_id,
                'trial_id': trial_id,
                'sampling_rate': sampling_rate
            }
            trial_meta_info.update(trial_info[trial_id])

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
                    root_path: str = './EEG',
                    **kwargs):

        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        all_files = os.listdir(root_path)
        records = [f for f in all_files if f.endswith('.mat') and
                   f.startswith('S') and not f.startswith('._')]

        records = [r for r in records if 'preproc' not in r.lower()]

        for record in records:
            subject_id = record.split('.')[0]
            csv_path = os.path.join(root_path, f'{subject_id}.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"CSV file not found: {csv_path}. Please create CSV files using the expinfo.m script as described in the documentation."
                )

        return sorted(records)

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
                'sampling_rate': self.sampling_rate,
                'eog_removal': self.eog_removal,
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
