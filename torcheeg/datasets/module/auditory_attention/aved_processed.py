import os
from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


class AVEDProcessedDataset(BaseDataset):
    r'''
    This dataset contains EEG recordings from 10 subjects listening to one of two competing speech audio streams under audio-video or audio-only conditions. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Zhang et al.
    - Year: 2024
    - Download URL: https://iiphci.ahu.edu.cn/toAuditoryAttention
    - Reference: ZHANG H, ZHANG J. Based on audio-video evoked auditory attention detection electroencephalogram dataset[J]. Journal of Tsinghua University (Science and Technology), 2024, 64(11): 1919-1926.
    - Stimulus: 16 stories selected from the Chinese short story collection "Strange Tales from a Chinese Studio", narrated by a male and a female speaker. Each trial lasts 152 seconds with the target speaker from either 90° left or right.
    - Signals: Electroencephalogram recorded in a double-walled soundproof booth at the Technical University of Denmark (DTU) using a 32-channel system and digitized at a sampling rate of 1000 Hz (downsampled to 128 Hz).
    - Rating: Attended speaker (1 for male, 2 for female), attended direction (1 for left, 2 for right), condition (audio-video or audio-only)

    In order to use this dataset, the download folder :obj:`eeg_preproc.zip` is required. After unzipping, the folder should contain the following files:

    .. code-block:: text

        eeg_preproc/
        ├── audio-video
        │   ├── sub1.csv
        │   ├── sub2.csv
        │   ├── sub3.csv
        │   ├── ...
        │   └── sub10.csv
        └── audio-only
            ├── sub1.csv
            ├── sub2.csv
            ├── sub3.csv
            ├── ...
            └── sub10.csv

    In the dataset, 50 Hz power line interference is removed, and the signals are band-pass filtered (0.1-50 Hz) and downsampled to 128 Hz. Subsequently, ocular and muscle artifacts are eliminated using independent component analysis (ICA). Finally, all EEG channels were re-referenced.

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import AVEDProcessedDataset
        from torcheeg import transforms

        dataset = AVEDProcessedDataset(root_path='./eeg_preproc',
                                       chunk_size=128,
                                       overlap=0,
                                       num_channel=32,
                                       online_transform=transforms.ToTensor(),
                                       label_transform=transforms.Compose([
                                           transforms.Select('attended_speaker'),
                                           transforms.Lambda(lambda x: int(x) - 1)
                                       ]))
        print(dataset[0])

    Args:
        root_path (str): Downloaded data files in CSV format (unzipped eeg_preproc folder) (default: :obj:`'./eeg_preproc'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. (default: :obj:`32`)
        online_transform (Callable, optional): The transformation of the EEG signals. The input is a :obj:`np.ndarray`, and the output is used as the first value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the output is used as the second value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        after_subject (Callable, optional): The hook performed on the subject to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    '''

    def __init__(self,
                 root_path: str = './eeg_preproc',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 32,
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
    def read_record(record: Dict, root_path: str = './eeg_preproc', **kwargs) -> Dict:
        condition = record['condition']
        subject_id = record['subject_id']

        filename = os.path.join(root_path, condition, f'sub{subject_id}.csv')
        data_df = pd.read_csv(filename, header=None)
        samples = data_df.values

        trial_length = 152 * 128
        num_trials = 16

        samples = samples.reshape(num_trials, trial_length, -1)
        samples = samples.transpose(0, 2, 1)

        attended_speaker = np.array([1, 2] * 8)
        attended_direction = np.array([1, 2] * 8)

        return {
            'samples': samples,
            'attended_speaker': attended_speaker,
            'attended_direction': attended_direction,
            'condition': condition,
            'subject_id': subject_id
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trials = 16
        num_channels = 32
        trial_length = 152 * 128

        samples = np.random.randn(num_trials, num_channels, trial_length)
        attended_speaker = np.array([1, 2] * 8)
        attended_direction = np.array([1, 2] * 8)

        return {
            'samples': samples,
            'attended_speaker': attended_speaker,
            'attended_direction': attended_direction,
            'condition': 'audio-video',
            'subject_id': 1
        }

    @staticmethod
    def process_record(record: Dict,
                       samples: np.ndarray,
                       attended_speaker: np.ndarray,
                       attended_direction: np.ndarray,
                       condition: str,
                       subject_id: int,
                       chunk_size: int = 128,
                       overlap: int = 0,
                       num_channel: int = 32,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        write_pointer = 0

        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id, :num_channel, :]

            if before_trial:
                trial_samples = before_trial(trial_samples)

            trial_meta_info = {
                'subject_id': subject_id,
                'trial_id': trial_id + 1,
                'attended_speaker': int(attended_speaker[trial_id]),
                'attended_direction': int(attended_direction[trial_id]),
                'condition': condition
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

                clip_id = f'{condition}_sub{subject_id}_{write_pointer}'
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

    def set_records(self, root_path: str = './eeg_preproc', **kwargs):
        assert os.path.exists(root_path), \
            f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        records = []
        conditions = ['audio-video', 'audio-only']

        for condition in conditions:
            condition_path = os.path.join(root_path, condition)
            assert os.path.exists(condition_path), \
                f'Condition path ({condition_path}) does not exist.'

            files = sorted([f for f in os.listdir(
                condition_path) if f.endswith('.csv')])

            for filename in files:
                subject_id = int(filename.replace(
                    'sub', '').replace('.csv', ''))
                records.append({
                    'condition': condition,
                    'subject_id': subject_id
                })

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
