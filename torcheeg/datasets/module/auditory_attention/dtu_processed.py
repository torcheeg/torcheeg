import os
from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy.io import loadmat

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


class DTUProcessedDataset(BaseDataset):
    r'''
    This dataset contains EEG recordings from 18 subjects listening to one of two competing speech audio streams, after matlab preprocessing. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Fuglsang et al.
    - Year: 2017
    - Download URL: https://zenodo.org/records/1199011
    - Reference: Fuglsang S A, Dau T, Hjortkjær J. Noise-robust cortical tracking of attended speech in real-world acoustic scenes[J]. NeuroImage, 2017, 156: 435-444.
    - Stimulus: Continuous speech in trials of ~50 seconds with target speakers positioned from -60° to 60°.
    - Signals: Electroencephalogram recorded in a double-walled soundproof booth at the Technical University of Denmark (DTU) using a 64-channel Biosemi system (channels 1-64: scalp EEG electrodes, channel 65: right mastoid electrode, channel 66: left mastoid electrode) and digitized at a sampling rate of 512 Hz (downsampled to 64 Hz).
    - Labels: Attended speaker (attended_speaker=1 for male, attended_speaker=2 for female).

    In order to use this dataset, the download file :obj:`DATA_preproc.zip` is required. After unzipping, the folder should contain the following files:

    .. code-block:: text

        DATA_preproc/
        ├── S1_data_preproc.mat
        ├── S2_data_preproc.mat
        ├── S3_data_preproc.mat
        ├── ...
        └── S18_data_preproc.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import DTUProcessedDataset
        from torcheeg import transforms

        dataset = DTUProcessedDataset(root_path='./DATA_preproc',
                                   offline_transform=transforms.To2d(),
                                   online_transform=transforms.ToTensor(),
                                   label_transform=transforms.Compose([
                                       transforms.Select('attended_speaker'),
                                       transforms.Lambda(lambd=lambda x: x - 1)
                                   ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 64, 64]),
        # label (int)

    Args:
        root_path (str): Path to the downloaded data files in MATLAB format (unzipped DATA_preproc.zip). (default: :obj:`'./DATA_preproc'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of an entire trial is used as a single chunk. (default: :obj:`64`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG data. (default: :obj:`0`)
        num_channel (int): Number of channels to use. The dataset contains 66 channels total (64 scalp EEG + 2 mastoid electrodes). (default: :obj:`64`)
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
                 root_path: str = './DATA_preproc',
                 chunk_size: int = 64,
                 overlap: int = 0,
                 num_channel: int = 64,
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
    def read_record(record: str, root_path: str = './DATA_preproc', **kwargs) -> Dict:
        mat_path = os.path.join(root_path, record)
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['data']

        mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
        mat_event_value = mat_event[0]['value']

        mat_eeg = matstruct_contents[0, 0]['eeg']

        eeg_data = []
        event_data = []

        for i in range(mat_eeg.shape[1]):
            eeg_data.append(mat_eeg[0, i])
            event_data.append(mat_event_value[i][0][0])

        return {
            'samples': np.array(eeg_data),
            'attended_speaker': np.array(event_data)
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trials = 60
        num_channels = 66
        trial_length = 3200

        samples = np.random.randn(num_trials, trial_length, num_channels)
        attended_speaker = np.random.randint(1, 3, num_trials)

        return {
            'samples': samples,
            'attended_speaker': attended_speaker
        }

    @staticmethod
    def process_record(record: str,
                       samples: np.ndarray,
                       attended_speaker: np.ndarray,
                       chunk_size: int = 64,
                       overlap: int = 0,
                       num_channel: int = 64,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        subject_id = record.split('_')[0]

        write_pointer = 0

        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id, :, :num_channel].T

            if before_trial:
                trial_samples = before_trial(trial_samples)

            trial_meta_info = {
                'subject_id': subject_id,
                'trial_id': trial_id,
                'attended_speaker': int(attended_speaker[trial_id])
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
                    root_path: str = './DATA_preproc',
                    **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        records = [f for f in os.listdir(
            root_path) if f.endswith('_data_preproc.mat')]
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
