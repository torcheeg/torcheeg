import os
from typing import Callable, Dict, Tuple, Union

import numpy as np
import scipy.io as scio

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset


class DREAMERData(BaseData):
    """Read and preprocess records for :class:`DREAMERDataset`."""

    channels = [
        'AF3', 'F7',  'F3',  'FC5', 'T7',  'P7',  'O1',  'O2',
        'P8',  'T8',  'FC6', 'F4',  'F8',  'AF4',
    ]

    grid = [
        ['-',  '-',   '-',  '-',   '-', '-',   '-',  '-',   '-'],
        ['-',  '-',   '-',  'AF3', '-', 'AF4', '-',  '-',   '-'],
        ['F7', '-',   'F3', '-',   '-', '-',   'F4', '-',   'F8'],
        ['-',  'FC5', '-',  '-',   '-', '-',   '-',  'FC6', '-'],
        ['T7', '-',   '-',  '-',   '-', '-',   '-',  '-',   'T8'],
        ['-',  '-',   '-',  '-',   '-', '-',   '-',  '-',   '-'],
        ['P7', '-',   '-',  '-',   '-', '-',   '-',  '-',   'P8'],
        ['-',  '-',   '-',  '-',   '-', '-',   '-',  '-',   '-'],
        ['-',  '-',   '-',  'O1',  '-', 'O2',  '-',  '-',   '-'],
    ]

    adjacency = {
        'AF3': ['F3', 'AF4'],
        'AF4': ['AF3', 'F4'],
        'F7':  ['F3', 'FC5', 'T7'],
        'F3':  ['AF3', 'F7', 'FC5'],
        'F4':  ['AF4', 'F8', 'FC6'],
        'F8':  ['F4', 'FC6', 'T8'],
        'FC5': ['F7', 'F3', 'T7'],
        'FC6': ['F4', 'F8', 'T8'],
        'T7':  ['F7', 'FC5', 'P7'],
        'T8':  ['F8', 'FC6', 'P8'],
        'P7':  ['T7'],
        'P8':  ['T8'],
        'O1':  ['O2'],
        'O2':  ['O1'],
    }

    def __init__(self,
                 mat_path: str = './DREAMER.mat',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 14,
                 num_baseline: int = 61,
                 baseline_chunk_size: int = 128,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            mat_path=mat_path,
            chunk_size=chunk_size,
            overlap=overlap,
            num_channel=num_channel,
            num_baseline=num_baseline,
            baseline_chunk_size=baseline_chunk_size
        )

    def read_record(self, record: str, mat_path: str = None, **kwargs) -> Dict:
        if mat_path is None:
            mat_path = self.mat_path
        mat_data = scio.loadmat(mat_path,
                                verify_compressed_data_integrity=False)

        return {
            'mat_data': mat_data,
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trials = 18
        num_channel = 14
        trial_length = 7680
        baseline_length = 61 * 128

        score_valence = np.random.uniform(1, 5, (num_trials, 1))
        score_arousal = np.random.uniform(1, 5, (num_trials, 1))
        score_dominance = np.random.uniform(1, 5, (num_trials, 1))

        stimuli_data = np.zeros((num_trials, 1), dtype=object)
        baseline_data = np.zeros((num_trials, 1), dtype=object)

        for i in range(num_trials):
            stimuli_data[i, 0] = np.random.rand(trial_length, num_channel)
            baseline_data[i, 0] = np.random.rand(baseline_length, num_channel)

        eeg_data = {
            'stimuli': np.array([[stimuli_data]]),
            'baseline': np.array([[baseline_data]])
        }

        subject_data = {
            'EEG': np.array([[eeg_data]]),
            'ScoreValence': np.array([[score_valence]]),
            'ScoreArousal': np.array([[score_arousal]]),
            'ScoreDominance': np.array([[score_dominance]])
        }

        data_array = np.zeros((1, 1), dtype=object)
        data_array[0, 0] = subject_data

        mat_data = {
            'DREAMER': np.array([[{
                'Data': data_array
            }]])
        }

        return {
            'mat_data': mat_data,
            'record': 0
        }

    def process_record(self, record: str, mat_data: Dict, chunk_size: int = None, overlap: int = None, num_channel: int = None, num_baseline: int = None, baseline_chunk_size: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if num_channel is None:
            num_channel = self.num_channel
        if num_baseline is None:
            num_baseline = self.num_baseline
        if baseline_chunk_size is None:
            baseline_chunk_size = self.baseline_chunk_size
        subject = record
        trial_len = len(mat_data['DREAMER'][0, 0]['Data']
                        [0, 0]['EEG'][0, 0]['stimuli'][0, 0])
        for trial_id in range(trial_len):
            trial_baseline_sample = mat_data['DREAMER'][0, 0]['Data'][0,
                                                                      subject]['EEG'][0, 0]['baseline'][0, 0][trial_id, 0]
            trial_baseline_sample = trial_baseline_sample[:, :num_channel].swapaxes(
                1, 0)
            trial_baseline_sample = trial_baseline_sample[:, :num_baseline * baseline_chunk_size].reshape(
                num_channel, num_baseline, baseline_chunk_size).mean(axis=1)
            trial_meta_info = {'subject_id': subject, 'trial_id': trial_id}
            trial_meta_info['valence'] = mat_data['DREAMER'][0,
                                                             0]['Data'][0, subject]['ScoreValence'][0, 0][trial_id, 0]
            trial_meta_info['arousal'] = mat_data['DREAMER'][0,
                                                             0]['Data'][0, subject]['ScoreArousal'][0, 0][trial_id, 0]
            trial_meta_info['dominance'] = mat_data['DREAMER'][0,
                                                               0]['Data'][0, subject]['ScoreDominance'][0, 0][trial_id, 0]
            trial_samples = mat_data['DREAMER'][0, 0]['Data'][0,
                                                              subject]['EEG'][0, 0]['stimuli'][0, 0][trial_id, 0]
            trial_samples = trial_samples[:, :num_channel].swapaxes(1, 0)
            start_at = 0
            if chunk_size <= 0:
                dynamic_chunk_size = trial_samples.shape[1] - start_at
            else:
                dynamic_chunk_size = chunk_size
            end_at = dynamic_chunk_size
            step = dynamic_chunk_size - overlap
            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:, start_at:end_at]
                t_eeg = clip_sample
                t_baseline = trial_baseline_sample
                record_info = {'start_at': start_at, 'end_at': end_at}
                record_info.update(trial_meta_info)
                yield {'eeg': t_eeg, 'info': record_info, 'baseline': t_baseline}
                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size

    def set_records(self, mat_path: str = None, **kwargs):
        if mat_path is None:
            mat_path = self.mat_path
        if not (os.path.exists(mat_path)):
            raise FileNotFoundError(
                f'mat_path ({mat_path}) does not exist. Please download the dataset and set the mat_path to the downloaded path.'
            )

        mat_data = scio.loadmat(mat_path,
                                verify_compressed_data_integrity=False)

        subject_len = len(mat_data['DREAMER'][0, 0]['Data'][0])  # 23
        return list(range(subject_len))


class DREAMERDataset(CachedDataset):

    r"""
    A multi-modal database consisting of electroencephalogram and electrocardiogram signals recorded during affect elicitation by means of audio-visual stimuli. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Katsigiannis et al.
    - Year: 2017
    - Download URL: https://zenodo.org/record/546113
    - Reference: Katsigiannis S, Ramzan N. DREAMER: A database for emotion recognition through EEG and ECG signals from wireless low-cost off-the-shelf devices[J]. IEEE journal of biomedical and health informatics, 2017, 22(1): 98-107.
    - Stimulus: 18 movie clips.
    - Signals: Electroencephalogram (14 channels at 128Hz), and electrocardiogram (2 channels at 256Hz) of 23 subjects.
    - Rating: Arousal, valence, like/dislike, dominance, familiarity (all ona scale from 1 to 5).

    In order to use this dataset, the download file :obj:`DREAMER.mat` is required.

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import DREAMERDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_locations


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                transforms.BandDifferentialEntropy(),
                transforms.ToGrid(get_locations(
                    DREAMERDataset.channels, DREAMERDataset.grid)),
                transforms.ToTensor(),
            ])
            label_transform = transforms.Compose([
                transforms.Select('valence'),
                transforms.Binary(3.0),
            ])
            return {
                "eeg": eeg_transform(sample["eeg"]),
                "label": label_transform(sample["info"]),
            }


        dataset = DREAMERDataset(mat_path='./DREAMER.mat').with_transform(transform_fn)
        print(dataset[0])

    Another example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import DREAMERDataset
        from torcheeg import transforms


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                transforms.To2d(),
                transforms.ToTensor(),
            ])
            label_transform = transforms.Compose([
                transforms.Select(['valence', 'arousal']),
                transforms.Binary(3.0),
                transforms.BinariesToCategory(),
            ])
            return {
                "eeg": eeg_transform(sample["eeg"]),
                "label": label_transform(sample["info"]),
            }


        dataset = DREAMERDataset(mat_path='./DREAMER.mat').with_transform(transform_fn)
        print(dataset[0])

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import DREAMERDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_adjacency
        from torcheeg.transforms.graph import ToG


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                ToG(get_adjacency(DREAMERDataset.channels, DREAMERDataset.adjacency)),
            ])
            label_transform = transforms.Compose([
                transforms.Select('arousal'),
                transforms.Binary(3.0),
            ])
            return {
                "eeg": eeg_transform(sample["eeg"]),
                "label": label_transform(sample["info"]),
            }


        dataset = DREAMERDataset(mat_path='./DREAMER.mat').with_transform(transform_fn)
        print(dataset[0])

    Args:
        mat_path (str): Downloaded data files in pickled matlab formats (default: :obj:`'./DREAMER.mat'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 14 channels are EEG signals. (default: :obj:`14`)
        num_baseline (int): Number of baseline signal chunks used. (default: :obj:`61`)
        baseline_chunk_size (int): Number of data points included in each baseline signal chunk. The baseline signal in the DREAMER dataset has a total of 7808 data points. (default: :obj:`128`)



        transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = DREAMERData

    channels = DREAMERData.channels

    grid = DREAMERData.grid

    adjacency = DREAMERData.adjacency

    def __init__(self, mat_path: str = './DREAMER.mat', chunk_size: int = 128, overlap: int = 0, num_channel: int = 14, num_baseline: int = 61, baseline_chunk_size: int = 128, transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'mat_path': mat_path, 'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel, 'num_baseline': num_baseline,
                  'baseline_chunk_size': baseline_chunk_size, 'transform': transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'mat_path': self.mat_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel, 'num_baseline': self.num_baseline, 'baseline_chunk_size': self.baseline_chunk_size, 'transform': self.transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
