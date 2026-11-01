import os
import pickle as pkl
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset


class DEAPData(BaseData):
    """Read and preprocess records for :class:`DEAPDataset`."""

    channels = [
        'FP1', 'AF3', 'F3',  'F7',  'FC5', 'FC1', 'C3',  'T7',
        'CP5', 'CP1', 'P3',  'P7',  'PO3', 'O1',  'OZ',  'PZ',
        'FP2', 'AF4', 'FZ',  'F4',  'F8',  'FC6', 'FC2', 'CZ',
        'C4',  'T8',  'CP6', 'CP2', 'P4',  'P8',  'PO4', 'O2',
    ]

    grid = [
        ['-',  '-',   '-',  'FP1', '-',  'FP2', '-',  '-',   '-'],
        ['-',  '-',   '-',  'AF3', '-',  'AF4', '-',  '-',   '-'],
        ['F7', '-',   'F3', '-',   'FZ', '-',   'F4', '-',   'F8'],
        ['-',  'FC5', '-',  'FC1', '-',  'FC2', '-',  'FC6', '-'],
        ['T7', '-',   'C3', '-',   'CZ', '-',   'C4', '-',   'T8'],
        ['-',  'CP5', '-',  'CP1', '-',  'CP2', '-',  'CP6', '-'],
        ['P7', '-',   'P3', '-',   'PZ', '-',   'P4', '-',   'P8'],
        ['-',  '-',   '-',  'PO3', '-',  'PO4', '-',  '-',   '-'],
        ['-',  '-',   '-',  'O1',  'OZ', 'O2',  '-',  '-',   '-'],
    ]

    adjacency = {
        'FP1': ['AF3'],
        'FP2': ['AF4'],
        'AF3': ['FP1', 'FZ', 'F3'],
        'AF4': ['FP2', 'F4', 'FZ'],
        'F7':  ['FC5'],
        'F3':  ['AF3', 'FC1', 'FC5'],
        'FZ':  ['AF3', 'AF4', 'FC2', 'FC1'],
        'F4':  ['AF4', 'FC6', 'FC2'],
        'F8':  ['FC6'],
        'FC5': ['F7', 'F3', 'C3', 'T7'],
        'FC1': ['F3', 'FZ', 'CZ', 'C3'],
        'FC2': ['FZ', 'F4', 'C4', 'CZ'],
        'FC6': ['F4', 'F8', 'T8', 'C4'],
        'T7':  ['FC5', 'CP5'],
        'C3':  ['FC5', 'FC1', 'CP1', 'CP5'],
        'CZ':  ['FC1', 'FC2', 'CP2', 'CP1'],
        'C4':  ['FC2', 'FC6', 'CP6', 'CP2'],
        'T8':  ['FC6', 'CP6'],
        'CP5': ['T7', 'C3', 'P3', 'P7'],
        'CP1': ['C3', 'CZ', 'PZ', 'P3'],
        'CP2': ['CZ', 'C4', 'P4', 'PZ'],
        'CP6': ['C4', 'T8', 'P8', 'P4'],
        'P7':  ['CP5'],
        'P3':  ['CP5', 'CP1', 'PO3'],
        'PZ':  ['CP1', 'CP2', 'PO4', 'PO3'],
        'P4':  ['CP2', 'CP6', 'PO4'],
        'P8':  ['CP6'],
        'PO3': ['P3', 'PZ', 'OZ', 'O1'],
        'PO4': ['PZ', 'P4', 'O2', 'OZ'],
        'O1':  ['PO3', 'OZ'],
        'OZ':  ['PO3', 'PO4', 'O2', 'O1'],
        'O2':  ['PO4', 'OZ'],
    }

    def __init__(self,
                 root_path: str = './data_preprocessed_python',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 32,
                 num_baseline: int = 3,
                 baseline_chunk_size: int = 128,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size,
            overlap=overlap,
            num_channel=num_channel,
            num_baseline=num_baseline,
            baseline_chunk_size=baseline_chunk_size
        )

    def read_record(self,
                    record: str,
                    root_path: str = None,
                    **kwargs) -> Dict:
        if root_path is None:
            root_path = self.root_path
        with open(os.path.join(root_path, record), 'rb') as f:
            pkl_data = pkl.load(f, encoding='iso-8859-1')

        samples = pkl_data['data']  # trial(40), channel(32), timestep(63*128)
        labels = pkl_data['labels']

        return {
            'samples': samples,
            'labels': labels
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        num_trials = 40
        num_channels = 32
        trial_length = 63 * 128

        samples = np.random.randn(num_trials, num_channels, trial_length)

        labels = np.random.uniform(1, 9, (num_trials, 4))

        return {
            'samples': samples,
            'labels': labels
        }

    def process_record(self, record: str, samples: np.ndarray, labels: np.ndarray, chunk_size: int = None, overlap: int = None, num_channel: int = None, num_baseline: int = None, baseline_chunk_size: int = None, **kwargs):
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
        subject_id = record
        for trial_id in range(len(samples)):
            trial_samples = samples[trial_id, :num_channel]
            trial_baseline_sample = trial_samples[:,
                                                  :baseline_chunk_size * num_baseline]
            trial_baseline_sample = trial_baseline_sample.reshape(
                num_channel, num_baseline, baseline_chunk_size).mean(axis=1)
            trial_meta_info = {'subject_id': subject_id, 'trial_id': trial_id}
            trial_rating = labels[trial_id]
            for label_index, label_name in enumerate(['valence', 'arousal', 'dominance', 'liking']):
                trial_meta_info[label_name] = trial_rating[label_index]
            start_at = baseline_chunk_size * num_baseline
            if chunk_size <= 0:
                dynamic_chunk_size = trial_samples.shape[1] - start_at
            else:
                dynamic_chunk_size = chunk_size
            end_at = start_at + dynamic_chunk_size
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

    def set_records(self, root_path: str = None, **kwargs):
        if root_path is None:
            root_path = self.root_path
        if not (os.path.exists(root_path)):
            raise FileNotFoundError(
                f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
            )
        records = sorted(filename for filename in os.listdir(root_path)
                         if not filename.startswith('.')
                         and filename.lower().endswith('.dat')
                         and os.path.isfile(os.path.join(root_path, filename)))
        if not records:
            raise ValueError(f'No DEAP .dat files were found in {root_path}.')
        return records


class DEAPDataset(CachedDataset):

    r"""
    A multimodal dataset for the analysis of human affective states. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Koelstra et al.
    - Year: 2012
    - Download URL: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
    - Reference: Koelstra S, Muhl C, Soleymani M, et al. DEAP: A database for emotion analysis; using physiological signals[J]. IEEE transactions on affective computing, 2011, 3(1): 18-31.
    - Stimulus: 40 one-minute long excerpts from music videos.
    - Signals: Electroencephalogram (32 channels at 512Hz, downsampled to 128Hz), skinconductance level (SCL), respiration amplitude, skin temperature,electrocardiogram, blood volume by plethysmograph, electromyograms ofZygomaticus and Trapezius muscles (EMGs), electrooculogram (EOG), face video (for 22 participants).
    - Rating: Arousal, valence, like/dislike, dominance (all ona scale from 1 to 9), familiarity (on a scale from 1 to 5).

    In order to use this dataset, the download folder :obj:`data_preprocessed_python` is required, containing the following files:

    .. code-block:: text

        data_preprocessed_python/
        ├── s01.dat
        ├── s02.dat
        ├── s03.dat
        ├── ...
        └── s32.dat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_locations


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                transforms.BandDifferentialEntropy(),
                transforms.ToGrid(get_locations(
                    DEAPDataset.channels, DEAPDataset.grid)),
                transforms.ToTensor(),
            ])
            label_transform = transforms.Compose([
                transforms.Select('valence'),
                transforms.Binary(5.0),
            ])
            return {
                "eeg": eeg_transform(sample["eeg"]),
                "label": label_transform(sample["info"]),
            }


        dataset = DEAPDataset(
            root_path='./data_preprocessed_python').with_transform(transform_fn)
        print(dataset[0])

    Another example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                transforms.To2d(),
                transforms.ToTensor(),
            ])
            label_transform = transforms.Compose([
                transforms.Select(['valence', 'arousal']),
                transforms.Binary(5.0),
                transforms.BinariesToCategory(),
            ])
            return {
                "eeg": eeg_transform(sample["eeg"]),
                "label": label_transform(sample["info"]),
            }


        dataset = DEAPDataset(
            root_path='./data_preprocessed_python').with_transform(transform_fn)
        print(dataset[0])

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_adjacency
        from torcheeg.transforms.graph import ToG


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                ToG(get_adjacency(DEAPDataset.channels, DEAPDataset.adjacency)),
            ])
            label_transform = transforms.Compose([
                transforms.Select('arousal'),
                transforms.Binary(5.0),
            ])
            return {
                "eeg": eeg_transform(sample["eeg"]),
                "label": label_transform(sample["info"]),
            }


        dataset = DEAPDataset(
            root_path='./data_preprocessed_python').with_transform(transform_fn)
        print(dataset[0])

    Args:
        root_path (str): Downloaded data files in pickled python/numpy (unzipped data_preprocessed_python.zip) formats (default: :obj:`'./data_preprocessed_python'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 32 channels are EEG signals. (default: :obj:`32`)
        num_baseline (int): Number of baseline signal chunks used. (default: :obj:`3`)
        baseline_chunk_size (int): Number of data points included in each baseline signal chunk. The baseline signal in the DEAP dataset has a total of 384 data points. (default: :obj:`128`)



        transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = DEAPData

    channels = DEAPData.channels

    grid = DEAPData.grid

    adjacency = DEAPData.adjacency

    def __init__(self, root_path: str = './data_preprocessed_python', chunk_size: int = 128, overlap: int = 0, num_channel: int = 32, num_baseline: int = 3, baseline_chunk_size: int = 128, transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel, 'num_baseline': num_baseline,
                  'baseline_chunk_size': baseline_chunk_size, 'transform': transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel, 'num_baseline': self.num_baseline, 'baseline_chunk_size': self.baseline_chunk_size, 'transform': self.transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
