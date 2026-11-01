import logging
import os
import re
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import scipy.io as scio

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset

_log = logging.getLogger('torcheeg')


class AMIGOSData(BaseData):
    """Read and preprocess records for :class:`AMIGOSDataset`."""

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
                 root_path: str = './data_preprocessed',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 14,
                 num_trial: int = 16,
                 skipped_subjects: List[int] = [9, 12, 21, 22, 23, 24, 33],
                 num_baseline: int = 5,
                 baseline_chunk_size: int = 128,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size,
            overlap=overlap,
            num_channel=num_channel,
            num_trial=num_trial,
            skipped_subjects=skipped_subjects,
            num_baseline=num_baseline,
            baseline_chunk_size=baseline_chunk_size
        )

    def read_record(self,
                    record: str,
                    root_path: str = None,
                    **kwargs) -> Dict:
        if root_path is None:
            root_path = self.root_path
        data = scio.loadmat(os.path.join(root_path, record),
                            verify_compressed_data_integrity=False)
        samples = data['joined_data'][
            0]  # trial (20), timestep(n*128), channel(17) (14 channels are EEGs)
        # label file
        labels = data['labels_selfassessment'][
            0]  # trial (20), label of different dimensions ((1, 12))
        return {
            'samples': samples,
            'labels': labels
        }

    @staticmethod
    def fake_record(record: str, **kwargs) -> Dict:
        num_trials = 20
        trial_length = 10 * 128
        num_channels = 17
        num_labels = 12

        samples = []
        labels = []

        for _ in range(num_trials):
            trial_sample = np.random.rand(trial_length, num_channels)
            trial_label = np.random.uniform(1, 9, (1, num_labels))

            samples.append(trial_sample)
            labels.append(trial_label)

        return {
            'record': 'Data_Preprocessed_P1.mat',
            'samples': np.array(samples),
            'labels': np.array(labels)
        }

    def process_record(self, record: str, samples: np.ndarray, labels: np.ndarray, chunk_size: int = None, overlap: int = None, num_channel: int = None, num_trial: int = None, skipped_subjects: List[int] = None, num_baseline: int = None, baseline_chunk_size: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if num_channel is None:
            num_channel = self.num_channel
        if num_trial is None:
            num_trial = self.num_trial
        if skipped_subjects is None:
            skipped_subjects = self.skipped_subjects
        if num_baseline is None:
            num_baseline = self.num_baseline
        if baseline_chunk_size is None:
            baseline_chunk_size = self.baseline_chunk_size
        subject = int(re.findall('Data_Preprocessed_P(\\d*).mat', record)[0])
        if subject in skipped_subjects:
            return
        max_len = len(samples)
        if not num_trial <= 0:
            max_len = min(len(samples), num_trial)
        for trial_id in range(max_len):
            trial_samples = samples[trial_id]
            trial_meta_info = {'subject_id': subject, 'trial_id': trial_id}
            trial_rating = labels[trial_id][0]
            if not sum(trial_samples.shape) or not sum(trial_rating.shape):
                if sum(trial_samples.shape) != sum(trial_rating.shape):
                    _log.info(
                        f'Find EEG signals without labels, or labels without EEG signals. Please check the {trial_id + 1}-th experiment of the {subject}-th subject in the file {record}. TorchEEG currently skipped the mismatched data.')
                continue
            trial_samples = trial_samples.swapaxes(1, 0)
            for label_index, label_name in enumerate(['arousal', 'valence', 'dominance', 'liking', 'familiarity', 'neutral', 'disgust', 'happiness', 'surprise', 'anger', 'fear', 'sadness']):
                trial_meta_info[label_name] = trial_rating[label_index]
            trial_baseline_sample = trial_samples[:num_channel,
                                                  :baseline_chunk_size * num_baseline]
            trial_baseline_sample = trial_baseline_sample.reshape(
                num_channel, num_baseline, baseline_chunk_size).mean(axis=1)
            start_at = baseline_chunk_size * num_baseline
            if chunk_size <= 0:
                dynamic_chunk_size = trial_samples.shape[1] - start_at
            else:
                dynamic_chunk_size = chunk_size
            end_at = start_at + dynamic_chunk_size
            step = dynamic_chunk_size - overlap
            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:num_channel, start_at:end_at]
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
        return os.listdir(root_path)


class AMIGOSDataset(CachedDataset):

    r"""
    A dataset for Multimodal research of affect, personality traits and mood on Individuals and GrOupS (AMIGOS). This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Miranda-Correa et al.
    - Year: 2018
    - Download URL: http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html
    - Reference: Miranda-Correa J A, Abadi M K, Sebe N, et al. Amigos: A dataset for affect, personality and mood research on individuals and groups[J]. IEEE Transactions on Affective Computing, 2018, 12(2): 479-493.
    - Stimulus: 16 short affective video extracts and 4 long affective video extracts from movies.
    - Signals: Electroencephalogram (14 channels at 128Hz), electrocardiogram (2 channels at 60Hz) and galvanic skin response (1 channel at 60Hz) of 40 subjects. For the first 16 trials, 40 subjects watched a set of short affective video extracts. For the last 4 trials, 37 of the participants of the previous experiment watched a set of long affective video extracts.
    - Rating: arousal (1-9), valence (1-9), dominance (1-9), liking (1-9), familiarity (1-9), neutral (0, 1), disgust (0, 1),happiness (0, 1), surprise (0, 1), anger (0, 1), fear (0, 1), and sadness (0, 1).

    In order to use this dataset, the download folder :obj:`data_preprocessed` is required, containing the following files:

    - Data_Preprocessed_P01.mat
    - Data_Preprocessed_P02.mat
    - Data_Preprocessed_P03.mat
    - ...
    - Data_Preprocessed_P40.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import AMIGOSDataset
        from torcheeg import transforms

        from torcheeg.datasets.utils import get_locations


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                transforms.BandDifferentialEntropy(),
                transforms.ToGrid(get_locations(
                    AMIGOSDataset.channels, AMIGOSDataset.grid)),
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


        dataset = AMIGOSDataset(
            root_path='./data_preprocessed').with_transform(transform_fn)
        print(dataset[0])

    Another example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import AMIGOSDataset
        from torcheeg import transforms


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                transforms.To2d(),
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


        dataset = AMIGOSDataset(
            root_path='./data_preprocessed').with_transform(transform_fn)
        print(dataset[0])

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import AMIGOSDataset
        from torcheeg import transforms
        from torcheeg.transforms.graph import ToG

        from torcheeg.datasets.utils import get_adjacency


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                ToG(get_adjacency(AMIGOSDataset.channels, AMIGOSDataset.adjacency)),
            ])
            label_transform = transforms.Compose([
                transforms.Select('valence'),
                transforms.Binary(5.0),
            ])
            return {
                "eeg": eeg_transform(sample["eeg"]),
                "label": label_transform(sample["info"]),
            }


        dataset = AMIGOSDataset(
            root_path='./data_preprocessed').with_transform(transform_fn)
        print(dataset[0])

    Args:
        root_path (str): Downloaded data files in matlab (unzipped data_preprocessed.zip) formats (default: :obj:`'./data_preprocessed'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 14 channels are EEG signals. (default: :obj:`14`)
        num_trial (int): Number of trials used, of which the first 16 trials are conducted with short videos and the last 4 trials are conducted with long videos. If set to -1, all trials are used. (default: :obj:`16`)
        skipped_subjects (int): The participant ID to be removed because there are some invalid data in the preprocessed version. (default: :obj:`[9, 12, 21, 22, 23, 24, 33]`)
        num_baseline (int): Number of baseline signal chunks used. (default: :obj:`5`)
        baseline_chunk_size (int): Number of data points included in each baseline signal chunk. The baseline signal in the AMIGOS dataset has a total of 640 data points. (default: :obj:`128`)



        transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = AMIGOSData

    channels = AMIGOSData.channels

    grid = AMIGOSData.grid

    adjacency = AMIGOSData.adjacency

    def __init__(self, root_path: str = './data_preprocessed', chunk_size: int = 128, overlap: int = 0, num_channel: int = 14, num_trial: int = 16, skipped_subjects: List[int] = [9, 12, 21, 22, 23, 24, 33], num_baseline: int = 5, baseline_chunk_size: int = 128, transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel, 'num_trial': num_trial, 'skipped_subjects': skipped_subjects, 'num_baseline': num_baseline,
                  'baseline_chunk_size': baseline_chunk_size, 'transform': transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel, 'num_trial': self.num_trial, 'skipped_subjects': self.skipped_subjects, 'num_baseline': self.num_baseline, 'baseline_chunk_size': self.baseline_chunk_size, 'transform': self.transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
