import glob
import json
import os
from typing import Callable, Dict, Tuple, Union

import mne
import numpy as np
import xmltodict

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset


class MAHNOBData(BaseData):
    """Read and preprocess records for :class:`MAHNOBDataset`."""

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
                 root_path: str = './Sessions',
                 chunk_size: int = 128,
                 sampling_rate: int = 128,
                 overlap: int = 0,
                 num_channel: int = 32,
                 num_baseline: int = 30,
                 baseline_chunk_size: int = 128,
                 num_trial_sample: int = 30,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size,
            sampling_rate=sampling_rate,
            overlap=overlap,
            num_channel=num_channel,
            num_baseline=num_baseline,
            baseline_chunk_size=baseline_chunk_size,
            num_trial_sample=num_trial_sample
        )

    def read_record(self,
                    record: str,
                    root_path: str = None,
                    sampling_rate: int = None,
                    num_channel: int = None,
                    num_baseline: int = None,
                    baseline_chunk_size: int = None,
                    **kwargs) -> Dict:
        if root_path is None:
            root_path = self.root_path
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if num_channel is None:
            num_channel = self.num_channel
        if num_baseline is None:
            num_baseline = self.num_baseline
        if baseline_chunk_size is None:
            baseline_chunk_size = self.baseline_chunk_size
        trial_dir = os.path.join(root_path, record)

        # record the common meta info for the trial
        label_file = os.path.join(trial_dir, 'session.xml')
        with open(label_file) as f:
            label_info = xmltodict.parse('\n'.join(f.readlines()))
        label_info = json.loads(json.dumps(label_info))['session']

        # extract signals
        sample_file = glob.glob(str(os.path.join(trial_dir, '*.bdf')))[0]

        raw = mne.io.read_raw_bdf(sample_file,
                                  preload=True,
                                  stim_channel='Status')
        events = mne.find_events(raw, stim_channel='Status')

        montage = mne.channels.make_standard_montage(kind='biosemi32')
        raw.set_montage(montage, on_missing='ignore')

        # pick channels
        raw.pick_channels(raw.ch_names[:num_channel])

        start_samp, end_samp = events[0][0] + 1, events[1][0] - 1

        # extract baseline signals
        trial_baseline_raw = raw.copy().crop(raw.times[0],
                                             raw.times[end_samp])
        trial_baseline_raw = trial_baseline_raw.resample(sampling_rate)

        trial_baseline_sample = trial_baseline_raw.to_data_frame().to_numpy(
        )[:, 1:].swapaxes(1, 0)  # channel(32), timestep(30 * 128)
        trial_baseline_sample = trial_baseline_sample[:, :num_baseline *
                                                      baseline_chunk_size]
        trial_baseline_sample = trial_baseline_sample.reshape(
            num_channel, num_baseline,
            baseline_chunk_size).mean(axis=1)  # channel(32), timestep(128)

        # extract experimental signals
        trial_raw = raw.copy().crop(raw.times[start_samp],
                                    raw.times[end_samp])
        trial_raw = trial_raw.resample(sampling_rate)
        trial_samples = trial_raw.to_data_frame().to_numpy()[:,
                                                             1:].swapaxes(
                                                                 1, 0)

        return {
            'trial_samples': trial_samples,
            'label_info': label_info,
            'trial_baseline_sample': trial_baseline_sample
        }

    @staticmethod
    def fake_record(record: str, **kwargs) -> Dict:
        num_channels = 32
        sampling_rate = 128
        duration = 30

        trial_samples = np.random.randn(num_channels, sampling_rate * duration)
        trial_baseline_sample = np.random.randn(num_channels, sampling_rate)

        label_info = {
            '@mediaFile': record,
            '@cutLenSec': duration,
            '@feltArsl': np.random.randint(1, 10),
            '@feltCtrl': np.random.randint(1, 10),
            '@feltEmo': np.random.randint(1, 10),
            '@feltPred': np.random.randint(1, 10),
            '@feltVlnc': np.random.randint(1, 10),
            '@isStim': 1,
            'subject': {
                '@id': np.random.randint(1, 33)
            }
        }

        return {
            'trial_samples': trial_samples,
            'label_info': label_info,
            'trial_baseline_sample': trial_baseline_sample
        }

    def process_record(self, record: str, trial_samples: np.ndarray, trial_baseline_sample: np.ndarray, label_info: Dict, chunk_size: int = None, overlap: int = None, num_trial_sample: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if num_trial_sample is None:
            num_trial_sample = self.num_trial_sample
        emodims = ['@feltArsl', '@feltCtrl', '@feltEmo',
                   '@feltPred', '@feltVlnc', '@isStim']
        if not '@feltArsl' in label_info:
            yield None
        trial_meta_info = {'subject_id': label_info['subject']['@id'],
                           'trial_id': label_info['@mediaFile'], 'duration': float(label_info['@cutLenSec'])}
        trial_meta_info.update({k[1:]: int(label_info[k]) for k in emodims})
        start_at = 0
        if chunk_size <= 0:
            dynamic_chunk_size = trial_samples.shape[1] - start_at
        else:
            dynamic_chunk_size = chunk_size
        end_at = dynamic_chunk_size
        step = dynamic_chunk_size - overlap
        max_len = trial_samples.shape[1]
        if not num_trial_sample <= 0:
            max_len = min(num_trial_sample * dynamic_chunk_size,
                          trial_samples.shape[1])
        while end_at <= max_len:
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
        files = os.listdir(root_path)
        return files


class MAHNOBDataset(CachedDataset):

    r"""
    MAHNOB-HCI is a multimodal database recorded in response to affective stimuli with the goal of emotion recognition and implicit tagging research. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Soleymani et al.
    - Year: 2011
    - Download URL: https://mahnob-db.eu/hci-tagging/
    - Reference: Soleymani M, Lichtenauer J, Pun T, et al. A multimodal database for affect recognition and implicit tagging[J]. IEEE transactions on affective computing, 2011, 3(1): 42-55.
    - Stimulus: 20 videos from famous movies. Each video clip lasts 34-117 seconds (may not be an integer), in addition to 30 seconds before the beginning of the affective stimuli experience and another 30 seconds after the end.
    - Signals: Electroencephalogram (32 channels at 512Hz), peripheral physiological signals (ECG, GSR, Temp, Resp at 256 Hz), and eye movement signals (at 60Hz) of 30-5=25 subjects (3 subjects with missing data records and 2 subjects with incomplete data records).
    - Rating: Arousal, valence, control and predictability (all ona scale from 1 to 9).

    In order to use this dataset, the download folder :obj:`Sessions` (Physiological files of emotion elicitation) is required, containing the following files:

    .. code-block:: text

        Sessions/
        ├── 1/
        │   ├── Part_1_N_Trial1_emotion.bdf
        │   └── session.xml
        ├── ...
        └── 3810/
            ├── Part_30_S_Trial20_emotion.bdf
            └── session.xml

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import MAHNOBDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_locations


        def preprocess_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                transforms.BandDifferentialEntropy(),
                transforms.ToGrid(get_locations(MAHNOBDataset.channels, MAHNOBDataset.grid)),
            ])
            return {
                **sample,
                'eeg': eeg_transform(sample['eeg']),
            }


        def transform_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            label_transform = transforms.Compose([
                transforms.Select('feltVlnc'),
                transforms.Binary(5.0),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = MAHNOBDataset(
            root_path='./Sessions',
            offline_transform=preprocess_fn,
            online_transform=transform_fn,
        )
        print(dataset[0])

    Another example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import MAHNOBDataset
        from torcheeg import transforms


        def preprocess_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                transforms.To2d(),
            ])
            return {
                **sample,
                'eeg': eeg_transform(sample['eeg']),
            }


        def transform_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            label_transform = transforms.Compose([
                transforms.Select(['feltVlnc', 'feltArsl']),
                transforms.Binary(5.0),
                transforms.BinariesToCategory(),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = MAHNOBDataset(
            root_path='./Sessions',
            offline_transform=preprocess_fn,
            online_transform=transform_fn,
        )
        print(dataset[0])

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import MAHNOBDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_adjacency
        from torcheeg.transforms.graph import ToG


        def transform_fn(sample: dict) -> dict:
            eeg_transform = transforms.Compose([
                ToG(get_adjacency(MAHNOBDataset.channels, MAHNOBDataset.adjacency)),
            ])
            label_transform = transforms.Compose([
                transforms.Select('feltArsl'),
                transforms.Binary(5.0),
            ])
            return {
                'eeg': eeg_transform(sample['eeg']),
                'label': label_transform(sample['info']),
            }


        dataset = MAHNOBDataset(
            root_path='./Sessions',
            online_transform=transform_fn,
        )
        print(dataset[0])

    Args:
        root_path (str): Downloaded data files in bdf and xml (unzipped Sessions.zip) formats (default: :obj:`'./Sessions'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`128`)
        sampling_rate (int): The number of data points taken over a second. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 32 channels are EEG signals. (default: :obj:`32`)
        num_baseline (int): Number of baseline signal chunks used. (default: :obj:`30`)
        baseline_chunk_size (int): Number of data points included in each baseline signal chunk. The baseline signal in the MAHNOB dataset has a total of 512 (downsampled to 128) * 30 data points. (default: :obj:`128`)
        num_trial_sample (int): Number of samples picked from each trial. If set to -1, all samples in trials are used. (default: :obj:`30`)



        offline_transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        online_transform (Callable, optional): Sample transform applied on each read. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = MAHNOBData

    channels = MAHNOBData.channels

    grid = MAHNOBData.grid

    adjacency = MAHNOBData.adjacency

    def __init__(self, root_path: str = './Sessions', chunk_size: int = 128, sampling_rate: int = 128, overlap: int = 0, num_channel: int = 32, num_baseline: int = 30, baseline_chunk_size: int = 128, num_trial_sample: int = 30, offline_transform: Union[None, Callable] = None, online_transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'sampling_rate': sampling_rate, 'overlap': overlap, 'num_channel': num_channel, 'num_baseline': num_baseline, 'baseline_chunk_size':
                  baseline_chunk_size, 'num_trial_sample': num_trial_sample, 'offline_transform': offline_transform, 'online_transform': online_transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'sampling_rate': self.sampling_rate, 'overlap': self.overlap, 'num_channel': self.num_channel, 'num_baseline': self.num_baseline, 'baseline_chunk_size': self.baseline_chunk_size, 'num_trial_sample': self.num_trial_sample, 'offline_transform': self.offline_transform, 'online_transform': self.online_transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
