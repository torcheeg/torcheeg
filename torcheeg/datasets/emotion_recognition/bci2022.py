import os
from typing import Any, Callable, Dict, Tuple, Union

import joblib
import numpy as np

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset

_FIRST_BATCH_CHANNEL = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T7', 'T8', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ',
    'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2'
]

_SECOND_BATCH_CHANNEL = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4',
    'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2', 'A2', 'A1'
]

_FIRST_TO_SECOND_ORDER = [
    _FIRST_BATCH_CHANNEL.index(c) for c in _SECOND_BATCH_CHANNEL
]

_VALENCE_DICT = {
    1: -1,  # negative
    2: -1,
    3: -1,
    4: -1,
    5: -1,
    6: -1,
    7: -1,
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: 0,  # neutral
    14: 0,
    15: 0,
    16: 0,
    17: 1,  # positive
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1
}

_EMOTION_DICT = {
    1: 0,  # anger
    2: 0,
    3: 0,
    4: 1,  # disgust
    5: 1,
    6: 1,
    7: 2,  # fear
    8: 2,
    9: 2,
    10: 3,  # sadness
    11: 3,
    12: 3,
    13: 4,  # neutral
    14: 4,
    15: 4,
    16: 4,
    17: 5,  # amusement
    18: 5,
    19: 5,
    20: 6,  # excitation
    21: 6,
    22: 6,
    23: 7,  # happy
    24: 7,
    25: 7,
    26: 8,  # warmth
    27: 8,
    28: 8
}


class BCI2022Data(BaseData):
    """Read and preprocess records for :class:`BCI2022Dataset`."""

    channels = [
        'FP1', 'FP2', 'FZ',  'F3',  'F4',  'F7',  'F8',  'FC1',
        'FC2', 'FC5', 'FC6', 'CZ',  'C3',  'C4',  'T7',  'T8',
        'CP1', 'CP2', 'CP5', 'CP6', 'PZ',  'P3',  'P4',  'P7',
        'P8',  'PO3', 'PO4', 'OZ',  'O1',  'O2',
    ]

    grid = [
        ['-',  '-',   '-',  'FP1', '-',  'FP2', '-',  '-',   '-'],
        ['F7', '-',   'F3', '-',   'FZ', '-',   'F4', '-',   'F8'],
        ['-',  'FC5', '-',  'FC1', '-',  'FC2', '-',  'FC6', '-'],
        ['T7', '-',   'C3', '-',   'CZ', '-',   'C4', '-',   'T8'],
        ['-',  'CP5', '-',  'CP1', '-',  'CP2', '-',  'CP6', '-'],
        ['P7', '-',   'P3', '-',   'PZ', '-',   'P4', '-',   'P8'],
        ['-',  '-',   '-',  'PO3', '-',  'PO4', '-',  '-',   '-'],
        ['-',  '-',   '-',  'O1',  'OZ', 'O2',  '-',  '-',   '-'],
    ]

    adjacency = {
        'FP1': ['F3', 'FZ'],
        'FP2': ['FZ', 'F4'],
        'F7':  ['FC5'],
        'F3':  ['FC1', 'FC5'],
        'FZ':  ['AF4', 'FC2', 'FC1'],
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
                 root_path: str = './2022EmotionPublic/TrainSet/',
                 chunk_size: int = 250,
                 overlap: int = 0,
                 channel_num: int = 30,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            chunk_size=chunk_size,
            overlap=overlap,
            channel_num=channel_num
        )

    def read_record(self, record: str, **kwargs) -> Dict:
        file_name = os.path.basename(record)  # an element from file name list
        samples = joblib.load(os.path.join(
            record, f'{file_name}.pkl'))  # channel(33), timestep(n*250)
        return {
            'samples': samples
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        duration = 10
        num_samples = duration * 550
        num_channels = 33

        eeg_data = np.random.randn(num_channels - 1, num_samples)

        events = np.zeros(num_samples)

        events[500] = 1
        events[525] = 240
        events[1500] = 241

        events[2000] = 2
        events[2025] = 240
        events[3000] = 241

        samples = np.vstack([eeg_data, events])

        return {
            'samples': samples
        }

    def process_record(self, record: str, samples: np.ndarray, chunk_size: int = None, overlap: int = None, channel_num: int = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if channel_num is None:
            channel_num = self.channel_num
        record_name = os.path.basename(record)
        reorder = 'TrainSet_first_batch' in record_name
        subject = record_name
        events = samples[-1]
        if reorder:
            samples = samples.take(_FIRST_TO_SECOND_ORDER, axis=0)
        trial_id = 0
        video_id = None
        start_at = None
        end_at = None
        for i, event in enumerate(events):
            if event in list(range(1, 29)):
                video_id = event
            elif event == 240:
                start_at = i
            elif event == 241:
                end_at = i
                if not (not video_id is None and (not start_at is None)):
                    raise ValueError(
                        f'Parse event fail for trial {trial_id} with video_id={video_id}, start_at={start_at}, end_at={end_at}!')
                trial_meta_info = {'trial_id': trial_id, 'video_id': video_id, 'subject_id': subject,
                                   'valence': _VALENCE_DICT[video_id], 'emotion': _EMOTION_DICT[video_id]}
                cur_start_at = start_at
                if chunk_size <= 0:
                    dynamic_chunk_size = end_at - start_at
                else:
                    dynamic_chunk_size = chunk_size
                cur_end_at = cur_start_at + dynamic_chunk_size
                step = dynamic_chunk_size - overlap
                while cur_end_at <= end_at:
                    t_eeg = samples[:channel_num, cur_start_at:cur_end_at]
                    record_info = {'start_at': cur_start_at,
                                   'end_at': cur_end_at}
                    record_info.update(trial_meta_info)
                    yield {'eeg': t_eeg, 'info': record_info}
                    cur_start_at = cur_start_at + step
                    cur_end_at = cur_start_at + dynamic_chunk_size
                trial_id += 1
                video_id = None
                start_at = None
                end_at = None

    def set_records(self, root_path: str = None, **kwargs):
        if root_path is None:
            root_path = self.root_path
        if not (os.path.exists(root_path)):
            raise FileNotFoundError(
                f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
            )
        outputs = []
        for train_set_batch in [
                'TrainSet_first_batch', 'TrainSet_second_batch'
        ]:
            # loop to access the first batch files
            file_list = os.listdir(os.path.join(root_path, train_set_batch))
            for file in file_list:
                # loop to access the second batch files
                file_path = os.path.join(root_path, train_set_batch, file)
                outputs.append(file_path)
        return outputs


class BCI2022Dataset(CachedDataset):

    r"""
    The 2022 EMOTION_BCI competition aims at tackling the cross-subject emotion recognition challenge and provides participants with a batch of EEG data from 80 participants with known emotional state information. Participants are required to establish an EEG computing model with cross-individual emotion recognition ability. The subjects' EEG data were used for real-time emotion recognition. This class generates training samples and test samples according to the given parameters and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Please refer to the downloaded URL.
    - Year: 2022
    - Download URL: https://oneuro.cn/n/competitiondetail/2022_emotion_bci/doc0
    - Reference: Please refer to the downloaded URL.
    - Stimulus: video clips.
    - Signals: Electroencephalogram (30 channels at 250Hz) and two channels of left/right mastoid signals from 80 subjects.
    - Rating: 28 video clips are annotated in valence and discrete emotion dimensions. The valence is divided into positive (1), negative (-1), and neutral (0). Discrete emotions are divided into anger (0), disgust (1), fear (2), sadness (3), neutral (4), amusement (5), excitation (6), happiness (7), and warmth (8).

    In order to use this dataset, the download folder :obj:`TrainSet` is required, containing the following files:

    .. code-block:: text

        TrainSet_first_batch/
        ├── sub1/
        ├── sub10/
        ├── sub11/
        └── ...

        TrainSet_second_batch/
        ├── sub55/
        ├── sub57/
        ├── sub59/
        └── ...

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import BCI2022Dataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_locations

        def preprocess_sample_1(sample):
            sample = dict(sample)
            sample['eeg'] = transforms.BandDifferentialEntropy()(sample['eeg'])
            sample['eeg'] = transforms.ToGrid(get_locations(BCI2022Dataset.channels, BCI2022Dataset.grid))(sample['eeg'])
            return sample

        def prepare_training_1(sample):
            sample = dict(sample)
            sample['eeg'] = transforms.ToTensor()(sample['eeg'])
            return {'eeg': sample['eeg'], 'labels': transforms.Select('emotion')(sample['info'])}

        dataset = BCI2022Dataset(root_path='./TrainSet', transform=preprocess_sample_1).with_transform(prepare_training_1)
        print(dataset[0])
        # EEG signal (torch.Tensor[4, 8, 9]),
        # coresponding baseline signal (torch.Tensor[4, 8, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import BCI2022Dataset
        from torcheeg import transforms

        def prepare_training_1(sample):
            sample = dict(sample)
            sample['eeg'] = transforms.ToTensor()(sample['eeg'])
            sample['eeg'] = transforms.To2d()(sample['eeg'])
            return {'eeg': sample['eeg'], 'labels': transforms.Compose([transforms.Select('emotion'), transforms.Lambda(lambda x: x + 1)])(sample['info'])}

        dataset = BCI2022Dataset(root_path='./TrainSet').with_transform(prepare_training_1)

        print(dataset[0])
        # EEG signal (torch.Tensor[30, 250]),
        # coresponding baseline signal (torch.Tensor[30, 250]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import BCI2022Dataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_adjacency

        def prepare_training_1(sample):
            sample = dict(sample)
            sample['eeg'] = transforms.ToG(get_adjacency(BCI2022Dataset.channels, BCI2022Dataset.adjacency))(sample['eeg'])
            return {'eeg': sample['eeg'], 'labels': transforms.Compose([transforms.Select('emotion'), transforms.Lambda(lambda x: x + 1)])(sample['info'])}

        dataset = BCI2022Dataset(root_path='./TrainSet').with_transform(prepare_training_1)

        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)

    Args:
        root_path (str): Downloaded data files in pickle (the TrainSet folder in unzipped 2022EmotionPublic.zip) formats (default: :obj:`'./TrainSet'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`250`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        channel_num (int): Number of channels used, of which the first 30 channels are EEG signals. (default: :obj:`30`)



        transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = BCI2022Data

    channels = BCI2022Data.channels

    grid = BCI2022Data.grid

    adjacency = BCI2022Data.adjacency

    def __init__(self, root_path: str = './2022EmotionPublic/TrainSet/', chunk_size: int = 250, overlap: int = 0, channel_num: int = 30, transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'chunk_size': chunk_size, 'overlap': overlap, 'channel_num': channel_num,
                  'transform': transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'channel_num': self.channel_num, 'transform': self.transform, 'num_worker': self.num_worker, 'verbose': self.verbose, 'io_size': self.io_size})
