import logging
import os
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import scipy.io as scio

from ...utils import get_random_dir_path
from ..base_data import BaseData
from ..cached_dataset import CachedDataset
from ..utils import get_grid_adjacency

_log = logging.getLogger('torcheeg')


class BCICIV2aData(BaseData):
    """Read and preprocess records for :class:`BCICIV2aDataset`."""

    channels = [
        'FZ', '0',  '1',  '2',  '3',  '4',  '5',  'C3',
        '6',  'CZ', '7',  'C4', '8',  '9',  '10', '11',
        '12', '13', '14', 'PZ', '15', '16',
    ]

    grid = [
        ['-', '-',  '-',  'FZ', '-',  '-',  '-'],
        ['-', '0',  '1',  '2',  '3',  '4',  '-'],
        ['5', 'C3', '6',  'CZ', '7',  'C4', '8'],
        ['-', '9',  '10', '11', '12', '13', '-'],
        ['-', '-',  '14', 'PZ', '15', '-',  '-'],
        ['-', '-',  '-',  '16', '-',  '-',  '-'],
    ]

    adjacency = get_grid_adjacency(grid)

    def __init__(self,
                 root_path: str = './BCICIV_2a_mat',
                 offset: int = 0,
                 chunk_size: int = 7 * 250,
                 overlap: int = 0,
                 num_channel: int = 22,
                 skip_trial_with_artifacts: bool = False,
                 transform: Union[None, Callable] = None):

        super().__init__(
            transform=transform,
            root_path=root_path,
            offset=offset,
            chunk_size=chunk_size,
            overlap=overlap,
            num_channel=num_channel,
            skip_trial_with_artifacts=skip_trial_with_artifacts
        )

    def read_record(self, record: str, **kwargs) -> Dict:
        a_data = scio.loadmat(record)['data']

        return {
            'a_data': a_data,
        }

    @staticmethod
    def fake_record(**kwargs) -> Dict:
        a_data = np.zeros((1, 9), dtype=object)

        for run_id in range(9):
            X = np.random.randn(10000, 25)
            trials = np.array([0, 300, 600])

            y = np.array([0, 1, 0])
            artifacts = np.array([0, 0, 0])

            inner_struct = np.zeros((1, 1), dtype=object)
            inner_data = [X, trials, y, None, None, artifacts]
            inner_struct[0, 0] = inner_data

            a_data[0, run_id] = inner_struct

        return {
            'record': 'A01T.mat',
            'a_data': a_data
        }

    def process_record(self, record: str, a_data: np.ndarray, offset: int = None, chunk_size: int = None, overlap: int = None, num_channel: int = None, skip_trial_with_artifacts: bool = None, **kwargs):
        if offset is None:
            offset = self.offset
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if num_channel is None:
            num_channel = self.num_channel
        if skip_trial_with_artifacts is None:
            skip_trial_with_artifacts = self.skip_trial_with_artifacts
        if chunk_size <= 0:
            dynamic_chunk_size = 7 * 250
        else:
            dynamic_chunk_size = int(chunk_size)
        file_name = os.path.splitext(os.path.basename(record))[0]
        subject = file_name[:-1]
        session = file_name[-1]
        for run_id in range(0, a_data.size):
            a_data1 = a_data[0, run_id]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_X = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_artifacts = a_data3[5]
            a_X = np.transpose(a_X)
            for trial_id in range(0, a_trial.size):
                trial_meta_info = {'subject_id': subject, 'trial_id': trial_id,
                                   'session': session, 'subject': subject, 'run': run_id}
                if a_artifacts[trial_id] != 0 and skip_trial_with_artifacts:
                    continue
                start_at = int(a_trial[trial_id] + offset)
                end_at = start_at + dynamic_chunk_size
                step = dynamic_chunk_size - overlap
                if trial_id < a_trial.size - 1:
                    trial_end_at = int(a_trial[trial_id + 1])
                else:
                    trial_end_at = a_X.shape[1]
                while end_at <= trial_end_at:
                    record_info = {'start_at': start_at, 'end_at': end_at}
                    record_info.update(trial_meta_info)
                    t_eeg = a_X[:num_channel, start_at:end_at]
                    record_info['label'] = int(a_y[trial_id])
                    yield {'eeg': t_eeg, 'info': record_info}
                    start_at = start_at + step
                    end_at = start_at + dynamic_chunk_size

    def set_records(self, root_path: str = None, **kwargs):
        if root_path is None:
            root_path = self.root_path
        if not (os.path.exists(root_path)):
            raise FileNotFoundError(
                f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
            )

        file_name_list = os.listdir(root_path)
        file_path_list = [
            os.path.join(root_path, file) for file in file_name_list
        ]

        return file_path_list


class BCICIV2aDataset(CachedDataset):

    r"""
    A dataset for motor imagery, BCI Competition 2008 Graz data set A (BCICIV_2a). This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Tangermann et al.
    - Year: 2012
    - Download URL: http://bnci-horizon-2020.eu/database/data-sets
    - Reference: Tangermann M, Müller K R, Aertsen A, et al. Review of the BCI competition IV[J]. Frontiers in neuroscience, 2012: 55.
    - Signals: Electroencephalogram (22 channels at 250Hz) and electrocardiogram (3 channels at 250Hz) of 9 subjects. Two sessions on different days were recorded for each subject. Each session is comprised of 6 runs separated by short breaks. One run consists of 48 trials (12 for each of the four possible classes), yielding a total of 288 trials per session. For more detail, please refer to https://www.bbci.de/competition/iv/desc_2a.pdf
    - Category: The imagination of movement of the left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4).

    In order to use this dataset, the download folder :obj:`BCICIV_2a_mat` is required, containing the following files:

    .. code-block:: text

        BCICIV_2a_mat/
        ├── A01T.mat
        ├── A01E.mat
        ├── ...
        └── A09E.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import BCICIV2aDataset
        from torcheeg import transforms

        def prepare_training_1(sample):
            sample = dict(sample)
            sample['eeg'] = transforms.To2d()(sample['eeg'])
            sample['eeg'] = transforms.ToTensor()(sample['eeg'])
            return {'eeg': sample['eeg'], 'labels': transforms.Compose([transforms.Select('label'), transforms.Lambda(lambda x: x - 1)])(sample['info'])}

        dataset = BCICIV2aDataset(root_path='./BCICIV_2a_mat').with_transform(prepare_training_1)
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 22, 1750])
        # label (int)

    Args:
        root_path (str): Downloaded data files in matlab (unzipped BCICIV_2a_mat.zip) formats (default: :obj:`'./BCICIV_2a_mat'`)
        offset (int): The number of data points to be discarded at the beginning of each trial. (default: :obj:`0`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`7 * 250`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 14 channels are EEG signals. (default: :obj:`22`)
        skip_trial_with_artifacts (bool): Whether to skip trials with artifacts. (default: :obj:`False`)



        transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    """

    data_class = BCICIV2aData

    channels = BCICIV2aData.channels

    grid = BCICIV2aData.grid

    adjacency = BCICIV2aData.adjacency

    def __init__(self, root_path: str = './BCICIV_2a_mat', offset: int = 0, chunk_size: int = 7 * 250, overlap: int = 0, num_channel: int = 22, skip_trial_with_artifacts: bool = False, transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', num_worker: int = 0, verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        params = {'root_path': root_path, 'offset': offset, 'chunk_size': chunk_size, 'overlap': overlap, 'num_channel': num_channel, 'skip_trial_with_artifacts':
                  skip_trial_with_artifacts, 'transform': transform, 'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'num_worker': num_worker, 'verbose': verbose}
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'root_path': self.root_path, 'chunk_size': self.chunk_size, 'overlap': self.overlap, 'num_channel': self.num_channel, 'skip_trial_with_artifacts': self.skip_trial_with_artifacts, 'transform': self.transform, 'io_path': self.io_path, 'io_size': self.io_size, 'io_mode': self.io_mode, 'num_worker': self.num_worker, 'verbose': self.verbose})
