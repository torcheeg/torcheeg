import os
from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
import scipy.io as scio

from ...utils import get_random_dir_path
from ..base_dataset import BaseDataset
from ..utils import get_positions


class M3CVDataset(BaseDataset):
    r'''
    A reliable EEG-based biometric system should be able to withstand changes in an individual's mental state (cross-task test) and still be able to successfully identify an individual after several days (cross-session test). The authors built an EEG dataset M3CV with 106 subjects, two sessions of experiment on different days, and multiple paradigms. Ninety-five of the subjects participated in two sessions of the experiments, separated by more than 6 days. The experiment includes 6 common EEG experimental paradigms including resting state, sensory and cognitive task, and brain-computer interface.

    - Author: Huang et al.
    - Year: 2022
    - Download URL: https://aistudio.baidu.com/aistudio/datasetdetail/151025/0
    - Signals: Electroencephalogram (64 channels and one marker channel at 250Hz).

    In order to use this dataset, the download dataset folder :obj:`aistudio` is required, containing the following files:

    .. code-block:: text

        aistudio/
        ├── Calibration_Info.csv
        ├── Enrollment_Info.csv
        ├── Testing_Info.csv
        ├── Calibration/
        ├── Testing/
        └── Enrollment/

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import M3CVDataset
        from torcheeg import transforms

        dataset = M3CVDataset(root_path='./aistudio',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(get_locations(M3CVDataset.channels, M3CVDataset.grid))
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('SubjectID'),
                                  transforms.StringToNumber()
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1000, 9, 9]),
        # coresponding baseline signal (torch.Tensor[1000, 9, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import M3CVDataset
        from torcheeg import transforms

        dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('SubjectID'),
                                  transforms.StringToNumber()
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 65, 1000]),
        # coresponding baseline signal (torch.Tensor[1, 65, 1000]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import M3CVDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_adjacency
        from torcheeg.transforms.pyg import ToG

        dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              online_transform=transforms.Compose([
                                  ToG(get_adjacency(M3CVDataset.channels, M3CVDataset.adjacency))
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('SubjectID'),
                                  transforms.StringToNumber()
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)

    Args:
        root_path (str): Downloaded data files in pickled python/numpy (unzipped aistudio.zip) formats (default: :obj:`'./aistudio'`)
        subset (str): In the competition, the M3CV dataset is splited into the Enrollment set, Calibration set, and Testing set. Please specify the subset to use, options include Enrollment, Calibration and Testing. (default: :obj:`'Enrollment'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`1000`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 32 channels are EEG signals. (default: :obj:`64`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)    
    '''

    channels = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
        'T7', 'T8', 'P7', 'P8', 'FZ', 'CZ', 'PZ', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5',
        'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2',
        'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5',
        'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8',
        'PO7', 'PO8', 'FPZ', 'CPZ', 'POZ', 'OZ', 'FCZ'
    ]

    grid = [
        ['-', '-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-', '-'],
        ['-', '-', 'AF7', '-', 'AF3', '-', 'AF4', '-', 'AF8', '-', '-'],
        ['-', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', '-'],
        [
            'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
            'FT10'
        ], ['-', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', '-'],
        [
            'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
            'TP10'
        ], ['-', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', '-'],
        ['-', '-', 'PO7', '-', 'PO3', 'POZ', 'PO4', '-', 'PO8', '-', '-'],
        ['-', '-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-', '-']
    ]

    adjacency = {
        'FP1': ['AF3', 'FPZ'],
        'FPZ': ['FP1', 'AF3', 'FP2', 'AF4'],
        'FP2': ['FPZ', 'AF4'],
        'AF7': ['F7', 'F5', 'F3'],
        'AF3': ['F3', 'FP1', 'F1', 'FPZ', 'FZ'],
        'AF4': ['FPZ', 'FZ', 'FP2', 'F2', 'F4'],
        'AF8': ['F4', 'F6', 'F8'],
        'F7': ['FT9', 'FT7', 'AF7', 'F5', 'FC5'],
        'F5': ['F7', 'FT7', 'AF7', 'FC5', 'F3', 'FC3'],
        'F3': ['AF7', 'F5', 'FC5', 'FC3', 'AF3', 'F1', 'FC1'],
        'F1': ['F3', 'FC3', 'AF3', 'FC1', 'FZ', 'FCZ'],
        'FZ': ['AF3', 'F1', 'FC1', 'FCZ', 'AF4', 'F2', 'FC2'],
        'F2': ['FZ', 'FCZ', 'AF4', 'FC2', 'F4', 'FC4'],
        'F4': ['AF4', 'F2', 'FC2', 'FC4', 'AF8', 'F6', 'FC6'],
        'F6': ['F4', 'FC4', 'AF8', 'FC6', 'F8', 'FT8'],
        'F8': ['AF8', 'F6', 'FC6', 'FT8', 'FT10'],
        'FT9': ['F7', 'FT7', 'T7'],
        'FT7': ['FT9', 'F7', 'T7', 'F5', 'FC5', 'C5'],
        'FC5': ['F7', 'FT7', 'T7', 'F5', 'C5', 'F3', 'FC3', 'C3'],
        'FC3': ['F5', 'FC5', 'C5', 'F3', 'C3', 'F1', 'FC1', 'C1'],
        'FC1': ['F3', 'FC3', 'C3', 'F1', 'C1', 'FZ', 'FCZ', 'CZ'],
        'FCZ': ['F1', 'FC1', 'C1', 'FZ', 'CZ', 'F2', 'FC2', 'C2'],
        'FC2': ['FZ', 'FCZ', 'CZ', 'F2', 'C2', 'F4', 'FC4', 'C4'],
        'FC4': ['F2', 'FC2', 'C2', 'F4', 'C4', 'F6', 'FC6', 'C6'],
        'FC6': ['F4', 'FC4', 'C4', 'F6', 'C6', 'F8', 'FT8', 'T8'],
        'FT8': ['F6', 'FC6', 'C6', 'F8', 'T8', 'FT10'],
        'FT10': ['F8', 'FT8', 'T8'],
        'T7': ['FT9', 'TP9', 'FT7', 'TP7', 'FC5', 'C5', 'CP5'],
        'C5': ['FT7', 'T7', 'TP7', 'FC5', 'CP5', 'FC3', 'C3', 'CP3'],
        'C3': ['FC5', 'C5', 'CP5', 'FC3', 'CP3', 'FC1', 'C1', 'CP1'],
        'C1': ['FC3', 'C3', 'CP3', 'FC1', 'CP1', 'FCZ', 'CZ', 'CPZ'],
        'CZ': ['FC1', 'C1', 'CP1', 'FCZ', 'CPZ', 'FC2', 'C2', 'CP2'],
        'C2': ['FCZ', 'CZ', 'CPZ', 'FC2', 'CP2', 'FC4', 'C4', 'CP4'],
        'C4': ['FC2', 'C2', 'CP2', 'FC4', 'CP4', 'FC6', 'C6', 'CP6'],
        'C6': ['FC4', 'C4', 'CP4', 'FC6', 'CP6', 'FT8', 'T8', 'TP8'],
        'T8': ['FC6', 'C6', 'CP6', 'FT8', 'TP8', 'FT10', 'TP10'],
        'TP9': ['T7', 'TP7', 'P7'],
        'TP7': ['TP9', 'T7', 'P7', 'C5', 'CP5', 'P5'],
        'CP5': ['T7', 'TP7', 'P7', 'C5', 'P5', 'C3', 'CP3', 'P3'],
        'CP3': ['C5', 'CP5', 'P5', 'C3', 'P3', 'C1', 'CP1', 'P1'],
        'CP1': ['C3', 'CP3', 'P3', 'C1', 'P1', 'CZ', 'CPZ', 'PZ'],
        'CPZ': ['C1', 'CP1', 'P1', 'CZ', 'PZ', 'C2', 'CP2', 'P2'],
        'CP2': ['CZ', 'CPZ', 'PZ', 'C2', 'P2', 'C4', 'CP4', 'P4'],
        'CP4': ['C2', 'CP2', 'P2', 'C4', 'P4', 'C6', 'CP6', 'P6'],
        'CP6': ['C4', 'CP4', 'P4', 'C6', 'P6', 'T8', 'TP8', 'P8'],
        'TP8': ['C6', 'CP6', 'P6', 'T8', 'P8', 'TP10'],
        'TP10': ['T8', 'TP8', 'P8'],
        'P7': ['TP9', 'TP7', 'CP5', 'P5', 'PO7'],
        'P5': ['TP7', 'P7', 'CP5', 'PO7', 'CP3', 'P3'],
        'P3': ['CP5', 'P5', 'PO7', 'CP3', 'CP1', 'P1', 'PO3'],
        'P1': ['CP3', 'P3', 'CP1', 'PO3', 'CPZ', 'PZ', 'POZ'],
        'PZ': ['CP1', 'P1', 'PO3', 'CPZ', 'POZ', 'CP2', 'P2', 'PO4'],
        'P2': ['CPZ', 'PZ', 'POZ', 'CP2', 'PO4', 'CP4', 'P4'],
        'P4': ['CP2', 'P2', 'PO4', 'CP4', 'CP6', 'P6', 'PO8'],
        'P6': ['CP4', 'P4', 'CP6', 'PO8', 'TP8', 'P8'],
        'P8': ['CP6', 'P6', 'PO8', 'TP8', 'TP10'],
        'PO7': ['P7', 'P5', 'P3'],
        'PO3': ['P3', 'P1', 'O1', 'PZ', 'POZ', 'OZ'],
        'POZ': ['P1', 'PO3', 'O1', 'PZ', 'OZ', 'P2', 'PO4', 'O2'],
        'PO4': ['PZ', 'POZ', 'OZ', 'P2', 'O2', 'P4'],
        'PO8': ['P4', 'P6', 'P8'],
        'O1': ['PO3', 'POZ', 'OZ'],
        'OZ': ['PO3', 'O1', 'POZ', 'PO4', 'O2'],
        'O2': ['POZ', 'OZ', 'PO4']
    }

    positions = get_positions(channels)

    def __init__(self,
                 root_path: str = './aistudio',
                 subset: str = 'Enrollment',
                 chunk_size: int = 1000,
                 overlap: int = 0,
                 num_channel: int = 64,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'subset': subset,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def read_record(record: Dict,
                    root_path: str = './aistudio',
                    subset: str = 'Enrollment', **kwargs) -> Dict:
        epoch_id = record['EpochID']
        trial_samples = scio.loadmat(
            os.path.join(root_path, subset, epoch_id))['epoch_data']
        return {
            'trial_samples': trial_samples,
        }

    @staticmethod
    def fake_record(record: Dict, **kwargs) -> Dict:
        num_channel = 64
        num_timepoint = 2000

        trial_samples = np.random.rand(num_channel, num_timepoint)
        return {
            'record': {
                'EpochID': 'fake_epoch',
                'SubjectID': 'fake_subject',
                'Session': 'fake_session',
                'Task': 'fake_task',
                'Usage': 'fake_usage',
            },
            'trial_samples': trial_samples,
        }

    @staticmethod
    def process_record(record: Dict,
                       trial_samples: np.ndarray,
                       chunk_size: int = 1000,
                       overlap: int = 0,
                       num_channel: int = 64,
                       offline_transform: Union[None, Callable] = None,
                       before_trial: Union[None, Callable] = None,
                       **kwargs):
        write_pointer = 0

        epoch_meta_info = {
            'epoch_id': record['EpochID'],
            'subject_id': record['SubjectID'],
            'session': record['Session'],
            'task': record['Task'],
            'usage': record['Usage'],
        }

        epoch_id = epoch_meta_info['epoch_id']

        if before_trial:
            trial_samples = before_trial(trial_samples)

        start_at = 0
        if chunk_size <= 0:
            dynamic_chunk_size = trial_samples.shape[1] - start_at
        else:
            dynamic_chunk_size = chunk_size

        # chunk with chunk size
        end_at = dynamic_chunk_size
        # calculate moving step
        step = dynamic_chunk_size - overlap

        while end_at <= trial_samples.shape[1]:
            clip_sample = trial_samples[:num_channel, start_at:end_at]
            t_eeg = clip_sample

            if not offline_transform is None:
                t_eeg = offline_transform(eeg=clip_sample)['eeg']

            clip_id = f'{epoch_id}_{write_pointer}'
            write_pointer += 1

            # record meta info for each signal
            record_info = {
                'start_at': start_at,
                'end_at': end_at,
                'clip_id': clip_id
            }
            record_info.update(epoch_meta_info)
            yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

            start_at = start_at + step
            end_at = start_at + dynamic_chunk_size

    def set_records(self,
                    root_path: str = './aistudio',
                    subset: str = 'Enrollment',
                    **kwargs):
        if not (os.path.exists(root_path)):
            raise FileNotFoundError(
                f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
            )

        if not (subset in ['Enrollment', 'Calibration', 'Testing']):
            raise ValueError(
                f"Unavailable subset name {subset}, and available options include 'Enrollment', 'Calibration', and 'Testing'."
            )

        df = pd.read_csv(os.path.join(root_path, f'{subset}_Info.csv'))
        return df.to_dict(orient='records')

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
                'subset': self.subset,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
