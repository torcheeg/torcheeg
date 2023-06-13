import os
import re
from typing import Any, Callable, Dict, Tuple, Union

import scipy.io as scio

from torcheeg.io import EEGSignalIO, MetaInfoIO

from ...constants.emotion_recognition.seed_iv import (
    SEED_IV_ADJACENCY_MATRIX, SEED_IV_CHANNEL_LOCATION_DICT)
from ..base_dataset import BaseDataset


class SEEDIVDataset(BaseDataset):
    r'''
    The SEED-IV dataset provided by the BCMI laboratory, which is led by Prof. Bao-Liang Lu. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Zheng et al.
    - Year: 2018
    - Download URL: https://ieeexplore.ieee.org/abstract/document/8283814
    - Reference: Zheng W L, Liu W, Lu Y, et al. Emotionmeter: A multimodal framework for recognizing human emotions[J]. IEEE transactions on cybernetics, 2018, 49(3): 1110-1122.
    - Stimulus: 168 film clips.
    - Signals: Electroencephalogram (62 channels at 200Hz) and eye movement data of 15 subjects (8 females). Each subject conducts the experiments in three sessions, and each session contains 24 trials (6 per emotional category) totally 15 people x 3 sessions x 24 trials.
    - Rating: neutral (0), sad (1), fear (2), and happy (3).

    In order to use this dataset, the download folder :obj:`eeg_raw_data` is required, containing the following files:
    
    - label.mat
    - readme.txt
    - 10_20131130.mat
    - ...
    - 9_20140704.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        dataset = SEEDIVDataset(io_path=f'./seed_iv',
                              root_path='./eeg_raw_data',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(SEEDIV_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: x + 1)
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[4, 9, 9]),
        # coresponding baseline signal (torch.Tensor[4, 9, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = SEEDIVDataset(io_path=f'./seed_iv',
                              root_path='./eeg_raw_data',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch.Tensor[62, 200]),
        # coresponding baseline signal (torch.Tensor[62, 200]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python
    
        dataset = SEEDIVDataset(io_path=f'./seed_iv',
                              root_path='./eeg_raw_data',
                              online_transform=transforms.Compose([
                                  ToG(SEED_IV_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
        
    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = SEEDIVDataset(io_path=f'./seed',
                              root_path='./eeg_raw_data',
                              online_transform=transforms.Compose([
                                  ToG(SEED_IV_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Select('emotion'),
                              num_worker=4)
            print(dataset[0])
            # EEG signal (torch_geometric.data.Data),
            # coresponding baseline signal (torch_geometric.data.Data),
            # label (int)

    Args:
        root_path (str): Downloaded data files in matlab (unzipped eeg_raw_data.zip) formats (default: :obj:`'./eeg_raw_data'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`800`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 62 channels are EEG signals. (default: :obj:`62`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/seed`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)    
    '''
    channel_location_dict = SEED_IV_ADJACENCY_MATRIX
    adjacency_matrix = SEED_IV_CHANNEL_LOCATION_DICT

    def __init__(self,
                 root_path: str = './eeg_raw_data',
                 chunk_size: int = 800,
                 overlap: int = 0,
                 num_channel: int = 62,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 io_path: str = './io/seed',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):
        # pass all arguments to super class
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
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose,
            'in_memory': in_memory
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def _load_data(file: Any = None,
                   chunk_size: int = 800,
                   overlap: int = 0,
                   num_channel: int = 62,
                   before_trial: Union[None, Callable] = None,
                   offline_transform: Union[None, Callable] = None,
                   after_trial: Union[None, Callable] = None,
                   **kwargs):
        file_path = file  # an element from file name list

        session_id = os.path.basename(os.path.dirname(file_path))
        _, file_name = os.path.split(file_path)

        subject = int(os.path.basename(file_name).split('.')[0].split('_')
                      [0])  # subject (15)
        date = int(os.path.basename(file_name).split('.')[0].split('_')
                   [1])  # period (3)

        samples = scio.loadmat(file_path,
                               verify_compressed_data_integrity=False
                               )  # trial (15), channel(62), timestep(n*200)
        # label file
        labels = [
            [
                1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0,
                3, 0, 3
            ],
            [
                2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0,
                1, 3, 1
            ],
            [
                1, 2,
                2, 1,
                3, 3,
                3, 1,
                1, 2,
                1, 0,
                2, 3,
                3, 0,
                2, 3,
                0, 0,
                2, 0,
                1, 0
            ]
        ]  # The labels with 0, 1, 2, and 3 denote the ground truth, neutral, sad, fear, and happy emotions, respectively.
        session_labels = labels[int(session_id) - 1]

        trial_name_ids = [(trial_name,
                           int(re.findall(r".*_eeg(\d+)", trial_name)[0]))
                          for trial_name in samples.keys()
                          if 'eeg' in trial_name]

        write_pointer = 0
        # loop for each trial
        for trial_name, trial_id in trial_name_ids:
            trial_samples = samples[trial_name]  # channel(62), timestep(n*200)
            if before_trial:
                trial_samples = before_trial(trial_samples)

            # record the common meta info
            trial_meta_info = {
                'subject_id': subject,
                'trial_id': trial_id,
                'session_id': session_id,
                'emotion': int(session_labels[trial_id - 1]),
                'date': date
            }

            # extract experimental signals
            start_at = 0
            if chunk_size <= 0:
                chunk_size = trial_samples.shape[1] - start_at

            # chunk with chunk size
            end_at = chunk_size
            # calculate moving step
            step = chunk_size - overlap

            trial_queue = []
            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:num_channel, start_at:end_at]

                t_eeg = clip_sample
                if not offline_transform is None:
                    t_eeg = offline_transform(eeg=clip_sample)['eeg']

                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1

                # record meta info for each signal
                record_info = {
                    'start_at': start_at,
                    'end_at': end_at,
                    'clip_id': clip_id
                }
                record_info.update(trial_meta_info)
                if after_trial:
                    trial_queue.append({
                        'eeg': t_eeg,
                        'key': clip_id,
                        'info': record_info
                    })
                else:
                    yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

                start_at = start_at + step
                end_at = start_at + chunk_size

            if len(trial_queue) and after_trial:
                trial_queue = after_trial(trial_queue)
                for obj in trial_queue:
                    assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
                    yield obj

    @staticmethod
    def _set_files(root_path: str = './eeg_raw_data', **kwargs):
        session_list = ['1', '2', '3']
        file_path_list = []
        for session in session_list:
            session_root_path = os.path.join(root_path, session)
            for file_name in os.listdir(session_root_path):
                file_path_list.append(os.path.join(session_root_path,
                                                   file_name))
        return file_path_list

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg = self.read_eeg(eeg_index)

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
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
