import os
from typing import Any, Callable, Dict, Tuple, Union

import pandas as pd
import scipy.io as scio

from torcheeg.io import EEGSignalIO, MetaInfoIO

from ..base_dataset import BaseDataset


class M3CVDataset(BaseDataset):
    r'''
    A reliable EEG-based biometric system should be able to withstand changes in an individual's mental state (cross-task test) and still be able to successfully identify an individual after several days (cross-session test). The authors built an EEG dataset M3CV with 106 subjects, two sessions of experiment on different days, and multiple paradigms. Ninety-five of the subjects participated in two sessions of the experiments, separated by more than 6 days. The experiment includes 6 common EEG experimental paradigms including resting state, sensory and cognitive task, and brain-computer interface.
    
    - Author: Huang et al.
    - Year: 2022
    - Download URL: https://aistudio.baidu.com/aistudio/datasetdetail/151025/0
    - Signals: Electroencephalogram (64 channels and one marker channel at 250Hz).

    In order to use this dataset, the download dataset folder :obj:`aistudio` is required, containing the following files:
    
    - Calibration_Info.csv
    - Enrollment_Info.csv
    - Testing_Info.csv
    - Calibration (unzipped Calibration.zip)
    - Testing (unzipped Testing.zip)
    - Enrollment (unzipped Enrollment.zip)

    An example dataset for CNN-based methods:

    .. code-block:: python
    
        dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(M3CV_CHANNEL_LOCATION_DICT)
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
    
        dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              online_transform=transforms.Compose([
                                  ToG(M3CV_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('SubjectID'),
                                  transforms.StringToNumber()
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              online_transform=transforms.Compose([
                                  ToG(M3CV_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('SubjectID'),
                                  transforms.StringToNumber()
                              ]),
                              num_worker=4)
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
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/m3cv`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)    
    '''
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
                 io_path: str = './io/m3cv',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):
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
            'verbose': verbose,
            'in_memory': in_memory
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def _load_data(file: Any = None,
                   root_path: str = './aistudio',
                   subset: str = 'Enrollment',
                   chunk_size: int = 1000,
                   overlap: int = 0,
                   num_channel: int = 64,
                   offline_transform: Union[None, Callable] = None,
                   before_trial: Union[None, Callable] = None,
                   after_trial: Union[None, Callable] = None,
                   **kwargs):
        start_idx, end_idx = file

        df = pd.read_csv(os.path.join(root_path, f'{subset}_Info.csv'))
        df = df.iloc[start_idx:end_idx]

        # calculate moving step
        write_pointer = 0

        start_epoch = None

        for _, epoch_info in df.iterrows():
            epoch_meta_info = {
                'epoch_id': epoch_info['EpochID'],
                'subject_id': epoch_info['SubjectID'],
                'session': epoch_info['Session'],
                'task': epoch_info['Task'],
                'usage': epoch_info['Usage'],
            }

            epoch_id = epoch_meta_info['epoch_id']

            if start_epoch is None:
                start_epoch = epoch_id

            trial_samples = scio.loadmat(
                os.path.join(root_path, subset, epoch_id))['epoch_data']
            if before_trial:
                trial_samples = before_trial(trial_samples)

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

                clip_id = f'after{start_epoch}_{write_pointer}'
                write_pointer += 1

                # record meta info for each signal
                record_info = {
                    'start_at': start_at,
                    'end_at': end_at,
                    'clip_id': clip_id
                }
                record_info.update(epoch_meta_info)
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
    def _set_files(root_path: str = './aistudio',
                   subset: str = 'Enrollment',
                   **kwargs):
        assert subset in [
            'Enrollment', 'Calibration', 'Testing'
        ], f"Unavailable subset name {subset}, and available options include 'Enrollment', 'Calibration', and 'Testing'."

        df = pd.read_csv(os.path.join(root_path, f'{subset}_Info.csv'))
        # split the dataset into 60 blocks, each of which contains len(df) // 60 samples, the start and end index of each block are recorded in block_list
        block_list = []
        for i in range(60):
            start = i * len(df) // 60
            end = (i + 1) * len(df) // 60
            block_list.append((start, end))
        # the last block contains the remaining samples
        if len(df) % 60 != 0:
            block_list[-1] = (block_list[-1][0], len(df))
        return block_list

    def __getitem__(self, index: int) -> Tuple:
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
