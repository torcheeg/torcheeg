import os
import re
from typing import Callable, Dict, Tuple, Union

import numpy as np
import scipy.io as scio

from ...utils import get_random_dir_path
from ..base_dataset import BaseDataset
from ..utils import get_positions


class TSUBenckmarkDataset(BaseDataset):
    r'''
    The benchmark dataset for SSVEP-Based brain-computer interfaces (TSUBenckmark) is provided by the Tsinghua BCI Lab. It presents a benchmark steady-state visual evoked potential (SSVEP) dataset acquired with a 40-target brain-computer interface (BCI) speller. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Wang et al.
    - Year: 2016
    - Download URL: http://bci.med.tsinghua.edu.cn/
    - Reference: Wang Y, Chen X, Gao X, et al. A benchmark dataset for SSVEP-based brain-computer interfaces[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2016, 25(10): 1746-1752.
    - Stimulus: Each trial started with a visual cue (a red square) indicating a target stimulus. The cue appeared for 0.5s on the screen. Subjects were asked to shift their gaze to the target as soon as possible within the cue duration. Following the cue offset, all stimuli started to flicker on the screen concurrently and lasted 5s. After stimulus offset, the screen was blank for 0.5s before the next trial began, which allowed the subjects to have short breaks between consecutive trials.
    - Signals: Electroencephalogram (64 channels at 250Hz) of 35 subjects. For each subject, the experiment consisted of 6 blocks. Each block contained 40 trials corresponding to all 40 characters indicated in a random order. Totally 35 people x 6 blocks x 40 trials.
    - Rating: Frequency and phase values for the 40 trials.

    In order to use this dataset, the download folder :obj:`data_preprocessed_python` is required, containing the following files:

    .. code-block:: text

        data_preprocessed_python/
        ├── Readme.txt
        ├── Sub_info.txt
        ├── 64-channels.loc
        ├── Freq_Phase.mat
        ├── S1.mat
        ├── ...
        └── S35.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import TSUBenckmarkDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_locations

        dataset = TSUBenckmarkDataset(root_path='./TSUBenchmark',
                                      offline_transform=transforms.Compose([
                                          transforms.BandDifferentialEntropy(),
                                          transforms.ToGrid(get_locations(TSUBenckmarkDataset.channels, TSUBenckmarkDataset.grid))
                                      ]),
                                      online_transform=transforms.ToTensor(),
                                      label_transform=transforms.Select(['trial_id']))
        print(dataset[0])
        # EEG signal (torch.Tensor[250, 10, 11]),
        # coresponding baseline signal (torch.Tensor[250, 10, 11]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = TSUBenckmarkDataset(root_path='./TSUBenchmark',
                                      online_transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.To2d()
                                      ]),
                                      label_transform=transforms.Select(['trial_id']))
        print(dataset[0])
        # EEG signal (torch.Tensor[64, 250]),
        # coresponding baseline signal (torch.Tensor[64, 250]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import TSUBenckmarkDataset
        from torcheeg.datasets.utils import get_adjacency

        dataset = TSUBenckmarkDataset(root_path='./TSUBenchmark',
                                      online_transform=transforms.Compose([
                                          ToG(get_adjacency(TSUBenckmarkDataset.channels, TSUBenckmarkDataset.adjacency))
                                      ]),
                                      label_transform=transforms.Select(['trial_id']))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)

    Args:
        root_path (str): Downloaded data files in matlab (unzipped TSUBenchmark.zip) formats (default: :obj:`'./TSUBenchmark'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`250`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 64 channels are EEG signals. (default: :obj:`64`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)

    '''

    channels = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
        'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5',
        'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3',
        'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4',
        'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]

    grid = [
        ['-', '-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-', '-'],
        ['-', '-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-', '-'],
        ['-', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', '-'],
        ['-', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', '-'],
        ['-', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', '-'],
        [
            'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
            'M2'
        ], ['-', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', '-'],
        ['-', '-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-', '-'],
        ['-', '-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-', '-'],
        ['-', '-', '-', '-', 'CB1', '-', 'CB2', '-', '-', '-', '-']
    ]

    adjacency = {
        'FP1': ['FPZ', 'AF3'],
        'FPZ': ['FP1', 'FP2', 'AF3', 'AF4'],
        'FP2': ['FPZ', 'AF4'],
        'AF3': ['F1', 'FP1', 'F3', 'FZ', 'FPZ'],
        'AF4': ['F2', 'FP2', 'FZ', 'F4', 'FPZ'],
        'F7': ['F5', 'FT7', 'FC5'],
        'F5': ['F7', 'F3', 'FC5', 'FT7', 'FC3'],
        'F3': ['F5', 'F1', 'FC3', 'FC5', 'FC1', 'AF3'],
        'F1': ['F3', 'FZ', 'FC1', 'AF3', 'FC3', 'FCZ'],
        'FZ': ['F1', 'F2', 'FCZ', 'FC1', 'FC2', 'AF3', 'AF4'],
        'F2': ['FZ', 'F4', 'FC2', 'AF4', 'FCZ', 'FC4'],
        'F4': ['F2', 'F6', 'FC4', 'FC2', 'FC6', 'AF4'],
        'F6': ['F4', 'F8', 'FC6', 'FC4', 'FT8'],
        'F8': ['F6', 'FT8', 'FC6'],
        'FT7': ['FC5', 'T7', 'F7', 'C5', 'F5'],
        'FC5': ['FT7', 'FC3', 'C5', 'F5', 'T7', 'C3', 'F7', 'F3'],
        'FC3': ['FC5', 'FC1', 'C3', 'F3', 'C5', 'C1', 'F5', 'F1'],
        'FC1': ['FC3', 'FCZ', 'C1', 'F1', 'C3', 'CZ', 'F3', 'FZ'],
        'FCZ': ['FC1', 'FC2', 'CZ', 'FZ', 'C1', 'C2', 'F1', 'F2'],
        'FC2': ['FCZ', 'FC4', 'C2', 'F2', 'CZ', 'C4', 'FZ', 'F4'],
        'FC4': ['FC2', 'FC6', 'C4', 'F4', 'C2', 'C6', 'F2', 'F6'],
        'FC6': ['FC4', 'FT8', 'C6', 'F6', 'C4', 'T8', 'F4', 'F8'],
        'FT8': ['FC6', 'T8', 'F8', 'C6', 'F6'],
        'T7': ['C5', 'TP7', 'FT7', 'M1', 'CP5', 'FC5'],
        'C5': ['T7', 'C3', 'CP5', 'FC5', 'TP7', 'CP3', 'FT7', 'FC3'],
        'C3': ['C5', 'C1', 'CP3', 'FC3', 'CP5', 'CP1', 'FC5', 'FC1'],
        'C1': ['C3', 'CZ', 'CP1', 'FC1', 'CP3', 'CPZ', 'FC3', 'FCZ'],
        'CZ': ['C1', 'C2', 'CPZ', 'FCZ', 'CP1', 'CP2', 'FC1', 'FC2'],
        'C2': ['CZ', 'C4', 'CP2', 'FC2', 'CPZ', 'CP4', 'FCZ', 'FC4'],
        'C4': ['C2', 'C6', 'CP4', 'FC4', 'CP2', 'CP6', 'FC2', 'FC6'],
        'C6': ['C4', 'T8', 'CP6', 'FC6', 'CP4', 'TP8', 'FC4', 'FT8'],
        'T8': ['C6', 'TP8', 'FT8', 'CP6', 'M2', 'FC6'],
        'M1': ['TP7', 'P7', 'T7'],
        'TP7': ['M1', 'CP5', 'P7', 'T7', 'P5', 'C5'],
        'CP5': ['TP7', 'CP3', 'P5', 'C5', 'P7', 'P3', 'T7', 'C3'],
        'CP3': ['CP5', 'CP1', 'P3', 'C3', 'P5', 'P1', 'C5', 'C1'],
        'CP1': ['CP3', 'CPZ', 'P1', 'C1', 'P3', 'PZ', 'C3', 'CZ'],
        'CPZ': ['CP1', 'CP2', 'PZ', 'CZ', 'P1', 'P2', 'C1', 'C2'],
        'CP2': ['CPZ', 'CP4', 'P2', 'C2', 'PZ', 'P4', 'CZ', 'C4'],
        'CP4': ['CP2', 'CP6', 'P4', 'C4', 'P2', 'P6', 'C2', 'C6'],
        'CP6': ['CP4', 'TP8', 'P6', 'C6', 'P4', 'P8', 'C4', 'T8'],
        'TP8': ['CP6', 'M2', 'P8', 'T8', 'P6', 'C6'],
        'M2': ['TP8', 'P8', 'T8'],
        'P7': ['P5', 'TP7', 'PO7', 'M1', 'CP5'],
        'P5': ['P7', 'P3', 'PO7', 'CP5', 'PO5', 'TP7', 'CP3'],
        'P3': ['P5', 'P1', 'PO5', 'CP3', 'PO7', 'PO3', 'CP5', 'CP1'],
        'P1': ['P3', 'PZ', 'PO3', 'CP1', 'PO5', 'POZ', 'CP3', 'CPZ'],
        'PZ': ['P1', 'P2', 'POZ', 'CPZ', 'PO3', 'PO4', 'CP1', 'CP2'],
        'P2': ['PZ', 'P4', 'PO4', 'CP2', 'POZ', 'PO6', 'CPZ', 'CP4'],
        'P4': ['P2', 'P6', 'PO6', 'CP4', 'PO4', 'PO8', 'CP2', 'CP6'],
        'P6': ['P4', 'P8', 'PO8', 'CP6', 'PO6', 'CP4', 'TP8'],
        'P8': ['P6', 'TP8', 'PO8', 'CP6', 'M2'],
        'PO7': ['PO5', 'P5', 'P7', 'P3'],
        'PO5': ['PO7', 'PO3', 'P3', 'O1', 'P5', 'P1'],
        'PO3': ['PO5', 'POZ', 'O1', 'P1', 'OZ', 'P3', 'PZ'],
        'POZ': ['PO3', 'PO4', 'OZ', 'PZ', 'O1', 'O2', 'P1', 'P2'],
        'PO4': ['POZ', 'PO6', 'O2', 'P2', 'OZ', 'PZ', 'P4'],
        'PO6': ['PO4', 'PO8', 'P4', 'O2', 'P2', 'P6'],
        'PO8': ['PO6', 'P6', 'P4', 'P8'],
        'O1': ['OZ', 'CB1', 'PO3', 'PO5', 'POZ'],
        'OZ': ['O1', 'O2', 'POZ', 'CB1', 'CB2', 'PO3', 'PO4'],
        'O2': ['OZ', 'CB2', 'PO4', 'POZ', 'PO6'],
        'CB1': ['O1', 'OZ'],
        'CB2': ['O2', 'OZ']
    }

    positions = get_positions(channels)

    def __init__(self,
                 root_path: str = './TSUBenchmark',
                 chunk_size: int = 250,
                 overlap: int = 0,
                 num_channel: int = 64,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[None, Callable] = None,
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
    def read_record(record: str,
                    root_path: str = './TSUBenchmark', **kwargs) -> Dict:

        samples = scio.loadmat(os.path.join(root_path,
                                            record))['data'].transpose(
                                                2, 3, 0, 1)

        freq_phase = scio.loadmat(os.path.join(root_path, 'Freq_Phase.mat'))
        freqs = freq_phase['freqs'][0]
        phases = freq_phase['phases'][0]

        return {
            'samples': samples,
            'freqs': freqs,
            'phases': phases
        }

    @staticmethod
    def fake_record(record: str, **kwargs) -> Dict:
        num_trials = 20
        num_blocks = 6
        num_channels = 64
        num_samples = 1500

        samples = np.random.rand(
            num_trials, num_blocks, num_channels, num_samples)
        freqs = np.random.rand(num_trials, num_blocks)
        phases = np.random.rand(num_trials, num_blocks)

        return {
            'record': 'S34.mat',
            'samples': samples,
            'freqs': freqs,
            'phases': phases
        }

    @staticmethod
    def process_record(record: str,
                       samples: np.ndarray,
                       freqs: np.ndarray,
                       phases: np.ndarray,
                       chunk_size: int = 250,
                       overlap: int = 0,
                       num_channel: int = 64,
                       offline_transform: Union[None, Callable] = None,
                       before_trial: Union[None, Callable] = None,
                       **kwargs):

        subject = int(re.findall(r'S(\d*).mat', record)[0])  # subject (35)

        write_pointer = 0

        for trial_id in range(samples.shape[0]):
            trial_meta_info = {
                'subject_id': subject,
                'trial_id': trial_id,
                'phases': phases[trial_id],
                'freqs': freqs[trial_id]
            }
            trial_samples = samples[trial_id]

            for block_id in range(trial_samples.shape[0]):
                block_meta_info = {'block_id': block_id}
                block_meta_info.update(trial_meta_info)
                block_samples = trial_samples[block_id]
                if before_trial:
                    block_samples = before_trial(block_samples)

                start_at = 0
                if chunk_size <= 0:
                    dynamic_chunk_size = block_samples.shape[1] - start_at
                else:
                    dynamic_chunk_size = chunk_size

                # chunk with chunk size
                end_at = dynamic_chunk_size
                # calculate moving step
                step = dynamic_chunk_size - overlap

                while end_at <= block_samples.shape[1]:
                    clip_sample = block_samples[:num_channel, start_at:end_at]

                    t_eeg = clip_sample
                    if not offline_transform is None:
                        t_eeg = offline_transform(eeg=clip_sample)['eeg']

                    clip_id = f'{record}_{write_pointer}'
                    write_pointer += 1

                    # record meta info for each signal
                    record_info = {
                        'start_at': start_at,
                        'end_at': end_at,
                        'clip_id': clip_id
                    }
                    record_info.update(block_meta_info)
                    yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

                    start_at = start_at + step
                    end_at = start_at + dynamic_chunk_size

    def set_records(self, **kwargs):
        root_path = kwargs.pop('root_path', './TSUBenchmark')  # str

        if not (os.path.exists(root_path)):
            raise FileNotFoundError(
                f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
            )

        file_list = os.listdir(root_path)
        skip_set = [
            'Readme.txt', 'Sub_info.txt', '64-channels.loc', '64-channels.loc',
            'Freq_Phase.mat'
        ]
        file_list = [f for f in file_list if f not in skip_set]
        return file_list

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
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
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
