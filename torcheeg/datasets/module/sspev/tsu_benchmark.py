import os
import re
from typing import Any, Callable, Dict, Tuple, Union

import scipy.io as scio

from torcheeg.io import EEGSignalIO, MetaInfoIO

from ...constants.ssvep.tsu_benchmark import (TSUBENCHMARK_ADJACENCY_MATRIX,
                                              TSUBENCHMARK_CHANNEL_LOCATION_DICT
                                              )
from ..base_dataset import BaseDataset


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
    
    - Readme.txt
    - Sub_info.txt
    - 64-channels.loc
    - Freq_Phase.mat
    - S1.mat
    - ...
    - S35.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        dataset = TSUBenckmarkDataset(io_path=f'./tsu_benchmark',
                              root_path='./TSUBenchmark',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(TSUBenckmark_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Select(['trial_id']))
        print(dataset[0])
        # EEG signal (torch.Tensor[250, 10, 11]),
        # coresponding baseline signal (torch.Tensor[250, 10, 11]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = TSUBenckmarkDataset(io_path=f'./tsu_benchmark',
                              root_path='./TSUBenchmark',
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
    
        dataset = TSUBenckmarkDataset(io_path=f'./tsu_benchmark',
                              root_path='./TSUBenchmark',
                              online_transform=transforms.Compose([
                                  ToG(TSUBenckmark_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Select(['trial_id']))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
        
    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = TSUBenckmarkDataset(io_path=f'./tsu_benchmark',
                              root_path='./TSUBenchmark',
                              online_transform=transforms.Compose([
                                  ToG(TSUBenckmark_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Select(['freq']),
                              num_worker=4)
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
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/tsu_benchmark`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    
    '''
    channel_location_dict = TSUBENCHMARK_CHANNEL_LOCATION_DICT
    adjacency_matrix = TSUBENCHMARK_ADJACENCY_MATRIX

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
                 io_path: str = './io/tsu_benchmark',
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
                   root_path: str = './TSUBenchmark',
                   chunk_size: int = 250,
                   overlap: int = 0,
                   num_channel: int = 64,
                   offline_transform: Union[None, Callable] = None,
                   before_trial: Union[None, Callable] = None,
                   after_trial: Union[None, Callable] = None,
                   **kwargs):
        file_name = file

        subject = int(re.findall(r'S(\d*).mat', file_name)[0])  # subject (35)
        freq_phase = scio.loadmat(os.path.join(root_path, 'Freq_Phase.mat'))
        freqs = freq_phase['freqs'][0]
        phases = freq_phase['phases'][0]

        samples = scio.loadmat(os.path.join(root_path,
                                            file_name))['data'].transpose(
                                                2, 3, 0, 1)
        # 40, 6, 64, 1500
        # Target number: 40
        # Block number: 6
        # Electrode number: 64
        # Time points: 1500

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
                    chunk_size = block_samples.shape[1] - start_at

                # chunk with chunk size
                end_at = chunk_size
                # calculate moving step
                step = chunk_size - overlap

                block_queue = []
                while end_at <= block_samples.shape[1]:
                    clip_sample = block_samples[:num_channel, start_at:end_at]

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
                    record_info.update(block_meta_info)
                    if after_trial:
                        block_queue.append({
                            'eeg': t_eeg,
                            'key': clip_id,
                            'info': record_info
                        })
                    else:
                        yield {
                            'eeg': t_eeg,
                            'key': clip_id,
                            'info': record_info
                        }

                    start_at = start_at + step
                    end_at = start_at + chunk_size

                if len(block_queue) and after_trial:
                    block_queue = after_trial(block_queue)
                    for obj in block_queue:
                        assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
                        yield obj

    @staticmethod
    def _set_files(**kwargs):
        root_path = kwargs.pop('root_path', './TSUBenchmark')  # str

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
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
