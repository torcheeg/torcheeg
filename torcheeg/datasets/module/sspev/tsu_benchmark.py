from typing import Callable, Dict, Tuple, Union

from ...constants.ssvep.tsu_benchmark import (TSUBENCHMARK_ADJACENCY_MATRIX,
                                                   TSUBENCHMARK_CHANNEL_LOCATION_DICT)
from ...functional.ssvep.tsu_benchmark import tsu_benchmark_constructor
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
                                  transforms.pyg.ToG(TSUBenckmark_ADJACENCY_MATRIX)
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
                                  transforms.pyg.ToG(TSUBenckmark_ADJACENCY_MATRIX)
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
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/tsu_benchmark`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
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
                 io_path: str = './io/tsu_benchmark',
                 num_worker: int = 0,
                 verbose: bool = True,
                 cache_size: int = 64 * 1024 * 1024 * 1024):
        tsu_benchmark_constructor(root_path=root_path,
                         chunk_size=chunk_size,
                         overlap=overlap,
                         num_channel=num_channel,
                         transform=offline_transform,
                         io_path=io_path,
                         num_worker=num_worker,
                         verbose=verbose,
                         cache_size=cache_size)
        super().__init__(io_path)

        self.root_path = root_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.num_channel = num_channel
        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.num_worker = num_worker
        self.verbose = verbose
        self.cache_size = cache_size

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.info.iloc[index].to_dict()

        eeg_index = str(info['clip_id'])
        eeg = self.eeg_io.read_eeg(eeg_index)

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
                'cache_size': self.cache_size
            })
