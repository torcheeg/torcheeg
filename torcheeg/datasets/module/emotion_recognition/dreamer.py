from typing import Callable, Dict, Tuple, Union

from ...constants.emotion_recognition.dreamer import (
    DREAMER_ADJACENCY_MATRIX, DREAMER_CHANNEL_LOCATION_DICT)
from ...functional.emotion_recognition.dreamer import dreamer_constructor
from ..base_dataset import BaseDataset


class DREAMERDataset(BaseDataset):
    r'''
    A multi-modal database consisting of electroencephalogram and electrocardiogram signals recorded during affect elicitation by means of audio-visual stimuli. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Katsigiannis et al.
    - Year: 2017
    - Download URL: https://zenodo.org/record/546113
    - Reference: Katsigiannis S, Ramzan N. DREAMER: A database for emotion recognition through EEG and ECG signals from wireless low-cost off-the-shelf devices[J]. IEEE journal of biomedical and health informatics, 2017, 22(1): 98-107.
    - Stimulus: 18 movie clips. 
    - Signals: Electroencephalogram (14 channels at 128Hz), and electrocardiogram (2 channels at 256Hz) of 23 subjects.
    - Rating: Arousal, valence, like/dislike, dominance, familiarity (all ona scale from 1 to 5).

    In order to use this dataset, the download file :obj:`DREAMER.mat` is required.

    An example dataset for CNN-based methods:

    .. code-block:: python

        dataset = DREAMERDataset(io_path=f'./dreamer',
                              mat_path='./DREAMER.mat',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DREAMER_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(3.0),
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[128, 9, 9]),
        # coresponding baseline signal (torch.Tensor[128, 9, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = DREAMERDataset(io_path=f'./dreamer',
                              mat_path='./DREAMER.mat',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(3.0),
                                  transforms.BinariesToCategory()
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 14, 128]),
        # coresponding baseline signal (torch.Tensor[1, 14, 128]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python
    
        dataset = DREAMERDataset(io_path=f'./dreamer',
                              mat_path='./DREAMER.mat',
                              online_transform=transforms.Compose([
                                  transforms.ToG(DREAMER_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('arousal'),
                                  transforms.Binary(3.0)
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = DREAMERDataset(io_path=f'./dreamer',
                              mat_path='./DREAMER.mat',
                              online_transform=transforms.Compose([
                                  transforms.ToG(DREAMER_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('arousal'),
                                  transforms.Binary(3.0)
                              ]),
                              num_worker=4)
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
    
    Args:
        mat_path (str): Downloaded data files in pickled matlab formats (default: :obj:`'./DREAMER.mat'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        channel_num (int): Number of channels used, of which the first 14 channels are EEG signals. (default: :obj:`14`)
        baseline_num (int): Number of baseline signal chunks used. (default: :obj:`61`)
        baseline_chunk_size (int): Number of data points included in each baseline signal chunk. The baseline signal in the DREAMER dataset has a total of 7808 data points. (default: :obj:`128`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/dreamer`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        cache_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`64 * 1024 * 1024 * 1024`)
    
    '''
    channel_location_dict = DREAMER_CHANNEL_LOCATION_DICT
    adjacency_matrix = DREAMER_ADJACENCY_MATRIX

    def __init__(self,
                 mat_path: str = './DREAMER.mat',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 channel_num: int = 14,
                 baseline_num: int = 61,
                 baseline_chunk_size: int = 128,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/dreamer',
                 num_worker: int = 0,
                 verbose: bool = True,
                 cache_size: int = 64 * 1024 * 1024 * 1024):
        dreamer_constructor(mat_path=mat_path,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            channel_num=channel_num,
                            baseline_num=baseline_num,
                            baseline_chunk_size=baseline_chunk_size,
                            transform=offline_transform,
                            io_path=io_path,
                            num_worker=num_worker,
                            verbose=verbose,
                            cache_size=cache_size)
        super().__init__(io_path)

        self.mat_path = mat_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.channel_num = channel_num
        self.baseline_num = baseline_num
        self.baseline_chunk_size = baseline_chunk_size
        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.num_worker = num_worker
        self.verbose = verbose
        self.cache_size = cache_size

    def __getitem__(self, index: int) -> Tuple:
        info = self.info.iloc[index].to_dict()

        eeg_index = str(info['clip_id'])
        eeg = self.eeg_io.read_eeg(eeg_index)

        baseline_index = str(info['baseline_id'])
        baseline = self.eeg_io.read_eeg(baseline_index)

        signal = eeg
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=eeg, baseline=baseline)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'mat_path': self.mat_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'channel_num': self.channel_num,
                'baseline_num': self.baseline_num,
                'baseline_chunk_size': self.baseline_chunk_size,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'cache_size': self.cache_size
            })
