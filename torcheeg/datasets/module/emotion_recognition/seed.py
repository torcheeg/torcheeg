from typing import Callable, Union, Tuple

from ..base_dataset import BaseDataset
from ...functional.emotion_recognition.seed import seed_constructor


class SEEDDataset(BaseDataset):
    r'''
    The SJTU Emotion EEG Dataset (SEED), is a collection of EEG datasets provided by the BCMI laboratory, which is led by Prof. Bao-Liang Lu. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Zheng et al.
    - Year: 2015
    - Download URL: https://bcmi.sjtu.edu.cn/home/seed/index.html
    - Reference: Zheng W L, Lu B L. Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks[J]. IEEE Transactions on Autonomous Mental Development, 2015, 7(3): 162-175.
    - Stimulus: 15 four-minute long film clips from six Chinese movies.
    - Signals: Electroencephalogram (62 channels at 200Hz) of 15 subjects, and eye movement data of 12 subjects. Each subject conducts the experiment three times, with an interval of about one week, totally 15 people x 3 times = 45
    - Rating: positive (-1), negative (0), and neutral (1).

    An example dataset for CNN-based methods:

    .. code-block:: python

        dataset = SEEDDataset(io_path=f'./seed',
                              root_path='./Preprocessed_EEG',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select(['emotion']),
                                  transforms.Lambda(x: x + 1)
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[200, 9, 9]),
        # coresponding baseline signal (torch.Tensor[200, 9, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = SEEDDataset(io_path=f'./seed',
                              root_path='./Preprocessed_EEG',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Lambda(lambda x: x.unsqueeze(0))
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['emotion']),
                                  transforms.Lambda(x: x + 1)
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[62, 200]),
        # coresponding baseline signal (torch.Tensor[62, 200]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python
    
        dataset = SEEDDataset(io_path=f'./seed',
                              root_path='./Preprocessed_EEG',
                              online_transform=transforms.Compose([
                                  transforms.ToG(SEED_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['emotion']),
                                  transforms.Lambda(x: x + 1)
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
        
    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup.

    Args:
        root_path (str): Downloaded data files in matlab (unzipped Preprocessed_EEG.zip) formats (default: :obj:`'./Preprocessed_EEG'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`200`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        channel_num (int): Number of channels used, of which the first 62 channels are EEG signals. (default: :obj:`62`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/seed`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`1`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    
    '''
    def __init__(self,
                 root_path: str = './Preprocessed_EEG',
                 chunk_size: int = 200,
                 overlap: int = 0,
                 channel_num: int = 62,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/seed',
                 num_worker: int = 1,
                 verbose: bool = True):
        seed_constructor(root_path=root_path,
                         chunk_size=chunk_size,
                         overlap=overlap,
                         channel_num=channel_num,
                         transform=offline_transform,
                         io_path=io_path,
                         num_worker=num_worker,
                         verbose=verbose)
        super().__init__(io_path)

        self.root_path = root_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.channel_num = channel_num
        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.io_path = io_path
        self.verbose = verbose

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.info.iloc[index].to_dict()

        eeg_index = str(info['clip_id'])
        eeg = self.eeg_io.read_eeg(eeg_index)

        if self.online_transform:
            eeg = self.online_transform(eeg)

        if self.label_transform:
            info = self.label_transform(info)

        if isinstance(info, list):
            return (eeg, *info)
        if isinstance(info, dict):
            return (eeg, *info.values())
        return eeg, info