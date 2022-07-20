from typing import Callable, Dict, Tuple, Union

from ...constants.emotion_recognition.mahnob import (
    MAHNOB_ADJACENCY_MATRIX, MAHNOB_CHANNEL_LOCATION_DICT)
from ...functional.emotion_recognition.mahnob import mahnob_constructor
from ..base_dataset import BaseDataset


class MAHNOBDataset(BaseDataset):
    r'''
    MAHNOB-HCI is a multimodal database recorded in response to affective stimuli with the goal of emotion recognition and implicit tagging research. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:
    
    - Author: Soleymani et al.
    - Year: 2011
    - Download URL: https://mahnob-db.eu/hci-tagging/
    - Reference: Soleymani M, Lichtenauer J, Pun T, et al. A multimodal database for affect recognition and implicit tagging[J]. IEEE transactions on affective computing, 2011, 3(1): 42-55.
    - Stimulus: 20 videos from famous movies. Each video clip lasts 34-117 seconds (may not be an integer), in addition to 30 seconds before the beginning of the affective stimuli experience and another 30 seconds after the end.
    - Signals: Electroencephalogram (32 channels at 512Hz), peripheral physiological signals (ECG, GSR, Temp, Resp at 256 Hz), and eye movement signals (at 60Hz) of 30-5=25 subjects (3 subjects with missing data records and 2 subjects with incomplete data records).
    - Rating: Arousal, valence, control and predictability (all ona scale from 1 to 9).
    
    In order to use this dataset, the download folder :obj:`Sessions` (Physiological files of emotion elicitation) is required, containing the following files:
    
    - 1

      + Part_1_N_Trial1_emotion.bdf
      + session.xml

    - ...
    - 3810
    
      + Part_30_S_Trial20_emotion.bdf
      + session.xml

    An example dataset for CNN-based methods:

    .. code-block:: python
    
        dataset = MAHNOBDataset(io_path=f'./mahnob',
                              root_path='./Sessions',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(MAHNOB_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('feltVlnc'),
                                  transforms.Binary(5.0),
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[128, 9, 9]),
        # coresponding baseline signal (torch.Tensor[128, 9, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = MAHNOBDataset(io_path=f'./mahnob',
                              root_path='./Sessions',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['feltVlnc', 'feltArsl']),
                                  transforms.Binary(5.0),
                                  transforms.BinariesToCategory()
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 32, 128]),
        # coresponding baseline signal (torch.Tensor[1, 32, 128]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python
    
        dataset = MAHNOBDataset(io_path=f'./mahnob',
                              root_path='./Sessions',
                              online_transform=transforms.Compose([
                                  transforms.ToG(MAHNOB_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('feltArsl'),
                                  transforms.Binary(5.0)
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = MAHNOBDataset(io_path=f'./mahnob',
                              root_path='./Sessions',
                              online_transform=transforms.Compose([
                                  transforms.ToG(MAHNOB_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('feltArsl'),
                                  transforms.Binary(5.0)
                              ]),
                              num_worker=4)
            print(dataset[0])
            # EEG signal (torch_geometric.data.Data),
            # coresponding baseline signal (torch_geometric.data.Data),
            # label (int)

    Args:
        root_path (str): Downloaded data files in bdf and xml (unzipped Sessions.zip) formats (default: :obj:`'./Sessions'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`128`)
        sampling_rate (int): The number of data points taken over a second. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        channel_num (int): Number of channels used, of which the first 32 channels are EEG signals. (default: :obj:`32`)
        baseline_num (int): Number of baseline signal chunks used. (default: :obj:`30`)
        baseline_chunk_size (int): Number of data points included in each baseline signal chunk. The baseline signal in the MAHNOB dataset has a total of 512 (downsampled to sampling_rate) * 30 data points. (default: :obj:`128`)
        trial_sample_num (int): Number of samples picked from each trial. If set to -1, all samples in trials are used. (default: :obj:`30`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/mahnob`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        cache_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`64 * 1024 * 1024 * 1024`)
    
    '''
    channel_location_dict = MAHNOB_CHANNEL_LOCATION_DICT
    adjacency_matrix = MAHNOB_ADJACENCY_MATRIX

    def __init__(self,
                 root_path: str = './Sessions',
                 chunk_size: int = 128,
                 sampling_rate: int = 128,
                 overlap: int = 0,
                 channel_num: int = 32,
                 baseline_num: int = 30,
                 baseline_chunk_size: int = 128,
                 trial_sample_num: int = 30,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/mahnob',
                 num_worker: int = 0,
                 verbose: bool = True,
                 cache_size: int = 64 * 1024 * 1024 * 1024):
        mahnob_constructor(root_path=root_path,
                           chunk_size=chunk_size,
                           sampling_rate=sampling_rate,
                           overlap=overlap,
                           channel_num=channel_num,
                           baseline_num=baseline_num,
                           baseline_chunk_size=baseline_chunk_size,
                           trial_sample_num=trial_sample_num,
                           transform=offline_transform,
                           io_path=io_path,
                           num_worker=num_worker,
                           verbose=verbose,
                           cache_size=cache_size)
        super().__init__(io_path)

        self.root_path = root_path
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
        self.overlap = overlap
        self.channel_num = channel_num
        self.baseline_num = baseline_num
        self.baseline_chunk_size = baseline_chunk_size
        self.trial_sample_num = trial_sample_num
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
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'sampling_rate': self.sampling_rate,
                'overlap': self.overlap,
                'channel_num': self.channel_num,
                'baseline_num': self.baseline_num,
                'baseline_chunk_size': self.baseline_chunk_size,
                'trial_sample_num': self.trial_sample_num,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'cache_size': self.cache_size
            })
