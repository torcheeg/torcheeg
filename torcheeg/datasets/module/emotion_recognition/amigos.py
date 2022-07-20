from typing import Callable, Dict, List, Tuple, Union

from ...constants.emotion_recognition.amigos import (
    AMIGOS_ADJACENCY_MATRIX, AMIGOS_CHANNEL_LOCATION_DICT)
from ...functional.emotion_recognition.amigos import amigos_constructor
from ..base_dataset import BaseDataset


class AMIGOSDataset(BaseDataset):
    r'''
    A dataset for Multimodal research of affect, personality traits and mood on Individuals and GrOupS (AMIGOS). This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Miranda-Correa et al.
    - Year: 2018
    - Download URL: http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html
    - Reference: Miranda-Correa J A, Abadi M K, Sebe N, et al. Amigos: A dataset for affect, personality and mood research on individuals and groups[J]. IEEE Transactions on Affective Computing, 2018, 12(2): 479-493.
    - Stimulus: 16 short affective video extracts and 4 long affective video extracts from movies.
    - Signals: Electroencephalogram (14 channels at 128Hz), electrocardiogram (2 channels at 60Hz) and galvanic skin response (1 channel at 60Hz) of 40 subjects. For the first 16 trials, 40 subjects watched a set of short affective video extracts. For the last 4 trials, 37 of the participants of the previous experiment watched a set of long affective video extracts.
    - Rating: arousal (1-9), valence (1-9), dominance (1-9), liking (1-9), familiarity (1-9), neutral (0, 1), disgust (0, 1),happiness (0, 1), surprise (0, 1), anger (0, 1), fear (0, 1), and sadness (0, 1).

    In order to use this dataset, the download folder :obj:`data_preprocessed` is required, containing the following files:
    
    - Data_Preprocessed_P01.mat
    - Data_Preprocessed_P02.mat
    - Data_Preprocessed_P03.mat
    - ...
    - Data_Preprocessed_P40.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        dataset = AMIGOSDataset(io_path=f'./amigos',
                                root_path='./data_preprocessed',
                                offline_transform=transforms.Compose([
                                    transforms.BandDifferentialEntropy(),
                                    transforms.ToGrid(AMIGOS_CHANNEL_LOCATION_DICT)
                                ]),
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Compose([
                                    transforms.Select('valence'),
                                    transforms.Binary(5.0),
                                ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[128, 9, 9]),
        # coresponding baseline signal (torch.Tensor[128, 9, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = AMIGOSDataset(io_path=f'./amigos',
                              root_path='./data_preprocessed',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[14, 128]),
        # coresponding baseline signal (torch.Tensor[14, 128]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python
    
        dataset = AMIGOSDataset(io_path=f'./amigos',
                              root_path='./data_preprocessed',
                              online_transform=transforms.Compose([
                                  transforms.ToG(AMIGOS_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
    
    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = AMIGOSDataset(io_path=f'./amigos',
                                    root_path='./data_preprocessed',
                                    online_transform=transforms.Compose([
                                        transforms.ToG(AMIGOS_ADJACENCY_MATRIX)
                                    ]),
                                    label_transform=transforms.Compose([
                                        transforms.Select('valence'),
                                        transforms.Binary(5.0),
                                    ]),
                                    num_worker=4)
            print(dataset[0])
            # EEG signal (torch_geometric.data.Data),
            # coresponding baseline signal (torch_geometric.data.Data),
            # label (int)

    Args:
        root_path (str): Downloaded data files in matlab (unzipped data_preprocessed.zip) formats (default: :obj:`'./data_preprocessed'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        channel_num (int): Number of channels used, of which the first 14 channels are EEG signals. (default: :obj:`14`)
        trial_num (int): Number of trials used, of which the first 16 trials are conducted with short videos and the last 4 trials are conducted with long videos. If set to -1, all trials are used. (default: :obj:`16`)
        skipped_subjects (int): The participant ID to be removed because there are some invalid data in the preprocessed version. (default: :obj:`[9, 12, 21, 22, 23, 24, 33]`)
        baseline_num (int): Number of baseline signal chunks used. (default: :obj:`5`)
        baseline_chunk_size (int): Number of data points included in each baseline signal chunk. The baseline signal in the AMIGOS dataset has a total of 640 data points. (default: :obj:`128`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/amigos`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        cache_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`64 * 1024 * 1024 * 1024`)

    '''
    channel_location_dict = AMIGOS_CHANNEL_LOCATION_DICT
    adjacency_matrix = AMIGOS_ADJACENCY_MATRIX

    def __init__(self,
                 root_path: str = './data_preprocessed',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 channel_num: int = 14,
                 trial_num: int = 16,
                 skipped_subjects: List[int] = [9, 12, 21, 22, 23, 24, 33],
                 baseline_num: int = 5,
                 baseline_chunk_size: int = 128,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/amigos',
                 num_worker: int = 0,
                 verbose: bool = True,
                 cache_size: int = 64 * 1024 * 1024 * 1024):
        amigos_constructor(root_path=root_path,
                           chunk_size=chunk_size,
                           overlap=overlap,
                           channel_num=channel_num,
                           trial_num=trial_num,
                           skipped_subjects=skipped_subjects,
                           baseline_num=baseline_num,
                           baseline_chunk_size=baseline_chunk_size,
                           transform=offline_transform,
                           io_path=io_path,
                           num_worker=num_worker,
                           verbose=verbose,
                           cache_size=cache_size)
        super().__init__(io_path)

        self.root_path = root_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.channel_num = channel_num
        self.trial_num = trial_num
        self.skipped_subjects = skipped_subjects
        self.baseline_num = baseline_num
        self.baseline_chunk_size = baseline_chunk_size
        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.verbose = verbose
        self.cache_size = cache_size

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
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
                'overlap': self.overlap,
                'channel_num': self.channel_num,
                'trial_num': self.trial_num,
                'skipped_subjects': self.skipped_subjects,
                'baseline_num': self.baseline_num,
                'baseline_chunk_size': self.baseline_chunk_size,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'verbose': self.verbose,
                'cache_size': self.cache_size
            })
