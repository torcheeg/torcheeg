from typing import Callable, Dict, Tuple, Union

from ...constants.emotion_recognition.mped import (MPED_ADJACENCY_MATRIX,
                                                   MPED_CHANNEL_LOCATION_DICT)
from ...functional.emotion_recognition.mped_feature import \
    mped_feature_constructor
from ..base_dataset import BaseDataset


class MPEDFeatureDataset(BaseDataset):
    r'''
    The Multi-Modal Physiological Emotion Database for Discrete Emotion (MPED), a multi-modal physiological emotion database, which collects four modal physiological signals, i.e., electroencephalogram (EEG), galvanic skin response, respiration, and electrocardiogram (ECG). Since the MPED dataset provides features based on matlab, this class implements the processing of these feature files to initialize the dataset. The relevant information of the dataset is as follows:

    - Author: Song et al.
    - Year: 2019
    - Download URL: https://github.com/tengfei000/mped/
    - Reference: Song T, Zheng W, Lu C, et al. MPED: A multi-modal physiological emotion database for discrete emotion recognition[J]. IEEE Access, 2019, 7: 12177-12191.
    - Stimulus: 28 videos from an emotion elicitation material database.
    - Signals: Electroencephalogram (62 channels at 1000Hz), respiration, galvanic skin reponse, and electrocardiogram of 23 subjects. Each subject conducts 30 experiments, totally 23 people x 30 experiments = 690
    - Rating: resting status (0), neutral (1), joy (2), funny (3), angry (4), fear (5), disgust (6), sadness (7).
    - Feature: HHS, Hjorth, PSD, STFT, and HOC

    In order to use this dataset, the download folder :obj:`EEG_feature` is required, containing the following files:
    
    - HHS
    - Hjorth
    - HOC
    - ...
    - STFT

    An example dataset for CNN-based methods:

    .. code-block:: python

        dataset = MPEDFeatureDataset(io_path=f'./mped',
                              root_path='./EEG_feature',
                              features=['PSD'],
                              offline_transform=transforms.ToGrid(MPED_CHANNEL_LOCATION_DICT),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select(['emotion'])
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[5, 9, 9]),
        # coresponding baseline signal (torch.Tensor[5, 9, 9]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python
    
        dataset = MPEDFeatureDataset(io_path=f'./mped',
                              root_path='./Preprocessed_EEG',
                              features=['PSD'],
                              online_transform=ToG(MPED_ADJACENCY_MATRIX),
                              label_transform=transforms.Compose([
                                  transforms.Select(['emotion'])
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
        
    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = MPEDFeatureDataset(io_path=f'./mped',
                              root_path='./EEG_feature',
                              feature=['PSD'],
                              offline_transform=transforms.ToGrid(MPED_CHANNEL_LOCATION_DICT),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select(['emotion'])
                              ]),
                              num_worker=4)
            print(dataset[0])
            # EEG signal (torch_geometric.data.Data),
            # coresponding baseline signal (torch_geometric.data.Data),
            # label (int)

    Args:
        root_path (str): Downloaded data files in matlab (unzipped EEG_feature.zip) formats (default: :obj:`'./EEG_feature'`)
        feature (list): A list of selected feature names. The selected features corresponding to each electrode will be concatenated together. Feature names supported by the MPED dataset include HHS, Hjorth, PSD, STFT, and HOC (default: :obj:`['PSD']`)
        num_channel (int): Number of channels used, of which the first 62 channels are EEG signals. (default: :obj:`62`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 3D EEG signal with shape (number of windows, number of electrodes, number of features), whose ideal output shape is also (number of windows, number of electrodes, number of features).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/mped_feature`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)    
    '''
    channel_location_dict = MPED_CHANNEL_LOCATION_DICT
    adjacency_matrix = MPED_ADJACENCY_MATRIX

    def __init__(self,
                 root_path: str = './EEG_feature',
                 feature: list = ['PSD'],
                 num_channel: int = 62,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 io_path: str = './io/mped_feature',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):
        mped_feature_constructor(root_path=root_path,
                                 feature=feature,
                                 num_channel=num_channel,
                                 before_trial=before_trial,
                                 transform=offline_transform,
                                 after_trial=after_trial,
                                 io_path=io_path,
                                 io_size=io_size,
                                 io_mode=io_mode,
                                 num_worker=num_worker,
                                 verbose=verbose)
        super().__init__(io_path=io_path,
                         io_size=io_size,
                         io_mode=io_mode,
                         in_memory=in_memory)

        self.root_path = root_path
        self.feature = feature
        self.num_channel = num_channel
        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.before_trial = before_trial
        self.after_trial = after_trial
        self.num_worker = num_worker
        self.verbose = verbose
        self.io_size = io_size

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
                'feature': self.feature,
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
