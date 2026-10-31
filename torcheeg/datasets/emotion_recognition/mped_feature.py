import os
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import scipy.io as scio

from ...utils import get_random_dir_path
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

    .. code-block:: text

        EEG_feature/
        ├── HHS/
        ├── Hjorth/
        ├── HOC/
        ├── ...
        └── STFT/

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import MPEDFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_locations

        dataset = MPEDFeatureDataset(root_path='./EEG_feature',
                                     features=['PSD'],
                                     offline_transform=transforms.ToGrid       (get_locations(MPEDFeatureDataset.channels, MPEDFeatureDataset.grid)),
                                     online_transform=transforms.ToTensor(),
                                     label_transform=transforms.Compose([
                                         transforms.Select('emotion')
                                     ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[5, 9, 9]),
        # coresponding baseline signal (torch.Tensor[5, 9, 9]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import MPEDFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets.utils import get_adjacency
        from torcheeg.transforms.pyg import ToG

        dataset = MPEDFeatureDataset(root_path='./Preprocessed_EEG',
                                     features=['PSD'],
                                     online_transform=ToG(get_adjacency(MPEDFeatureDataset.channels, MPEDFeatureDataset.adjacency)),
                                     label_transform=transforms.Compose([
                                         transforms.Select('emotion')
                                     ]))
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
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)    
    '''

    channels = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
        'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
        'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
        'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
        'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]

    grid = [
        ['-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-'],
        ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
        ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
        ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
        ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
        ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
        ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
        ['-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-'],
        ['-', '-', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '-', '-']
    ]

    adjacency = {
        'FP1': ['FPZ', 'AF3'],
        'FPZ': ['FP1', 'FP2'],
        'FP2': ['FPZ', 'AF4'],
        'AF3': ['FP1', 'F5', 'F3', 'F1'],
        'AF4': ['F2', 'F4', 'F6', 'FP2'],
        'F7': ['F5', 'FT7'],
        'F5': ['F7', 'AF3', 'F3', 'FC5'],
        'F3': ['AF3', 'F5', 'FC3', 'F1'],
        'F1': ['AF3', 'F3', 'FC1', 'FZ'],
        'FZ': ['F1', 'FCZ', 'F2'],
        'F2': ['FZ', 'FC2', 'F4', 'AF4'],
        'F4': ['F2', 'FC4', 'F6', 'AF4'],
        'F6': ['AF4', 'F4', 'FC6', 'F8'],
        'F8': ['F6', 'FT8'],
        'FT7': ['F7', 'FC5', 'T7'],
        'FC5': ['F5', 'FT7', 'C5', 'FC3'],
        'FC3': ['F3', 'FC5', 'C3', 'FC1'],
        'FC1': ['F1', 'FC3', 'C1', 'FCZ'],
        'FCZ': ['FZ', 'FC1', 'CZ', 'FC2'],
        'FC2': ['F2', 'FCZ', 'C2', 'FC4'],
        'FC4': ['F4', 'FC2', 'C4', 'FC6'],
        'FC6': ['F6', 'FC4', 'C6', 'FT8'],
        'FT8': ['F8', 'FC6', 'T8'],
        'T7': ['FT7', 'C5', 'TP7'],
        'C5': ['FC5', 'T7', 'C3', 'CP5'],
        'C3': ['FC3', 'C5', 'C1', 'CP3'],
        'C1': ['FC1', 'C3', 'CP1', 'CZ'],
        'CZ': ['FCZ', 'C1', 'CPZ', 'C2'],
        'C2': ['FC2', 'CZ', 'CP2', 'C4'],
        'C4': ['FC4', 'C2', 'CP4', 'C6'],
        'C6': ['FC6', 'C4', 'CP6', 'T8'],
        'T8': ['FT8', 'C6', 'TP8'],
        'TP7': ['T7', 'CP5', 'P7'],
        'CP5': ['C5', 'TP7', 'P5', 'CP3'],
        'CP3': ['C3', 'CP5', 'P3', 'CP1'],
        'CP1': ['C1', 'CP3', 'P1', 'CPZ'],
        'CPZ': ['CZ', 'CP1', 'PZ', 'CP2'],
        'CP2': ['C2', 'CPZ', 'P2', 'CP4'],
        'CP4': ['C4', 'CP2', 'P4', 'CP6'],
        'CP6': ['C6', 'CP4', 'P6', 'TP8'],
        'TP8': ['T8', 'CP6', 'P8'],
        'P7': ['TP7', 'P5', 'PO7'],
        'P5': ['CP5', 'P7', 'PO5', 'P3'],
        'P3': ['CP3', 'P5', 'P1'],
        'P1': ['CP1', 'P3', 'PO3', 'PZ'],
        'PZ': ['CPZ', 'P1', 'POZ', 'P2'],
        'P2': ['CP2', 'PZ', 'PO4', 'P4'],
        'P4': ['CP4', 'P2', 'P6'],
        'P6': ['CP6', 'P4', 'P8'],
        'P8': ['TP8', 'P6', 'PO8'],
        'PO7': ['P7', 'PO5', 'CB1'],
        'PO5': ['P5', 'PO7', 'CB1', 'PO3'],
        'PO3': ['P1', 'PO5', 'O1', 'POZ'],
        'POZ': ['PZ', 'PO3', 'OZ', 'PO4'],
        'PO4': ['P2', 'POZ', 'O2', 'PO6'],
        'PO6': ['P6', 'PO4', 'CB2', 'PO8'],
        'PO8': ['P8', 'PO6', 'CB2'],
        'CB1': ['PO7', 'PO5', 'O1'],
        'O1': ['CB1', 'PO3', 'OZ'],
        'OZ': ['POZ', 'O1', 'O2'],
        'O2': ['PO4', 'OZ', 'CB2'],
        'CB2': ['PO6', 'O2', 'PO8']
    }

    def __init__(self,
                 root_path: str = './EEG_feature',
                 feature: list = ['PSD'],
                 num_channel: int = 62,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
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
            'feature': feature,
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
                    root_path: str = './EEG_feature',
                    feature: list = ['PSD'],
                    **kwargs) -> Dict:

        samples_dict = {}
        for cur_feature in feature:
            samples_dict[cur_feature] = scio.loadmat(
                os.path.join(root_path, cur_feature, record),
                verify_compressed_data_integrity=False)  # (1, 30)

        return {
            'samples_dict': samples_dict
        }

    @staticmethod
    def fake_record(record: str, **kwargs) -> Dict:
        num_trials = 30
        num_channels = 62
        num_timepoints = 120

        samples_dict = {
            'PSD': {
                'PSD': np.random.rand(1, num_trials, num_channels, num_timepoints, 5),
                '__header__': None,
                '__version__': '1.0',
                '__globals__': []
            },
            'Hjorth': {
                'alpha': np.random.rand(1, num_trials, num_channels, num_timepoints, 3),
                'beta': np.random.rand(1, num_trials, num_channels, num_timepoints, 3),
                'delta': np.random.rand(1, num_trials, num_channels, num_timepoints, 3),
                'gamma': np.random.rand(1, num_trials, num_channels, num_timepoints, 3),
                'theta': np.random.rand(1, num_trials, num_channels, num_timepoints, 3),
                'whole': np.random.rand(1, num_trials, num_channels, num_timepoints, 3),
                '__header__': None,
                '__version__': '1.0',
                '__globals__': []
            }
        }
        return {
            'samples_dict': samples_dict
        }

    @staticmethod
    def process_record(record: str,
                       samples_dict: dict,
                       num_channel: int = 62,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        labels = [
            0, 2, 1, 4, 3, 5, 6, 7, 1, 2, 3, 6, 7, 4, 5, 0, 1, 5, 6, 2, 2, 1, 7,
            6, 4, 4, 3, 5, 3, 7
        ]  # 0-resting status, 1-neutral, 2-joy, 3-funny, 4-angry, 5-fear, 6-disgust, 7-sadness

        subject = os.path.basename(record).split('.')[0].split('_')[0]

        write_pointer = 0

        for trial_id in range(30):
            trial_samples = []
            for _, samples in samples_dict.items():
                for sub_band in samples.keys():
                    # HHS: hhs_A, hhs_E
                    # Hjorth/HOC: alpha, beta, delta, gamma, theta, whole
                    # STFT: STFT
                    # PSD: PSD
                    if not str.startswith(sub_band, '__'):
                        # not '__header__', '__version__', '__globals__'
                        trial_samples += [
                            samples[sub_band][0][trial_id][:num_channel],
                        ]
                        # PSD: (62, 120, 5)
                        # Hjorth: (62, 120, 3)
                        # HOC: (62, 120, 20)
                        # HHS: (62, 120, 5)
                        # STFT: (62, 120, 5)
            trial_samples = np.concatenate(trial_samples,
                                           axis=-1)  # (62, 120, num_features)
            if before_trial:
                trial_samples = before_trial(trial_samples)

            # record the common meta info
            trial_meta_info = {
                'subject_id': subject,
                'trial_id': trial_id,
                'emotion': labels[trial_id]
            }
            num_clips = trial_samples.shape[1]

            for clip_id in range(num_clips):
                # PSD: (62, 5)
                # Hjorth: (62, 3 * 6)
                # HOC: (62, 20 * 6)
                # STFT: (62, 5)
                # HHS: (62, 5 * 2)
                clip_sample = trial_samples[:, clip_id]

                t_eeg = clip_sample
                if not offline_transform is None:
                    t_eeg = offline_transform(eeg=clip_sample)['eeg']

                clip_id = f'{record}_{write_pointer}'
                write_pointer += 1

                # record meta info for each signal
                record_info = {'clip_id': clip_id}
                record_info.update(trial_meta_info)
                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

    def set_records(self,
                    root_path: str = './EEG_feature',
                    feature: list = ['PSD'],
                    **kwargs):
        if not (os.path.exists(root_path)):
            raise FileNotFoundError(
                f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
            )
        available_features = os.listdir(root_path)

        if not (set(feature).issubset(set(available_features))):
            raise ValueError(
                'The features supported by the MPEDFeature dataset are HHS, Hjorth, PSD, STFT, HOC. The input features are not a subset of the list of supported features.'
            )

        file_list = os.listdir(os.path.join(root_path, available_features[0]))
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
