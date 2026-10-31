import os
import re
from typing import Callable, Dict, Tuple, Union

import numpy as np
import scipy.io as scio

from ...utils import get_random_dir_path
from ..base_dataset import BaseDataset
from ..utils import get_positions


class SEEDFeatureDataset(BaseDataset):
    r'''
    The SJTU Emotion EEG Dataset (SEED), is a collection of EEG datasets provided by the BCMI laboratory, which is led by Prof. Bao-Liang Lu. Since the SEED dataset provides features based on matlab, this class implements the processing of these feature files to initialize the dataset. The relevant information of the dataset is as follows:

    - Author: Zheng et al.
    - Year: 2015
    - Download URL: https://bcmi.sjtu.edu.cn/home/seed/index.html
    - Reference: Zheng W L, Lu B L. Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks[J]. IEEE Transactions on Autonomous Mental Development, 2015, 7(3): 162-175.
    - Stimulus: 15 four-minute long film clips from six Chinese movies.
    - Signals: Electroencephalogram (62 channels at 200Hz) of 15 subjects, and eye movement data of 12 subjects. Each subject conducts the experiment three times, with an interval of about one week, totally 15 people x 3 times = 45
    - Rating: positive (1), negative (-1), and neutral (0).
    - Feature: de_movingAve, de_LDS, psd_movingAve, psd_LDS, dasm_movingAve, dasm_LDS, rasm_movingAve, rasm_LDS, asm_movingAve, asm_LDS, dcau_movingAve, dcau_LDS of 1-second long windows

    In order to use this dataset, the download folder :obj:`Preprocessed_EEG` is required, containing the following files:

    .. code-block:: text

        Preprocessed_EEG/
        ├── label.mat
        ├── readme.txt
        ├── 10_20131130.mat
        ├── ...
        └── 9_20140704.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import SEEDFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets import SEEDDataset
        from torcheeg.datasets.utils import get_locations

        dataset = SEEDFeatureDataset(root_path='./ExtractedFeatures',
                                     feature=['de_movingAve'],
                                     offline_transform=transforms.ToGrid       (get_locations(SEEDDataset.channels, SEEDDataset.grid)),
                                     online_transform=transforms.ToTensor(),
                                     label_transform=transforms.Compose([
                                         transforms.Select('emotion'),
                                         transforms.Lambda(lambda x: x + 1)
                                     ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[5, 9, 9]),
        # coresponding baseline signal (torch.Tensor[5, 9, 9]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import SEEDFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets import SEEDDataset
        from torcheeg.datasets.utils import get_adjacency
        from torcheeg.transforms.pyg import ToG

        dataset = SEEDFeatureDataset(root_path='./Preprocessed_EEG',
                                     features=['de_movingAve'],
                                     online_transform=ToG(get_adjacency(SEEDDataset.channels, SEEDDataset.adjacency)),
                                     label_transform=transforms.Compose([
                                         transforms.Select('emotion'),
                                         transforms.Lambda(x: x + 1)
                                     ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)

    Args:  
        root_path (str): Downloaded data files in matlab (unzipped ExtractedFeatures.zip) formats (default: :obj:`'./ExtractedFeatures'`)
        feature (list): A list of selected feature names. The selected features corresponding to each electrode will be concatenated together. Feature names supported by the SEED dataset include de_movingAve, de_LDS, psd_movingAve, and etc. If you want to know other supported feature names, please refer to :obj:`SEEDFeatureDataset.feature_list` (default: :obj:`['de_movingAve']`)
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

    positions = get_positions(channels)

    def __init__(self,
                 root_path: str = './ExtractedFeatures',
                 feature: list = ['de_movingAve'],
                 num_channel: int = 62,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
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
            'after_session': after_session,
            'after_subject': after_subject,
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
                    root_path: str = './ExtractedFeatures',
                    **kwargs) -> Dict:

        file_name, _ = record

        samples = scio.loadmat(os.path.join(root_path, file_name),
                               verify_compressed_data_integrity=False
                               )  # trial (15), channel(62), timestep(n*200)
        # label file
        labels = scio.loadmat(
            os.path.join(root_path, 'label.mat'),
            verify_compressed_data_integrity=False)['label'][0]

        return {
            'samples': samples,
            'labels': labels
        }

    @staticmethod
    def fake_record(record: str, **kwargs) -> Dict:
        num_trials = 15
        num_channels = 62
        num_timesteps = 200
        num_bands = 5

        samples = {}
        for trial in range(1, num_trials + 1):
            key = f'de_movingAve{trial}'
            samples[key] = np.random.randn(
                num_timesteps, num_channels, num_bands) * 0.5

        labels = np.random.randint(0, 3, num_trials)

        return {
            'record': ('10_20131130.mat', 0),
            'samples': samples,
            'labels': labels
        }

    @staticmethod
    def process_record(record: Tuple,
                       samples: Dict,
                       labels: np.ndarray,
                       feature: list = ['de_movingAve'],
                       num_channel: int = 62,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        file_name, session_id = record

        subject = int(os.path.basename(file_name).split('.')[0].split('_')
                      [0])  # subject (15)
        date = int(os.path.basename(file_name).split('.')[0].split('_')
                   [1])  # period (3)

        trial_ids = [
            int(re.findall(r"de_movingAve(\d+)", key)[0])
            for key in samples.keys() if 'de_movingAve' in key
        ]
        write_pointer = 0
        # loop for each trial
        for trial_id in trial_ids:
            # extract baseline signals
            trial_samples = []
            for cur_feature in feature:
                trial_samples.append(
                    samples[cur_feature + str(trial_id)].transpose(
                        (1, 0, 2)))  # timestep(n), channel(62), bands(5)
            trial_samples = np.concatenate(
                trial_samples,
                axis=-1)[:, :
                         # timestep(n*k), channel(62), features(5)
                         num_channel]

            if before_trial:
                trial_samples = before_trial(trial_samples)

            # record the common meta info
            trial_meta_info = {
                'subject_id': subject,
                'trial_id': trial_id,
                'emotion': int(labels[trial_id - 1]),
                'date': date,
                'session_id': session_id
            }

            for i, clip_sample in enumerate(trial_samples):
                t_eeg = clip_sample
                if not offline_transform is None:
                    t_eeg = offline_transform(eeg=clip_sample)['eeg']

                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1

                # record meta info for each signal
                record_info = {
                    'start_at': i * 200,
                    'end_at': (i + 1) * 200,
                    'clip_id': clip_id
                }
                record_info.update(trial_meta_info)
                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

    def set_records(self, root_path: str = './ExtractedFeatures', **kwargs):
        if not (os.path.exists(root_path)):
            raise FileNotFoundError(
                f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
            )
        file_list = os.listdir(root_path)
        skip_set = ['label.mat', 'readme.txt']
        file_list = [f for f in file_list if f not in skip_set]

        # get subject_id
        subject_id_list = set()
        for file_name in file_list:
            subject_id = int(file_name.split('_')[0])
            subject_id_list.add(subject_id)
        subject_id_list = list(subject_id_list)

        # get session_id
        file_name_session_id_list = []
        for subject_id in subject_id_list:
            subject_file_list = [
                file_name for file_name in file_list
                if int(file_name.split('_')[0]) == subject_id
            ]
            # sort the subject_file_list, based on the date
            subject_file_list.sort(key=lambda x: int(x.split('_')[1][:-4]))
            # get file and corresponding session_id
            for i, file_name in enumerate(subject_file_list):
                file_name_session_id_list.append((file_name, i))

        return file_name_session_id_list

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
