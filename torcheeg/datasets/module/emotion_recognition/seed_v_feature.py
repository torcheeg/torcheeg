import os
import pickle as pkl
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


class SEEDVFeatureDataset(BaseDataset):
    r'''
    The SEED-V dataset provided by the BCMI laboratory, which is led by Prof. Bao-Liang Lu. Since the SEED dataset provides features based on matlab, this class implements the processing of these feature files to initialize the dataset. The relevant information of the dataset is as follows:

    - Author: Liu et al.
    - Year: 2021
    - Download URL: https://bcmi.sjtu.edu.cn/home/seed/seed-v.html
    - Reference: Liu W, Qiu J L, Zheng W L, et al. Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition[J]. IEEE Transactions on Cognitive and Developmental Systems, 2021, 14(2): 715-729.
    - Stimulus: 15 pieces of stimulating material.
    - Signals: Electroencephalogram (62 channels at 200Hz) and eye movement data of 20 subjects (20 females). Each subject conducts the experiments in three sessions, and each session contains 15 trials (3 per emotional category) totally 20 people x 3 sessions x 15 trials.
    - Rating: disgust (0), fear (1), sad (2), neutral (3), happy (4).

    In order to use this dataset, the download folder :obj:`EEG_DE_features` is required, containing the following folder:
    
    - 1_123.npz
    - 2_123.npz
    - ...

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import SEEDVFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets.constants_v import SEED_V_CHANNEL_LOCATION_DICT
        
        dataset = SEEDVFeatureDataset(root_path='./EEG_DE_features',
                                       offline_transform=transforms.ToGrid         (SEED_V_CHANNEL_LOCATION_DICT),
                                       online_transform=transforms.ToTensor(),
                                       label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch.Tensor[5, 9, 9]),
        # coresponding baseline signal (torch.Tensor[5, 9, 9]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import SEEDVFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets.constants import SEED_ADJACENCY_MATRIX
        from torcheeg.transforms.pyg import ToG
        
        dataset = SEEDVFeatureDataset(root_path='./EEG_DE_features',
                                       online_transform=ToG(SEED_ADJACENCY_MATRIX),
                                       label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
        
    Args:
        root_path (str): Downloaded data files in matlab (unzipped ExtractedFeatures.zip) formats (default: :obj:`'./ExtractedFeatures'`)
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

    def __init__(self,
                 root_path: str = './EEG_DE_features',
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
    def process_record(num_channel: int = 62,
                       offline_transform: Union[None, Callable] = None,
                       before_trial: Union[None, Callable] = None,
                       file: Any = None,
                       **kwargs):
        # get file name from path
        file_name = os.path.basename(file)

        # split with '_', the fist part is subject id
        subject_id = file_name.split('_')[0]

        # load the file
        data_npz = np.load(file)
        data = pkl.loads(data_npz['data'])
        label = pkl.loads(data_npz['label'])

        write_pointer = 0

        # loop all trials
        for global_trial_id in range(len(data.keys())):
            trial_samples = data[list(data.keys())[global_trial_id]]
            trial_labels = label[global_trial_id]

            # split trial to 3 sessions
            session_id = global_trial_id // 15
            trial_id = global_trial_id % 15

            trial_meta_info = {
                'subject_id': subject_id,
                'session_id': session_id,
                'trial_id': trial_id
            }

            if before_trial:
                trial_samples = before_trial(trial_samples)

            # loop all clips
            for i in range(trial_samples.shape[0]):
                clip_sample = trial_samples[i]
                clip_label = trial_labels[i]

                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1

                record_info = {
                    'start_at': i * 800,
                    'end_at': (i + 1) *
                    800,  # The size of the sliding time windows for feature 
                    'clip_id': clip_id,
                    'emotion': int(clip_label)
                }
                record_info.update(trial_meta_info)

                t_eeg = clip_sample.reshape(62, 5)[:num_channel]
                if not offline_transform is None:
                    t_eeg = offline_transform(eeg=t_eeg)['eeg']

                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

    def set_records(self, root_path: str = './EEG_DE_features', **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_path_list = os.listdir(root_path)
        file_path_list = [
            os.path.join(root_path, file_path) for file_path in file_path_list
            if file_path.endswith('.npz')
        ]

        return file_path_list

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
