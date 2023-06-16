import os
import re
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import scipy.io as scio

from torcheeg.io import EEGSignalIO, MetaInfoIO

from ...constants.emotion_recognition.seed import (SEED_ADJACENCY_MATRIX,
                                                   SEED_CHANNEL_LOCATION_DICT)
from ..base_dataset import BaseDataset


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
    
    - label.mat
    - readme.txt
    - 10_20131130.mat
    - ...
    - 9_20140704.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        dataset = SEEDFeatureDataset(io_path=f'./seed',
                              root_path='./ExtractedFeatures',
                              feature=['de_movingAve'],
                              offline_transform=transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
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
    
        dataset = SEEDFeatureDataset(io_path=f'./seed',
                              root_path='./Preprocessed_EEG',
                              features=['de_movingAve'],
                              online_transform=ToG(SEED_ADJACENCY_MATRIX),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(x: x + 1)
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
        
    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = SEEDFeatureDataset(io_path=f'./seed_feature',
                              root_path='./ExtractedFeatures',
                              feature=['de_movingAve'],
                              offline_transform=transforms.ToGrid(SEED_CHANNEL_LOCATION_DICT),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: x + 1)
                              ]),
                              num_worker=4)
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
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/seed_feature`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)    
    '''
    channel_location_dict = SEED_CHANNEL_LOCATION_DICT
    adjacency_matrix = SEED_ADJACENCY_MATRIX

    def __init__(self,
                 root_path: str = './ExtractedFeatures',
                 feature: list = ['de_movingAve'],
                 num_channel: int = 62,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 io_path: str = './io/seed_feature',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):
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
            'verbose': verbose,
            'in_memory': in_memory
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def _load_data(file: Any = None,
                   root_path: str = './ExtractedFeatures',
                   feature: list = ['de_movingAve'],
                   num_channel: int = 62,
                   before_trial: Union[None, Callable] = None,
                   offline_transform: Union[None, Callable] = None,
                   after_trial: Union[None, Callable] = None,
                   **kwargs):
        file_name = file

        subject = int(os.path.basename(file_name).split('.')[0].split('_')
                      [0])  # subject (15)
        date = int(os.path.basename(file_name).split('.')[0].split('_')
                   [1])  # period (3)

        samples = scio.loadmat(os.path.join(root_path, file_name),
                               verify_compressed_data_integrity=False
                               )  # trial (15), channel(62), timestep(n*200)
        # label file
        labels = scio.loadmat(
            os.path.join(root_path, 'label.mat'),
            verify_compressed_data_integrity=False)['label'][0]

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
                trial_samples.append(samples[
                    cur_feature +
                    str(trial_id)])  # channel(62), timestep(n), bands(5)
            trial_samples = np.concatenate(
                trial_samples,
                axis=-1)[:
                         num_channel]  # channel(62), timestep(n), features(5*k)
            trial_samples = trial_samples.transpose((1, 0, 2))

            if before_trial:
                trial_samples = before_trial(trial_samples)

            # record the common meta info
            trial_meta_info = {
                'subject_id': subject,
                'trial_id': trial_id,
                'emotion': int(labels[trial_id - 1]),
                'date': date
            }

            trial_queue = []
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
                if after_trial:
                    trial_queue.append({
                        'eeg': t_eeg,
                        'key': clip_id,
                        'info': record_info
                    })
                else:
                    yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

            if len(trial_queue) and after_trial:
                trial_queue = after_trial(trial_queue)
                for obj in trial_queue:
                    assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
                    yield obj

    @staticmethod
    def _set_files(**kwargs):
        root_path = kwargs.pop('root_path', './ExtractedFeatures')  # str
        file_list = os.listdir(root_path)
        skip_set = ['label.mat', 'readme.txt']
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
