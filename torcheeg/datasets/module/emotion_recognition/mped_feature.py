import os
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import scipy.io as scio

from torcheeg.io import EEGSignalIO, MetaInfoIO

from ...constants.emotion_recognition.mped import (MPED_ADJACENCY_MATRIX,
                                                   MPED_CHANNEL_LOCATION_DICT)
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
                                  transforms.Select('emotion')
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
                                  transforms.Select('emotion')
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
                                  transforms.Select('emotion')
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
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
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
                   root_path: str = './EEG_feature',
                   feature: list = ['PSD'],
                   num_channel: int = 62,
                   before_trial: Union[None, Callable] = None,
                   offline_transform: Union[None, Callable] = None,
                   after_trial: Union[None, Callable] = None,
                   **kwargs):
        file_name = file

        labels = [
            0, 2, 1, 4, 3, 5, 6, 7, 1, 2, 3, 6, 7, 4, 5, 0, 1, 5, 6, 2, 2, 1, 7,
            6, 4, 4, 3, 5, 3, 7
        ]  # 0-resting status, 1-neutral, 2-joy, 3-funny, 4-angry, 5-fear, 6-disgust, 7-sadness

        subject = os.path.basename(file_name).split('.')[0].split('_')[0]

        samples_dict = {}
        for cur_feature in feature:
            samples_dict[cur_feature] = scio.loadmat(
                os.path.join(root_path, cur_feature, file_name),
                verify_compressed_data_integrity=False)  # (1, 30)
        write_pointer = 0

        for trial_id in range(30):
            trial_samples = []
            for cur_feature, samples in samples_dict.items():
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

            trial_queue = []
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

                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1

                # record meta info for each signal
                record_info = {'clip_id': clip_id}
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
    def _set_files(root_path: str = './EEG_feature',
                   feature: list = ['PSD'],
                   **kwargs):
        avaliable_features = os.listdir(root_path)

        assert set(feature).issubset(
            set(avaliable_features)
        ), 'The features supported by the MPEDFeature dataset are HHS, Hjorth, PSD, STFT, HOC. The input features are not a subset of the list of supported features.'

        file_list = os.listdir(os.path.join(root_path, avaliable_features[0]))
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
