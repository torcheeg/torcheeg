import os
from typing import Any, Callable, Dict, Tuple, Union

import mne

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset

mne.set_log_level('CRITICAL')


class SEEDVDataset(BaseDataset):
    r'''
    The SEED-V dataset provided by the BCMI laboratory, which is led by Prof. Bao-Liang Lu. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Liu et al.
    - Year: 2021
    - Download URL: https://bcmi.sjtu.edu.cn/home/seed/seed-v.html
    - Reference: Liu W, Qiu J L, Zheng W L, et al. Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition[J]. IEEE Transactions on Cognitive and Developmental Systems, 2021, 14(2): 715-729.
    - Stimulus: 15 pieces of stimulating material.
    - Signals: Electroencephalogram (62 channels at 200Hz) and eye movement data of 20 subjects (20 females). Each subject conducts the experiments in three sessions, and each session contains 15 trials (3 per emotional category) totally 20 people x 3 sessions x 15 trials.
    - Rating: disgust (0), fear (1), sad (2), neutral (3), happy (4).

    In order to use this dataset, the download folder :obj:`EEG_raw` is required, containing the following files:
    
    - 10_1_20180507.cnt
    - 10_2_20180524.cnt
    - 10_3_20180626.cnt
    - ...
    - 9_3_20180728.cnt

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import SEEDVDataset
        from torcheeg import transforms
        from torcheeg.datasets.constants import SEED_V_CHANNEL_LOCATION_DICT
        
        dataset = SEEDVDataset(root_path='./EEG_raw',
                                offline_transform=transforms.Compose([
                                    transforms.BandDifferentialEntropy(),
                                    transforms.ToGrid(SEED_V_CHANNEL_LOCATION_DICT)
                                ]),
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Compose([
                                    transforms.Select('emotion'),
                                    transforms.Lambda(lambda x: x + 1)
                                ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[4, 9, 9]),
        # coresponding baseline signal (torch.Tensor[4, 9, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import SEEDVDataset
        from torcheeg import transforms

        dataset = SEEDVDataset(root_path='./EEG_raw',
                                online_transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.To2d()
                                ]),
                                label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch.Tensor[62, 200]),
        # coresponding baseline signal (torch.Tensor[62, 200]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import SEEDVDataset
        from torcheeg import transforms
        from torcheeg.datasets.constants import SEED_V_ADJACENCY_MATRIX
        from torcheeg.transforms.pyg import ToG
        
        dataset = SEEDVDataset(root_path='./EEG_raw',
                                online_transform=transforms.Compose([
                                    ToG(SEED_V_ADJACENCY_MATRIX)
                                ]),
                                label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
        
    Args:
        root_path (str): Downloaded data files in matlab (unzipped EEG_raw.zip) formats (default: :obj:`'./EEG_raw'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`800`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 62 channels are EEG signals. (default: :obj:`62`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)    
    '''

    def __init__(self,
                 root_path: str = './EEG_raw',
                 chunk_size: int = 800,
                 overlap: int = 0,
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
            'chunk_size': chunk_size,
            'overlap': overlap,
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
    def process_record(file: Any = None,
                       chunk_size: int = 800,
                       overlap: int = 0,
                       num_channel: int = 62,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        file_name = os.path.basename(file)
        # split with _, the first part is the subject_id, the second part is the session_id, the third part is the date
        subject_id, session_id, date = file_name.split('_')[:3]

        labels = [[4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
                  [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
                  [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0]]

        trial_labels = labels[int(session_id) - 1]

        eeg_raw = mne.io.read_raw_cnt(file)

        useless_ch = ['M1', 'M2', 'VEO', 'HEO']
        eeg_raw.drop_channels(useless_ch)

        data_matrix = eeg_raw.get_data()

        start_end_list = [
            {
                'start_seconds': [
                    30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227,
                    2435, 2667, 2932, 3204
                ],
                'end_seconds': [
                    102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401,
                    2607, 2901, 3172, 3359
                ]
            },
            {
                'start_seconds': [
                    30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966,
                    2186, 2333, 2490, 2741
                ],
                'end_seconds': [
                    267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153,
                    2302, 2428, 2709, 2817
                ]
            },
            {
                'start_seconds': [
                    30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055,
                    2307, 2457, 2726, 2888
                ],
                'end_seconds': [
                    321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275,
                    2425, 2664, 2857, 3066
                ]
            },
        ]
        start_seconds = start_end_list[int(session_id) - 1]['start_seconds']
        end_seconds = start_end_list[int(session_id) - 1]['end_seconds']

        write_pointer = 0

        for trial_id, (start_second,
                       end_second) in enumerate(zip(start_seconds,
                                                    end_seconds)):

            trial_meta_info = {
                'subject_id': subject_id,
                'session_id': session_id,
                'date': date,
                'trial_id': trial_id,
                'emotion': trial_labels[trial_id]
            }

            trial_samples = data_matrix[:,
                                        start_second * 1000:end_second * 1000]

            # downsample to 200Hz
            trial_samples = trial_samples[:, ::5]

            #  EEG data are then processed with a bandpass filter between 1 Hz and 75 Hz.
            trial_samples = mne.filter.filter_data(trial_samples,
                                                   sfreq=200,
                                                   l_freq=1,
                                                   h_freq=75)

            if not before_trial is None:
                trial_samples = before_trial(trial_samples)

            # extract experimental signals
            start_at = 0
            if chunk_size <= 0:
                dynamic_chunk_size = trial_samples.shape[1] - start_at
            else:
                dynamic_chunk_size = chunk_size

            # chunk with chunk size
            end_at = dynamic_chunk_size
            # calculate moving step
            step = dynamic_chunk_size - overlap

            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:num_channel, start_at:end_at]

                t_eeg = clip_sample
                if not offline_transform is None:
                    t_eeg = offline_transform(eeg=clip_sample)['eeg']

                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1

                record_info = {
                    'clip_id': clip_id,
                    'start_at': start_at,
                    'end_at': end_at
                }
                record_info.update(trial_meta_info)

                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size

    def set_records(self, root_path: str = './EEG_raw', **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_list = os.listdir(root_path)
        file_list = [
            os.path.join(root_path, file) for file in file_list
            if file.endswith('.cnt')
        ]
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
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
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
