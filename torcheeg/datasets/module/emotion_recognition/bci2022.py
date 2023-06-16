import os
from typing import Any, Callable, Dict, Tuple, Union

import joblib

from torcheeg.io import EEGSignalIO, MetaInfoIO

from ...constants.emotion_recognition.bci2022 import (
    BCI2022_ADJACENCY_MATRIX, BCI2022_CHANNEL_LOCATION_DICT)
from ..base_dataset import BaseDataset

FIRST_BATCH_CHANNEL = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T7', 'T8', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ',
    'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2'
]

SECOND_BATCH_CHANNEL = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4',
    'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2', 'A2', 'A1'
]

FIRST_TO_SECOND_ORDER = [
    FIRST_BATCH_CHANNEL.index(c) for c in SECOND_BATCH_CHANNEL
]

VALENCE_DICT = {
    1: -1,  # negative
    2: -1,
    3: -1,
    4: -1,
    5: -1,
    6: -1,
    7: -1,
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: 0,  # neutral
    14: 0,
    15: 0,
    16: 0,
    17: 1,  # positive
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1
}

EMOTION_DICT = {
    1: 0,  # anger	
    2: 0,
    3: 0,
    4: 1,  # disgust
    5: 1,
    6: 1,
    7: 2,  # fear
    8: 2,
    9: 2,
    10: 3,  # sadness
    11: 3,
    12: 3,
    13: 4,  # neutral
    14: 4,
    15: 4,
    16: 4,
    17: 5,  # amusement
    18: 5,
    19: 5,
    20: 6,  # excitation
    21: 6,
    22: 6,
    23: 7,  # happy
    24: 7,
    25: 7,
    26: 8,  # warmth
    27: 8,
    28: 8
}


class BCI2022Dataset(BaseDataset):
    r'''
    The 2022 EMOTION_BCI competition aims at tackling the cross-subject emotion recognition challenge and provides participants with a batch of EEG data from 80 participants with known emotional state information. Participants are required to establish an EEG computing model with cross-individual emotion recognition ability. The subjects' EEG data were used for real-time emotion recognition. This class generates training samples and test samples according to the given parameters and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Please refer to the downloaded URL.
    - Year: 2022
    - Download URL: https://oneuro.cn/n/competitiondetail/2022_emotion_bci/doc0
    - Reference: Please refer to the downloaded URL.
    - Stimulus: video clips.
    - Signals: Electroencephalogram (30 channels at 250Hz) and two channels of left/right mastoid signals from 80 subjects.
    - Rating: 28 video clips are annotated in valence and discrete emotion dimensions. The valence is divided into positive (1), negative (-1), and neutral (0). Discrete emotions are divided into anger (0), disgust (1), fear (2), sadness (3), neutral (4), amusement (5), excitation (6), happiness (7), and warmth (8).

    In order to use this dataset, the download folder :obj:`TrainSet` is required, containing the following files:
    
    - TrainSet_first_batch

        + sub1
        + sub10
        + sub11
        + ...

    - TrainSet_second_batch

        + sub55
        + sub57
        + sub59
        + ...

    An example dataset for CNN-based methods:

    .. code-block:: python

        dataset = BCI2022Dataset(io_path=f'./bci2022',
                              root_path='./TrainSet',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(BCI2022_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch.Tensor[4, 8, 9]),
        # coresponding baseline signal (torch.Tensor[4, 8, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = BCI2022Dataset(io_path=f'./bci2022',
                              root_path='./TrainSet',
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch.Tensor[30, 250]),
        # coresponding baseline signal (torch.Tensor[30, 250]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python
    
        dataset = BCI2022Dataset(io_path=f'./bci2022',
                              root_path='./TrainSet',
                              online_transform=transforms.Compose([
                                  transforms.ToG(BCI2022_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
        
    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = BCI2022Dataset(io_path=f'./bci2022',
                              root_path='./TrainSet',
                              online_transform=transforms.Compose([
                                  transforms.ToG(BCI2022_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Select('emotion'),
                              num_worker=4)
            print(dataset[0])
            # EEG signal (torch_geometric.data.Data),
            # coresponding baseline signal (torch_geometric.data.Data),
            # label (int)

    Args:
        root_path (str): Downloaded data files in pickle (the TrainSet folder in unzipped 2022EmotionPublic.zip) formats (default: :obj:`'./TrainSet'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`250`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        channel_num (int): Number of channels used, of which the first 30 channels are EEG signals. (default: :obj:`30`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/bci2022`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)    
    '''
    channel_location_dict = BCI2022_CHANNEL_LOCATION_DICT
    adjacency_matrix = BCI2022_ADJACENCY_MATRIX

    def __init__(self,
                 root_path: str = './2022EmotionPublic/TrainSet/',
                 chunk_size: int = 250,
                 overlap: int = 0,
                 channel_num: int = 30,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 io_path: str = './io/bci2022',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):
        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'channel_num': channel_num,
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
                   chunk_size: int = 250,
                   overlap: int = 0,
                   channel_num: int = 30,
                   before_trial: Union[None, Callable] = None,
                   offline_transform: Union[None, Callable] = None,
                   after_trial: Union[None, Callable] = None,
                   **kwargs):

        file_name = os.path.basename(file)  # an element from file name list
        reorder = 'TrainSet_first_batch' in file_name  # bool, the first batch needs to be reordered

        subject = file_name  # subject (54)
        samples = joblib.load(os.path.join(
            file, f'{file_name}.pkl'))  # channel(33), timestep(n*250)

        events = samples[-1]
        if reorder:
            samples = samples.take(FIRST_TO_SECOND_ORDER, axis=0)

        # initial pointers
        trial_id = 0
        write_pointer = 0

        video_id = None
        start_at = None
        end_at = None

        # loop for each trial
        for i, event in enumerate(events):

            if event in list(range(1, 29)):
                # Video events 1-28: Different events correspond to different experimental video materials
                video_id = event
            elif event == 240:
                # Current trial video start event 240: This event appears 0.1s after the video event, indicating that the video starts to play.
                start_at = i
            elif event == 241:
                # Current trial video end event 241: This event indicates that the video ends playing. Block end event 243: This event indicates the end of the block.
                end_at = i
                assert (not video_id is None) and (
                    not start_at is None
                ), f'Parse event fail for trial {trial_id} with video_id={video_id}, start_at={start_at}, end_at={end_at}!'

                trial_meta_info = {
                    'trial_id': trial_id,
                    'video_id': video_id,
                    'subject_id': subject,
                    'valence': VALENCE_DICT[video_id],
                    'emotion': EMOTION_DICT[video_id],
                }

                cur_start_at = start_at
                if chunk_size <= 0:
                    chunk_size = end_at - start_at
                cur_end_at = cur_start_at + chunk_size
                step = chunk_size - overlap

                trial_queue = []

                if before_trial:
                    samples[:channel_num, cur_start_at:end_at] = before_trial(
                        samples[:channel_num, cur_start_at:end_at])

                while cur_end_at <= end_at:
                    t_eeg = samples[:channel_num, cur_start_at:cur_end_at]
                    if not offline_transform is None:
                        t_eeg = offline_transform(eeg=t_eeg)['eeg']

                    clip_id = f'{subject}_{write_pointer}'
                    write_pointer += 1

                    record_info = {
                        'start_at': cur_start_at,
                        'end_at': cur_end_at,
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
                        yield {
                            'eeg': t_eeg,
                            'key': clip_id,
                            'info': record_info
                        }

                    cur_start_at = cur_start_at + step
                    cur_end_at = cur_start_at + chunk_size

                if len(trial_queue) and after_trial:
                    trial_queue = after_trial(trial_queue)
                    for obj in trial_queue:
                        assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
                        yield obj

                # prepare for the next trial
                trial_id += 1
                video_id = None
                start_at = None
                end_at = None

    @staticmethod
    def _set_files(root_path: str = './data_preprocessed_python', **kwargs):
        outputs = []
        for train_set_batch in [
                'TrainSet_first_batch', 'TrainSet_second_batch'
        ]:
            # loop to access the first batch files
            file_list = os.listdir(os.path.join(root_path, train_set_batch))
            for file in file_list:
                # loop to access the second batch files
                file_path = os.path.join(root_path, train_set_batch, file)
                outputs.append(file_path)
        return outputs

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
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'channel_num': self.channel_num,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })