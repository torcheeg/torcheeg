import os
from typing import Callable, Dict, Tuple, Union, Any
import scipy.io as scio
from ...constants.emotion_recognition.dreamer import (
    DREAMER_ADJACENCY_MATRIX, DREAMER_CHANNEL_LOCATION_DICT)
from ..base_dataset import BaseDataset
from torcheeg.io import EEGSignalIO, MetaInfoIO


class DREAMERDataset(BaseDataset):
    r'''
    A multi-modal database consisting of electroencephalogram and electrocardiogram signals recorded during affect elicitation by means of audio-visual stimuli. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Katsigiannis et al.
    - Year: 2017
    - Download URL: https://zenodo.org/record/546113
    - Reference: Katsigiannis S, Ramzan N. DREAMER: A database for emotion recognition through EEG and ECG signals from wireless low-cost off-the-shelf devices[J]. IEEE journal of biomedical and health informatics, 2017, 22(1): 98-107.
    - Stimulus: 18 movie clips. 
    - Signals: Electroencephalogram (14 channels at 128Hz), and electrocardiogram (2 channels at 256Hz) of 23 subjects.
    - Rating: Arousal, valence, like/dislike, dominance, familiarity (all ona scale from 1 to 5).

    In order to use this dataset, the download file :obj:`DREAMER.mat` is required.

    An example dataset for CNN-based methods:

    .. code-block:: python

        dataset = DREAMERDataset(io_path=f'./dreamer',
                              mat_path='./DREAMER.mat',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DREAMER_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(3.0),
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[4, 9, 9]),
        # coresponding baseline signal (torch.Tensor[4, 9, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = DREAMERDataset(io_path=f'./dreamer',
                              mat_path='./DREAMER.mat',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select(['valence', 'arousal']),
                                  transforms.Binary(3.0),
                                  transforms.BinariesToCategory()
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 14, 128]),
        # coresponding baseline signal (torch.Tensor[1, 14, 128]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python
    
        dataset = DREAMERDataset(io_path=f'./dreamer',
                              mat_path='./DREAMER.mat',
                              online_transform=transforms.Compose([
                                  ToG(DREAMER_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('arousal'),
                                  transforms.Binary(3.0)
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = DREAMERDataset(io_path=f'./dreamer',
                              mat_path='./DREAMER.mat',
                              online_transform=transforms.Compose([
                                  ToG(DREAMER_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('arousal'),
                                  transforms.Binary(3.0)
                              ]),
                              num_worker=4)
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
    
    Args:
        mat_path (str): Downloaded data files in pickled matlab formats (default: :obj:`'./DREAMER.mat'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`128`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 14 channels are EEG signals. (default: :obj:`14`)
        num_baseline (int): Number of baseline signal chunks used. (default: :obj:`61`)
        baseline_chunk_size (int): Number of data points included in each baseline signal chunk. The baseline signal in the DREAMER dataset has a total of 7808 data points. (default: :obj:`128`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/dreamer`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)       
    '''
    channel_location_dict = DREAMER_CHANNEL_LOCATION_DICT
    adjacency_matrix = DREAMER_ADJACENCY_MATRIX

    def __init__(self,
                 mat_path: str = './DREAMER.mat',
                 chunk_size: int = 128,
                 overlap: int = 0,
                 num_channel: int = 14,
                 num_baseline: int = 61,
                 baseline_chunk_size: int = 128,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 io_path: str = './io/dreamer',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):
        # pass all arguments to super class
        params = {
            'mat_path': mat_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'num_baseline': num_baseline,
            'baseline_chunk_size': baseline_chunk_size,
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
                   mat_path: str = './DREAMER.mat',
                   chunk_size: int = 128,
                   overlap: int = 0,
                   num_channel: int = 14,
                   num_baseline: int = 61,
                   baseline_chunk_size: int = 128,
                   before_trial: Union[None, Callable] = None,
                   offline_transform: Union[None, Callable] = None,
                   after_trial: Union[None, Callable] = None,
                   **kwargs):
        subject = file
        mat_data = scio.loadmat(mat_path,
                                verify_compressed_data_integrity=False)

        trial_len = len(
            mat_data['DREAMER'][0, 0]['Data'][0,
                                              0]['EEG'][0,
                                                        0]['stimuli'][0,
                                                                      0])  # 18

        write_pointer = 0
        # loop for each trial
        for trial_id in range(trial_len):
            # extract baseline signals
            trial_baseline_sample = mat_data['DREAMER'][0, 0]['Data'][
                0, subject]['EEG'][0, 0]['baseline'][0, 0][trial_id, 0]
            trial_baseline_sample = trial_baseline_sample[:, :num_channel].swapaxes(
                1, 0)  # channel(14), timestep(61*128)
            trial_baseline_sample = trial_baseline_sample[:, :num_baseline *
                                                          baseline_chunk_size].reshape(
                                                              num_channel,
                                                              num_baseline,
                                                              baseline_chunk_size
                                                          ).mean(
                                                              axis=1
                                                          )  # channel(14), timestep(128)

            # record the common meta info
            trial_meta_info = {'subject_id': subject, 'trial_id': trial_id}

            trial_meta_info['valence'] = mat_data['DREAMER'][0, 0]['Data'][
                0, subject]['ScoreValence'][0, 0][trial_id, 0]
            trial_meta_info['arousal'] = mat_data['DREAMER'][0, 0]['Data'][
                0, subject]['ScoreArousal'][0, 0][trial_id, 0]
            trial_meta_info['dominance'] = mat_data['DREAMER'][0, 0]['Data'][
                0, subject]['ScoreDominance'][0, 0][trial_id, 0]

            trial_samples = mat_data['DREAMER'][0, 0]['Data'][
                0, subject]['EEG'][0, 0]['stimuli'][0, 0][trial_id, 0]
            trial_samples = trial_samples[:, :num_channel].swapaxes(
                1, 0)  # channel(14), timestep(n*128)

            if before_trial:
                trial_samples = before_trial(trial_samples)

            start_at = 0
            if chunk_size <= 0:
                chunk_size = trial_samples.shape[1] - start_at

            # chunk with chunk size
            end_at = chunk_size
            # calculate moving step
            step = chunk_size - overlap

            trial_queue = []
            while end_at <= trial_samples.shape[1]:
                clip_sample = trial_samples[:, start_at:end_at]

                t_eeg = clip_sample
                t_baseline = trial_baseline_sample

                if not offline_transform is None:
                    t = offline_transform(eeg=clip_sample,
                                          baseline=trial_baseline_sample)
                    t_eeg = t['eeg']
                    t_baseline = t['baseline']

                # put baseline signal into IO
                if not 'baseline_id' in trial_meta_info:
                    trial_base_id = f'{subject}_{write_pointer}'
                    yield {'eeg': t_baseline, 'key': trial_base_id}
                    write_pointer += 1
                    trial_meta_info['baseline_id'] = trial_base_id

                clip_id = f'{subject}_{write_pointer}'
                write_pointer += 1

                # record meta info for each signal
                record_info = {
                    'start_at': start_at,
                    'end_at': end_at,
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

                start_at = start_at + step
                end_at = start_at + chunk_size

            if len(trial_queue) and after_trial:
                trial_queue = after_trial(trial_queue)
                for obj in trial_queue:
                    assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
                    yield obj

    @staticmethod
    def _set_files(mat_path: str = './DREAMER.mat', **kwargs):

        mat_data = scio.loadmat(mat_path,
                                verify_compressed_data_integrity=False)

        subject_len = len(mat_data['DREAMER'][0, 0]['Data'][0])  # 23
        return list(range(subject_len))

    def __getitem__(self, index: int) -> Tuple:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg = self.read_eeg(eeg_index)

        baseline_index = str(info['baseline_id'])
        baseline = self.read_eeg(baseline_index)

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
                'mat_path': self.mat_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'num_baseline': self.num_baseline,
                'baseline_chunk_size': self.baseline_chunk_size,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
