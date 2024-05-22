import os
import pickle as pkl
from typing import Any, Callable, Dict, Tuple, Union

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset
from .faced import EMOTION_DICT, VALENCE_DICT


class FACEDFeatureDataset(BaseDataset):
    r'''
    The FACED dataset was provided by the Tsinghua Laboratory of Brain and Intelligence. The Finer-grained Affective Computing EEG Dataset (FACED) recorded 32-channel EEG signals from 123 subjects. During the experiment, subjects watched 28 emotion-elicitation video clips covering nine emotion categories (amusement, inspiration, joy, tenderness; anger, fear, disgust, sadness, and neutral emotion), providing a fine-grained and balanced categorization on both the positive and negative sides of emotion.
    This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Please refer to the downloaded URL.
    - Year: 2023
    - Download URL: https://www.synapse.org/#!Synapse:syn50614194/files/
    - Reference: Please refer to the downloaded URL.
    - Stimulus: video clips.
    - Signals: Electroencephalogram (30 channels at 250Hz) and two channels of left/right mastoid signals from 123 subjects.
    - Rating: 28 video clips are annotated in valence and discrete emotion dimensions. The valence is divided into positive (1), negative (-1), and neutral (0). Discrete emotions are divided into anger (0), disgust (1), fear (2), sadness (3), neutral (4), amusement (5), inspiration (6), joy (7), and tenderness (8).

    In order to use this dataset, the download folder :obj:`EEG_Features`(download from this url: https://www.synapse.org/#!Synapse:syn52368847) is required, containing the following files:
    
    - EEG_Features
        - DE
            + sub000.pkl.pkl (two .pkl is a mistake, when __init__ function is auto called, it will be renamed to sub000.pkl by func: rename_pkl_files)
            + sub001.pkl.pkl
            + sub002.pkl.pkl
            + ...
        - PSD
            + sub000.pkl.pkl
            + ...
    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import FACEDFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets.constants import FACED_CHANNEL_LOCATION_DICT
        
        dataset = FACEDFeatureDataset(root_path='./EEG_Features/DE',
                                       offline_transform=transforms.ToGrid(FACED_CHANNEL_LOCATION_DICT),
                                       online_transform=transforms.ToTensor(),
                                       label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch.Tensor[5, 8, 9]),
        # coresponding baseline signal (torch.Tensor[5, 8, 9]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import FACEDFeatureDataset
        from torcheeg import transforms
        from torcheeg.datasets.constants import FACED_ADJACENCY_MATRIX
        from torcheeg.transforms.pyg import ToG
        
        dataset = FACEDFeatureDataset(root_path='./EEG_Features/DE',
                                       online_transform=ToG(FACED_ADJACENCY_MATRIX),
                                       label_transform=transforms.Select('emotion'))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)
        
    Args:
        root_path (str): Downloaded data files in matlab (unzipped EEG_Features.zip) formats (default: :obj:`'./EEG_Features/DE'`, optional: :obj:`'./EEG_Features/PSD'`)
        num_channel (int): Number of channels used, of which the first 30 channels are EEG signals. (default: :obj:`30`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 3D EEG signal with shape (number of electrodes, number of windows, number of frequency bands), whose ideal output shape is also (number of electrodes, number of windows, number of frequency bands).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)    
    '''

    def rename_pkl_files(self, root_path):
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith('.pkl.pkl'):
                    old_file_path = os.path.join(dirpath, filename)
                    # new file name, remove the last repetitive '.pkl'
                    new_filename = filename[:-4]
                    new_file_path = os.path.join(dirpath, new_filename)
                    # rename file
                    os.rename(old_file_path, new_file_path)

    def __init__(self,
                 root_path: str = './EEG_Features/DE',
                 num_channel: int = 30,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        self.rename_pkl_files(root_path)
        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'num_channel': num_channel,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
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
    def process_record(num_channel: int = 30,
                       offline_transform: Union[None, Callable] = None,
                       before_trial: Union[None, Callable] = None,
                       file: Any = None,
                       **kwargs):
        # get file name from path ,such as 'sub087.pkl'
        file_name = os.path.basename(file)

        subject_id = int(file_name.split('.')[0]
                         [3:])  # get int value from 'sub087.pkl', such as 87

        # load the file
        with open(os.path.join(file), 'rb') as f:
            data = pkl.load(
                f, encoding='iso-8859-1'
            )  # 28(trials), 32(channels), 30s(time points), 5(frequency bands)

        write_pointer = 0

        # loop all trials
        for trial_id in range(len(data)):
            trial_samples = data[
                trial_id, :
                num_channel]  # 30(channels), 30s(time points), 5(frequency bands)

            trial_meta_info = {'subject_id': subject_id, 'trial_id': trial_id}

            if before_trial:
                trial_samples = before_trial(trial_samples)

            # loop all clips
            for i in range(trial_samples.shape[1]):
                t_eeg = trial_samples[:, i]

                clip_id = f'{file_name}_{write_pointer}'
                write_pointer += 1

                record_info = {
                    'start_at': i * 250,
                    'end_at': (i + 1) *
                    250,  # The size of the sliding time windows for feature 
                    'clip_id': clip_id,
                    'valence': VALENCE_DICT[trial_id + 1],
                    'emotion': EMOTION_DICT[trial_id + 1],
                }
                record_info.update(trial_meta_info)

                if not offline_transform is None:
                    t_eeg = offline_transform(eeg=t_eeg)['eeg']

                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

    def set_records(self, root_path: str = './EEG_Features/DE', **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_path_list = os.listdir(root_path)
        file_path_list.sort()
        file_path_list = [
            os.path.join(root_path, file_path) for file_path in file_path_list
            if file_path.endswith('.pkl')
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
