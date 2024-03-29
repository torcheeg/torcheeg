import os
import logging
from typing import Any, Callable, Dict, Tuple, Union
import numpy as np
import scipy.io as scio

from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path

log = logging.getLogger(__name__)


class BCICIV2aDataset(BaseDataset):
    r'''
    A dataset for motor imagery, BCI Competition 2008 Graz data set A (BCICIV_2a). This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:

    - Author: Tangermann et al.
    - Year: 2012
    - Download URL: http://bnci-horizon-2020.eu/database/data-sets
    - Reference: Tangermann M, Müller K R, Aertsen A, et al. Review of the BCI competition IV[J]. Frontiers in neuroscience, 2012: 55.
    - Signals: Electroencephalogram (22 channels at 250Hz) and electrocardiogram (3 channels at 250Hz) of 9 subjects. Two sessions on different days were recorded for each subject. Each session is comprised of 6 runs separated by short breaks. One run consists of 48 trials (12 for each of the four possible classes), yielding a total of 288 trials per session. For more detail, please refer to https://www.bbci.de/competition/iv/desc_2a.pdf
    - Category: The imagination of movement of the left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4).

    In order to use this dataset, the download folder :obj:`BCICIV_2a_mat` is required, containing the following files:

    - http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat
    - http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01E.mat
    - ...
    - http://bnci-horizon-2020.eu/database/data-sets/001-2014/A09E.mat

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import BCICIV2aDataset
        from torcheeg import transforms

        dataset = BCICIV2aDataset(root_path='./BCICIV_2a_mat',
                                  online_transform=transforms.Compose([
                                      transforms.To2d(),
                                      transforms.ToTensor()
                                  ]),
                                  label_transform=transforms.Compose([
                                      transforms.Select('label'),
                                      transforms.Lambda(lambda x: x - 1)
                                  ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 22, 1750])
        # label (int)

    Args:
        root_path (str): Downloaded data files in matlab (unzipped BCICIV_2a_mat.zip) formats (default: :obj:`'./BCICIV_2a_mat'`)
        offset (int): The number of data points to be discarded at the beginning of each trial. (default: :obj:`0`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`7 * 250`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 14 channels are EEG signals. (default: :obj:`22`)
        skip_trial_with_artifacts (bool): Whether to skip trials with artifacts. (default: :obj:`False`)
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
                 root_path: str = './BCICIV_2a_mat',
                 offset: int = 0,
                 chunk_size: int = 7 * 250,
                 overlap: int = 0,
                 num_channel: int = 22,
                 skip_trial_with_artifacts: bool = False,
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
            'offset': offset,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'skip_trial_with_artifacts': skip_trial_with_artifacts,
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
                       offset: int = 0,
                       chunk_size: int = 128,
                       overlap: int = 0,
                       num_channel: int = 14,
                       skip_trial_with_artifacts: bool = False,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        if chunk_size <= 0:
            dynamic_chunk_size = 7 * 250
        else:
            dynamic_chunk_size = int(chunk_size)

        # get file name without extension
        file_name = os.path.splitext(os.path.basename(file))[0]
        # the last letter of the file name is the session, the rest is the subject
        subject = file_name[:-1]
        session = file_name[-1]
        write_pointer = 0

        a_data = scio.loadmat(file)['data']

        for run_id in range(0, a_data.size):
            # a_data: (1, 9) struct, 1-3: 25 channel EOG test (eyes open, eyes closed, movement), 4-9: 6 runs

            a_data1 = a_data[0, run_id]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_X = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_artifacts = a_data3[5]

            a_X = np.transpose(a_X)  # to channel number, data point number
            if before_trial:
                a_X = before_trial(a_X)

            # for EOG test, a_trial is []
            for trial_id in range(0, a_trial.size):
                trial_meta_info = {
                    'subject_id': subject,
                    'trial_id': trial_id,
                    'session': session,
                    'subject': subject,
                    'run': run_id
                }

                if (a_artifacts[trial_id] != 0 and skip_trial_with_artifacts):
                    continue

                start_at = int(a_trial[trial_id] + offset)
                end_at = start_at + dynamic_chunk_size
                step = dynamic_chunk_size - overlap

                if trial_id < a_trial.size - 1:
                    trial_end_at = int(a_trial[trial_id + 1])
                else:
                    trial_end_at = a_X.shape[1]

                # print(start_at)
                # print(end_at)
                while end_at <= trial_end_at:
                    clip_id = f'{file_name}_{write_pointer}'

                    record_info = {
                        'start_at': start_at,
                        'end_at': end_at,
                        'clip_id': clip_id
                    }
                    record_info.update(trial_meta_info)

                    t_eeg = a_X[:num_channel, start_at:end_at]
                    if offline_transform is not None:
                        t = offline_transform(eeg=t_eeg)
                        t_eeg = t['eeg']

                    record_info['label'] = int(a_y[trial_id])

                    yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}
                    # print(t_eeg.shape)

                    write_pointer += 1

                    start_at = start_at + step
                    end_at = start_at + dynamic_chunk_size

    def set_records(self, root_path: str = './BCICIV_2a_mat', **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_name_list = os.listdir(root_path)
        file_path_list = [
            os.path.join(root_path, file) for file in file_name_list
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
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'skip_trial_with_artifacts': self.skip_trial_with_artifacts,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'after_session': self.after_session,
                'after_subject': self.after_subject,
                'io_path': self.io_path,
                'io_size': self.io_size,
                'io_mode': self.io_mode,
                'num_worker': self.num_worker,
                'verbose': self.verbose
            })
