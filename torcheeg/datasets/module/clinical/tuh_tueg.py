import copy
import glob
import os
import re
from typing import Any, Callable, Dict, Tuple, Union

import mne
import numpy as np

from ..base_dataset import BaseDataset


def _read_edf_header(file_path):
    f = open(file_path, "rb")
    header = f.read(88)
    f.close()
    return header


def _parse_age_and_gender_from_edf_header(file_path):
    '''
    from braindecode
    '''
    header = _read_edf_header(file_path)
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = header[8:].decode("ascii")
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    gender = "X"
    found_gender = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_gender) == 1:
        gender = found_gender[0]
    return {'age': age, 'gender': gender}


def _parse_metadata_from_path(file_path):
    # "edf": contains the raw edf data. Note that the channel configurations
    #     and sample frequency can vary.

    # "000": three-digit designator that simply organizes the
    #         corpus into groups of 100 patients

    # "aaaaaaaa": a randomized subject ID

    # "s001_2015_12_30": a session number in chronological order. Subject
    #         "aaaaaaab" had two sessions (s001 and s002). The date the session
    #         is included in the session number, though this is often
    #         not precise due to the way the data was originally collected (it
    #         is usually within 6 months of when the session actually occurred).

    # "01_tcp_ar": indicates which montage defenition these edf files are
    #             compatible with.

    # "aaaaaaaa_s001_t000.edf": the EEG signal data. "t000"
    #     refers to the first token generated from the EDF conversion
    #     process. "aaaaaaab/s003_2002_12_31" for example had 3 tokens
    #     generated (t000 to t002).

    # get the file name without the extension
    file_name = os.path.basename(file_path).split('.')[0]
    # get the subject id
    subject_id = file_name.split('_')[0]
    # get the session id
    session_id = file_name.split('_')[1]
    # get the token id
    token_id = file_name.split('_')[2]

    # get montage definition
    montage = file_path.split('/')[-2]
    # get the date
    date = file_path.split('/')[-3].split('_')[1]
    # get the digit
    digit = file_path.split('/')[-4]

    return {
        'subject_id': subject_id,
        'session_id': session_id,
        'token_id': token_id,
        'montage': montage,
        'date': date,
        'digit': digit
    }


class TUHTUEGDataset(BaseDataset):
    r'''
    The TUH EEG Corpus (TUEG): A rich archive of 26,846 clinical EEG recordings collected at Temple University Hospital (TUH) from 2002 - 2017. This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). Due to the large scale of the data set, this class does not support offline data preprocessing for the time being to avoid taking up too much hard disk space. The relevant information of the dataset is as follows:
    
    - Author: Harati et al.
    - Year: 2012
    - Download URL: https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    - Reference: Harati A, Lopez S, Obeid I, et al. The TUH EEG CORPUS: A big data resource for automated EEG interpretation[C]//2014 IEEE signal processing in medicine and biology symposium (SPMB). IEEE, 2014: 1-5.
    - Signals: Electroencephalogram (21 channels at 256Hz).
    - Information: Age, gender, date, digit, montage, subject id, session id, token id.
    
    In order to use this dataset, the download folder :obj:`edf` is required, containing the following files:
    
    - 000
    - 001
    - 002
    - ...
    - 150
    An example dataset for CNN-based methods:
    .. code-block:: python
    
        dataset = TUHTUEGDataset(io_path=f'./tuh_tueg',
                              root_path='./edf',
                              num_channel=21,
                              chunk_size=200,
                              overlap=0,
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('age')
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 21, 200])
    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:
    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = TUHTUEGDataset(io_path=f'./tuh_tueg',
                              root_path='./edf',
                              num_channel=21,
                              chunk_size=200,
                              overlap=0,
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('age')
                              ]),
                              num_worker=2)
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 21, 200])
    Args:
        root_path (str): Downloaded data files (default: :obj:`'./edf'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`200`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 21 channels are EEG signals. (default: :obj:`21`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/tuh_tueg`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)
    '''
    def __init__(self,
                 root_path: str = './edf',
                 chunk_size: int = 200,
                 overlap: int = 0,
                 num_channel: int = 21,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 io_path: str = './io/tuh_tueg',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):
        if before_trial is None:
            before_trial = self.default_before_trial
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
    def default_before_trial(raw: mne.io.Raw) -> mne.io.Raw:
        # pick channels
        if raw.ch_names[0].endswith('-REF'):
            # ar_ch_names, e.g., EEG A1-REF
            raw = raw.pick_channels([
                'EEG A1-REF', 'EEG A2-REF', 'EEG FP1-REF', 'EEG FP2-REF',
                'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',
                'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
                'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
                'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF',
                'EEG PZ-REF'
            ])
        else:
            # le_ch_names, e.g., EEG A1-LE
            raw = raw.pick_channels([
                'EEG A1-LE', 'EEG A2-LE', 'EEG FP1-LE', 'EEG FP2-LE',
                'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE',
                'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE',
                'EEG CZ-LE', 'EEG PZ-LE'
            ])

        # resample
        raw = raw.resample(200, npad='auto')

        # scale signals to micro volts (requires load)
        raw = raw.apply_function(lambda x: x * 1e6)

        # filter
        raw = raw.filter(1., 75.)

        # clip outlier values to +/- 800 micro volts
        raw = raw.apply_function(lambda x: np.clip(x, -800, 800))

        # # common average reference
        # raw = raw.set_eeg_reference(ref_channels='average', projection=True)
        # # Average reference projection was added, but has not been applied yet. Use the apply_proj method to apply it.
        # raw.apply_proj()
        return raw
    
    @staticmethod
    def process_record(file: Any = None,
                       num_channel: int = 21,
                       chunk_size: int = 200,
                       overlap: int = 0,
                       before_trial: Union[None, Callable] = None,
                       after_trial: Union[Callable, None] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        record_info_list = []

        # read info relevant for preprocessing from raw without loading it
        info = {'file_path': file}
        # parse age and gender information from EDF header
        info.update(_parse_age_and_gender_from_edf_header(file))

        # parse metadata from path
        info.update(_parse_metadata_from_path(file))

        raw = mne.io.read_raw_edf(file, preload=True)
        if not before_trial is None:
            raw = before_trial(raw)

        trial_length = len(raw)
        start_at = 0
        if chunk_size <= 0:
            chunk_size = trial_length - start_at

        # chunk with chunk size
        end_at = start_at + chunk_size
        # calculate moving step
        step = chunk_size - overlap

        write_pointer = 0

        file_name = os.path.basename(file).split('.')[0]
        trial_queue = []
        while end_at <= trial_length:
            clip_sample = raw.get_data(start=start_at, stop=end_at)
            clip_sample = clip_sample[:num_channel, :]

            if not offline_transform is None:
                t = offline_transform(eeg=clip_sample)
                t_eeg = t['eeg']

            record_info = copy.deepcopy(info)
            record_info['start_at'] = start_at
            record_info['end_at'] = end_at
            clip_id = f'{file_name}_{write_pointer}'
            record_info['clip_id'] = clip_id

            if not after_trial is None:
                trial_queue.append({
                    'eeg': t_eeg,
                    'key': clip_id,
                    'info': record_info
                })
            else:
                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

            start_at = start_at + step
            end_at = start_at + chunk_size
            write_pointer += 1

            record_info_list.append(record_info)

        if len(trial_queue) and after_trial:
            trial_queue = after_trial(trial_queue)
            for obj in trial_queue:
                assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
                yield obj

    def set_records(self, root_path: str = './edf', **kwargs):
        return list(glob.glob(os.path.join(root_path, '**/*.edf'), recursive=True))

    def __getitem__(self, index: int) -> Tuple:
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        baseline_index = str(info['baseline_id'])
        baseline = self.read_eeg(eeg_record, baseline_index)

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
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'io_path': self.io_path,
                'io_size': self.io_size,
                'io_mode': self.io_mode,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'in_memory': self.in_memory
            })