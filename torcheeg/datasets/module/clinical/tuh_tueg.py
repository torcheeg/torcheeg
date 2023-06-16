import copy
import glob
import os
import re
import shutil
from multiprocessing import Manager
from typing import Any, Callable, Dict, Tuple, Union

import mne
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from torcheeg.io import MetaInfoIO

from ..base_dataset import BaseDataset


class MockLock():
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


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
                              chunk_size=1000,
                              overlap=1000,
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('age')
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 21, 1000])

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = TUHTUEGDataset(io_path=f'./tuh_tueg',
                              root_path='./edf',
                              num_channel=21,
                              chunk_size=1000,
                              overlap=1000,
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('age')
                              ]),
                              num_worker=2)
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 21, 1000])

    Args:
        root_path (str): Downloaded data files (default: :obj:`'./edf'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. (default: :obj:`1000`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 21 channels are EEG signals. (default: :obj:`21`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/tuh_tueg`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    '''
    def __init__(self,
                 root_path: str = './edf',
                 chunk_size: int = 1000,
                 overlap: int = 0,
                 num_channel: int = 21,
                 online_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/tuh_tueg',
                 num_worker: int = 0,
                 verbose: bool = True):
        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'online_transform': online_transform,
            'label_transform': label_transform,
            'io_path': io_path,
            'num_worker': num_worker,
            'verbose': verbose
        }
        # save all arguments to __dict__
        self.__dict__.update(params)

        # new IO
        if not self.exist(io_path):
            print(
                f'dataset does not exist at path {io_path}, generating files to path...'
            )
            # make the root dictionary
            os.makedirs(io_path, exist_ok=True)

            # init sub-folders
            meta_info_io_path = os.path.join(io_path, 'info.csv')
            MetaInfoIO(meta_info_io_path)

            if num_worker == 0:
                lock = MockLock()  # do nothing, just for compatibility
                # if catch error, then delete the database
                try:
                    for file in tqdm(self.set_records(**params),
                                     disable=not verbose,
                                     desc="[PROCESS]"):
                        self.save_record(file=file,
                                           lock=lock,
                                           process_record=self.process_record,
                                           **params)
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e
            else:
                # lock for lmdb writter, LMDB only allows single-process writes
                manager = Manager()
                lock = manager.Lock()
                # if catch error, then delete the database
                try:
                    Parallel(n_jobs=self.num_worker)(
                        delayed(self.save_record)(file=file,
                                                    lock=lock,
                                                    process_record=self.process_record,
                                                    **params)
                        for file in tqdm(self.set_records(**params),
                                         disable=not verbose,
                                         desc="[PROCESS]"))
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(io_path)
                    raise e

        print(f'dataset already exists at path {io_path}, reading from path...')

        meta_info_io_path = os.path.join(io_path, 'info.csv')
        info_io = MetaInfoIO(meta_info_io_path)
        self.info = info_io.read_all()

    @staticmethod
    def save_record(io_path: str = None,
                      file: Any = None,
                      lock: Any = None,
                      process_record=None,
                      **kwargs):

        meta_info_io_path = os.path.join(io_path, 'info.csv')
        info_io = MetaInfoIO(meta_info_io_path)

        gen = process_record(file=file, **kwargs)
        # loop for data yield by process_record, until to the end of the data
        while True:
            try:
                # call process_record of the class
                # get the current class name
                obj = next(gen)

            except StopIteration:
                break

            with lock:
                if 'info' in obj:
                    info_io.write_info(obj['info'])

    @staticmethod
    def process_record(file_path: Any = None,
                   chunk_size: int = 1000,
                   overlap: int = 0,
                   **kwargs):

        record_info_list = []

        # read info relevant for preprocessing from raw without loading it
        info = {'file_path': file_path}
        # parse age and gender information from EDF header
        info.update(_parse_age_and_gender_from_edf_header(file_path))

        # parse metadata from path
        info.update(_parse_metadata_from_path(file_path))

        raw = mne.io.read_raw_edf(file_path, preload=False)

        trial_length = len(raw)
        start_at = 0
        if chunk_size <= 0:
            chunk_size = trial_length - start_at

        # chunk with chunk size
        end_at = start_at + chunk_size
        # calculate moving step
        step = chunk_size - overlap

        write_pointer = 0

        file_name = os.path.basename(file_path).split('.')[0]
        while end_at <= trial_length:
            record_info = copy.deepcopy(info)
            record_info['start_at'] = start_at
            record_info['end_at'] = end_at
            record_info['clip_id'] = f'{file_name}_{write_pointer}'

            start_at = start_at + step
            end_at = start_at + chunk_size
            write_pointer += 1

            record_info_list.append(record_info)

        return record_info_list

    def set_records(self, root_path: str = './edf', **kwargs):
        return glob.glob(root_path, recursive=True)

    def read_eeg(self, file_path: str, start_at: int,
                 end_at: int) -> np.ndarray:
        r'''
        Query the corresponding EEG signal in the file with :obj:`mne`.

        Args:
            file_path (str): The path of the file.
            start_at (int): The start index of the EEG signal.
            end_at (int): The end index of the EEG signal.
            
        Returns:
            np.ndarray: The EEG signal sample.
        '''
        raw = mne.io.read_raw_edf(file_path, preload=False)
        return raw.get_data()[:self.num_channels, start_at:end_at]

    def __getitem__(self, index: int) -> Tuple:
        info = self.read_info(index)

        signal = self.read_eeg(file_path=info['file_path'],
                               start_at=info['start_at'],
                               end_at=info['end_at'])
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=signal)['eeg']

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
                'label_transform': self.label_transform,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
