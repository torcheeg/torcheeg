import logging
import os
import shutil
from typing import Any, Callable, Dict
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from torcheeg.io import EEGSignalIO, MetaInfoIO

MAX_QUEUE_SIZE = 1024

log = logging.getLogger(__name__)


class MockLock():
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class BaseDataset(Dataset):
    def __init__(self,
                 io_path: str = None,
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 in_memory: bool = False,
                 num_worker: int = 0,
                 verbose: bool = True,
                 after_trial: Callable = None,
                 after_session: Callable = None,
                 after_subject: Callable = None,
                 **kwargs):
        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = io_mode
        self.in_memory = in_memory
        self.num_worker = num_worker
        self.verbose = verbose
        self.after_trial = after_trial
        self.after_session = after_session
        self.after_subject = after_subject

        # new IO
        if not self.exist(self.io_path):
            print(
                f'dataset does not exist at path {self.io_path}, generating files to path...'
            )
            # make the root dictionary
            os.makedirs(self.io_path, exist_ok=True)

            records = self.set_records(**kwargs)
            if self.num_worker == 0:
                try:
                    for file_id, file in tqdm(enumerate(records),
                                              disable=not self.verbose,
                                              desc="[PROCESS]",
                                              total=len(records)):
                        self.save_record(io_path=self.io_path,
                                         io_size=self.io_size,
                                         io_mode=self.io_mode,
                                         file=file,
                                         file_id=file_id,
                                         process_record=self.process_record,
                                         **kwargs)
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e
            else:
                # catch the exception
                try:
                    Parallel(n_jobs=self.num_worker)(
                        delayed(self.save_record)(
                            io_path=io_path,
                            io_size=io_size,
                            io_mode=io_mode,
                            file_id=file_id,
                            file=file,
                            process_record=self.process_record,
                            **kwargs)
                        for file_id, file in tqdm(enumerate(records),
                                                  disable=not self.verbose,
                                                  desc="[PROCESS]",
                                                  total=len(records)))
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e

            # get the global io
            self.eeg_io_router, self.info = self.get_pointer(io_path=io_path,
                                                             io_size=io_size,
                                                             io_mode=io_mode)
            # catch the exception
            try:
                self.post_process_record(after_trial=after_trial,
                                         after_session=after_session,
                                         after_subject=after_subject)
            except Exception as e:
                # shutil to delete the database
                shutil.rmtree(self.io_path)
                raise e
        else:
            print(
                f'dataset already exists at path {self.io_path}, reading from path...'
            )
            # get the global io
            self.eeg_io_router, self.info = self.get_pointer(io_path=io_path,
                                                             io_size=io_size,
                                                             io_mode=io_mode)

    def set_records(self, **kwargs):
        '''
        The block method for generating the database. It is used to describe which data blocks need to be processed to generate the database. It is called in parallel by :obj:`joblib.Parallel` in :obj:`__init__` of the class.

        Args:
            lock (joblib.parallel.Lock): The lock for IO writter. (default: :obj:`None`)
            **kwargs: The arguments derived from __init__ of the class.

        .. code-block:: python

        def set_records(self, root_path: str = None, **kwargs):
                # e.g., return file name list for process_record to process
                return os.listdir(root_path)

        '''
        raise NotImplementedError(
            "Method set_records is not implemented in class BaseDataset")

    def get_pointer(self, io_path: str, io_size: int, io_mode: str):
        '''
        The method for initializing the database. It is used to describe how to initialize the database. It is called in :obj:`__init__` of the class.
        '''
        # get all records
        records = os.listdir(io_path)
        # filter the records with the prefix '_record_'
        records = list(filter(lambda x: '_record_' in x, records))

        # for every record, get the io_path, and init the info_io and eeg_io
        eeg_io_router = {}
        info_merged = None

        for record in records:
            meta_info_io_path = os.path.join(io_path, record, 'info.csv')
            eeg_signal_io_path = os.path.join(io_path, record, 'eeg')
            info_io = MetaInfoIO(meta_info_io_path)
            eeg_io = EEGSignalIO(eeg_signal_io_path,
                                 io_size=io_size,
                                 io_mode=io_mode)
            eeg_io_router[record] = eeg_io
            info_df = info_io.read_all()

            assert '_record_id' not in info_df.columns, \
                "column '_record_id' is a forbidden reserved word and is used to index the corresponding IO. Please replace your '_record_id' with another name."
            info_df['_record_id'] = record

            if info_merged is None:
                info_merged = info_df
            else:
                info_merged = pd.concat([info_merged, info_df],
                                        ignore_index=True)
        return eeg_io_router, info_merged

    @staticmethod
    def save_record(io_path: str = None,
                    io_size: int = 10485760,
                    io_mode: str = 'lmdb',
                    file: Any = None,
                    file_id: int = None,
                    process_record=None,
                    **kwargs):
        _record_id = str(file_id)
        meta_info_io_path = os.path.join(io_path, f'_record_{_record_id}',
                                         'info.csv')
        eeg_signal_io_path = os.path.join(io_path, f'_record_{_record_id}',
                                          'eeg')

        info_io = MetaInfoIO(meta_info_io_path)
        eeg_io = EEGSignalIO(eeg_signal_io_path,
                             io_size=io_size,
                             io_mode=io_mode)

        gen = process_record(file=file, **kwargs)
        # loop for data yield by process_record, until to the end of the data
        while True:
            try:
                # call process_record of the class
                # get the current class name
                obj = next(gen)

            except StopIteration:
                break

            if 'eeg' in obj and 'key' in obj:
                eeg_io.write_eeg(obj['eeg'], obj['key'])
            if 'info' in obj:
                info_io.write_info(obj['info'])

    @staticmethod
    def process_record(file: Any = None, **kwargs):
        '''
        The IO method for generating the database. It is used to describe how files are processed to generate the database. It is called in parallel by :obj:`joblib.Parallel` in :obj:`__init__` of the class.

        Args:
            file (Any): The file to be processed. It is an element in the list returned by set_records. (default: :obj:`Any`)
            **kwargs: The arguments derived from :obj:`__init__` of the class.

        .. code-block:: python

            def process_record(file: Any = None, chunk_size: int = 128, **kwargs):
                # process file
                eeg = np.ndarray((chunk_size, 64, 128), dtype=np.float32)
                key = '1'
                info = {
                    'subject': '1',
                    'session': '1',
                    'run': '1',
                    'label': '1'
                }
                yield {
                    'eeg': eeg,
                    'key': key,
                    'info': info
                }

        '''

        raise NotImplementedError(
            "Method process_record is not implemented in class BaseDataset")

    def post_process_record(self,
                            after_trial: Callable = None,
                            after_session: Callable = None,
                            after_subject: Callable = None):
        '''
        The hook method for post-processing the data. It is used to describe how to post-process the data.
        '''
        pbar = tqdm(total=len(self),
                    disable=not self.verbose,
                    desc="[POST-PROCESS]")

        # if all the hooks are None, then return
        if after_trial is None and after_session is None and after_subject is None:
            return

        if 'subject_id' in self.info.columns:
            subject_df = self.info.groupby('subject_id')
        else:
            subject_df = [(None, self.info)]
            if not after_subject is None:
                print(
                    "No subject_id column found in info, after_subject hook is ignored."
                )
        if after_subject is None:
            after_subject = lambda x: x

        for _, subject_info in subject_df:

            subject_record_list = []
            subject_index_list = []
            subject_samples = []

            # check if have a session_id column
            if 'session_id' in subject_info.columns:
                session_df = subject_info.groupby('session_id')
            else:
                session_df = [(None, subject_info)]
                if not after_session is None:
                    print(
                        "No session_id column found in info, after_session hook is ignored."
                    )
            if after_session is None:
                after_session = lambda x: x

            for _, session_info in session_df:

                # check if have a trial_id column
                if 'trial_id' in session_info.columns:
                    trial_df = session_info.groupby('trial_id')
                else:
                    trial_df = [(None, session_info)]
                    if not after_trial is None:
                        print(
                            "No trial_id column found in info, after_trial hook is ignored."
                        )
                if after_trial is None:
                    after_trial = lambda x: x

                session_samples = []
                for _, trial_info in trial_df:
                    trial_samples = []
                    for i in range(len(trial_info)):
                        eeg_index = str(trial_info.iloc[i]['clip_id'])
                        eeg_record = str(trial_info.iloc[i]['_record_id'])

                        # if i == 0 and 'baseline_id' in trial_info.columns:
                        #     baseline_index = str(
                        #         trial_info.iloc[i]['baseline_id'])
                        #     subject_record_list.append(eeg_record)
                        #     subject_index_list.append(baseline_index)

                        #     eeg = self.read_eeg(eeg_record, baseline_index)
                        #     trial_samples += [eeg]

                        subject_record_list.append(eeg_record)
                        subject_index_list.append(eeg_index)

                        eeg = self.read_eeg(eeg_record, eeg_index)
                        trial_samples += [eeg]

                        pbar.update(1)

                    trial_samples = self.hook_data_interface(
                        after_trial, trial_samples)
                    session_samples += trial_samples

                session_samples = self.hook_data_interface(
                    after_session, session_samples)
                subject_samples += session_samples

            subject_samples = self.hook_data_interface(after_subject,
                                                       subject_samples)

            # save the data
            for i, eeg in enumerate(subject_samples):
                eeg_index = str(subject_index_list[i])
                eeg_record = str(subject_record_list[i])

                self.write_eeg(eeg_record, eeg_index, eeg)

        pbar.close()

    @staticmethod
    def hook_data_interface(hook: Callable, data: Any):
        # like [np.random.randn(32, 128), np.random.randn(32, 128)]
        if isinstance(data[0], np.ndarray):
            data = np.stack(data, axis=0)
        elif isinstance(data[0], torch.Tensor):
            data = torch.stack(data, axis=0)
        # else list

        # shape like (2, 32, 128)
        data = hook(data)

        # back to list like [np.random.randn(32, 128), np.random.randn(32, 128)]
        if isinstance(data, np.ndarray):
            data = np.split(data, data.shape[0], axis=0)
            data = [np.squeeze(d, axis=0) for d in data]
        elif isinstance(data, torch.Tensor):
            data = torch.split(data, data.shape[0], dim=0)
            data = [torch.squeeze(d, axis=0) for d in data]
        # else list
        return data

    def read_eeg(self, record: str, key: str) -> Any:
        r'''
        Query the corresponding EEG signal in the EEGSignalIO according to the the given :obj:`key`. If :obj:`self.in_memory` is set to :obj:`True`, then EEGSignalIO will be read into memory and directly index the specified EEG signal in memory with the given :obj:`key` on subsequent reads.

        Args:
            record (str): The record id of the EEG signal to be queried.
            key (str): The index of the EEG signal to be queried.
            
        Returns:
            any: The EEG signal sample.
        '''
        eeg_io = self.eeg_io_router[record]
        if self.in_memory:
            return eeg_io.read_eeg_in_memory(key)
        return eeg_io.read_eeg(key)

    def write_eeg(self, record: str, key: str, eeg: Any):
        r'''
        Update the corresponding EEG signal in the EEGSignalIO according to the the given :obj:`key`.

        Args:
            record (str): The record id of the EEG signal to be queried.
            key (str): The index of the EEG signal to be queried.
            eeg (any): The EEG signal sample to be updated.
        '''
        eeg_io = self.eeg_io_router[record]
        if self.in_memory:
            eeg_io.write_eeg_in_memory(eeg=eeg, key=key)
        eeg_io.write_eeg(eeg=eeg, key=key)

    def read_info(self, index: int) -> Dict:
        r'''
        Query the corresponding meta information in the MetaInfoIO according to the the given :obj:`index`.

        In meta infomation, clip_id is required. Specifies the corresponding key of EEG in EEGSginalIO, which can be used to index EEG samples based on :obj:`self.read_eeg(key)`.

        Args:
            index (int): The index of the meta information to be queried.
            
        Returns:
            dict: The meta information.
        '''
        return self.info.iloc[index].to_dict()

    def exist(self, io_path: str) -> bool:
        '''
        Check if the database IO exists.

        Args:
            io_path (str): The path of the database IO.
        Returns:
            bool: True if the database IO exists, otherwise False.
        '''

        return os.path.exists(io_path)

    def __getitem__(self, index: int) -> any:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        return eeg, info

    def get_labels(self) -> list:
        '''
        Get the labels of the dataset.

        Returns:
            list: The list of labels.
        '''
        labels = []
        for i in range(len(self)):
            _, label = self.__getitem__(i)
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.info)

    def __copy__(self) -> 'BaseDataset':
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)

        result.eeg_io_router, result.info = self.get_pointer(
            io_path=result.io_path,
            io_size=result.io_size,
            io_mode=result.io_mode)
        return result

    @property
    def repr_body(self) -> Dict:
        return {
            'io_path': self.io_path,
            'io_size': self.io_size,
            'io_mode': self.io_mode,
            'in_memory': self.in_memory
        }

    @property
    def repr_tail(self) -> Dict:
        return {'length': self.__len__()}

    def __repr__(self) -> str:
        # init info
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            # line end
            if i:
                format_string += ','
            format_string += '\n'
            # str param
            if isinstance(v, str):
                format_string += f"    {k}='{v}'"
            else:
                format_string += f"    {k}={v}"
        format_string += '\n)'
        # other info
        format_string += '\n'
        for i, (k, v) in enumerate(self.repr_tail.items()):
            if i:
                format_string += ', '
            format_string += f'{k}={v}'
        return format_string
