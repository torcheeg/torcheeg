import logging
import os
import shutil
from typing import Any, Callable, Dict, Union
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from torcheeg.io import EEGSignalIO, MetaInfoIO

log = logging.getLogger('torcheeg')


class BaseDataset(Dataset):

    def __init__(self,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 after_trial: Callable = None,
                 after_session: Callable = None,
                 after_subject: Callable = None,
                 **kwargs):
        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = io_mode
        self.num_worker = num_worker
        self.verbose = verbose
        self.after_trial = after_trial
        self.after_session = after_session
        self.after_subject = after_subject

        # new IO
        if not self.exist(self.io_path) or self.io_mode == 'memory':
            log.info(
                f'🔍 | Processing EEG data. Processed EEG data has been cached to \033[92m{io_path}\033[0m.'
            )
            log.info(
                f'⏳ | Monitoring the detailed processing of a record for debugging. The processing of other records will only be reported in percentage to keep it clean.'
            )
            # make the root dictionary
            os.makedirs(self.io_path, exist_ok=True)

            records = self.set_records(**kwargs)
            if self.num_worker == 0:
                try:
                    worker_results = []
                    for file_id, file in tqdm(enumerate(records),
                                              disable=not self.verbose,
                                              desc="[PROCESS]",
                                              total=len(records),
                                              position=0,
                                              leave=None):
                        worker_results.append(
                            self.save_record(io_path=self.io_path,
                                             io_size=self.io_size,
                                             io_mode=self.io_mode,
                                             file=file,
                                             file_id=file_id,
                                             process_record=self.process_record,
                                             verbose=self.verbose,
                                             **kwargs))
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e
            else:
                # catch the exception
                try:
                    worker_results = Parallel(n_jobs=self.num_worker)(
                        delayed(self.save_record)(
                            io_path=io_path,
                            io_size=io_size,
                            io_mode=io_mode,
                            file_id=file_id,
                            file=file,
                            process_record=self.process_record,
                            verbose=self.verbose,
                            **kwargs)
                        for file_id, file in tqdm(enumerate(records),
                                                  disable=not self.verbose,
                                                  desc="[PROCESS]",
                                                  total=len(records),
                                                  position=0,
                                                  leave=None))
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e

            if not self.io_mode == 'memory':
                log.info(
                    f'✅ | All processed EEG data has been cached to {io_path}.'
                )
                log.info(
                    f'😊 | Please set \033[92mio_path\033[0m to \033[92m{io_path}\033[0m for the next run, to directly read from the cache if you wish to skip the data processing step.'
                )

            eeg_io_router = {}
            info_merged = []

            for worker_result in worker_results:
                worker_eeg_io = worker_result['eeg_io']
                worker_info_io = worker_result['info_io']
                worker_record = worker_result['record']

                eeg_io_router[worker_record] = worker_eeg_io
                worker_info = worker_info_io.read_all()

                assert '_record_id' not in worker_info.columns, \
                    "column '_record_id' is a forbidden reserved word and is used to index the corresponding IO. Please replace your '_record_id' with another name."
                worker_info['_record_id'] = worker_record

                info_merged.append(worker_info)

            self.eeg_io_router = eeg_io_router
            self.info = pd.concat(info_merged, ignore_index=True)

            if self.after_trial is not None or self.after_session is not None or self.after_subject is not None:
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
            log.info(
                f'🔍 | Detected cached processing results, reading cache from {self.io_path}.'
            )
            # get all records
            records = os.listdir(io_path)
            # filter the records with the prefix '_record_'
            records = list(filter(lambda x: '_record_' in x, records))
            # sort the records
            records = sorted(records, key=lambda x: int(x.split('_')[2]))

            # for every record, get the io_path, and init the info_io and eeg_io
            eeg_io_router = {}
            info_merged = []

            assert len(
                records
            ) > 0, "The io_path, {}, is corrupted. Please delete this folder and try again.".format(
                io_path)

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

                info_merged.append(info_df)

            self.eeg_io_router = eeg_io_router
            self.info = pd.concat(info_merged, ignore_index=True)

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

    @staticmethod
    def save_record(io_path: Union[None, str] = None,
                    io_size: int = 1048576,
                    io_mode: str = 'lmdb',
                    file: Any = None,
                    file_id: int = None,
                    process_record: Callable = None,
                    verbose: bool = True,
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

        if file_id == 0:
            pbar = tqdm(disable=not verbose,
                        desc=f"[RECORD {file}]",
                        position=1,
                        leave=None)

            # pbar.write(
            #     "Monitoring the detailed processing of a record. The detailed processing of other records will not be reported to keep it clean."
            # )

        # loop for data yield by process_record, until to the end of the data
        while True:
            try:
                # call process_record of the class
                # get the current class name
                obj = next(gen)

                if file_id == 0:
                    pbar.update(1)

            except StopIteration:
                break

            if 'eeg' in obj and 'key' in obj:
                eeg_io.write_eeg(obj['eeg'], obj['key'])
            if 'info' in obj:
                info_io.write_info(obj['info'])

        if file_id == 0:
            pbar.close()

        return {
            'eeg_io': eeg_io,
            'info_io': info_io,
            'record': f'_record_{_record_id}'
        }

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
            # if not after_subject is None:
            #     log.info(
            #         "No subject_id column found in info, after_subject hook is ignored."
            #     )
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
                # if not after_session is None:
                #     log.info(
                #         "No session_id column found in info, after_session hook is ignored."
                #     )
            if after_session is None:
                after_session = lambda x: x

            for _, session_info in session_df:

                # check if have a trial_id column
                if 'trial_id' in session_info.columns:
                    trial_df = session_info.groupby('trial_id')
                else:
                    trial_df = [(None, session_info)]
                    if not after_trial is None:
                        log.info(
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
        Query the corresponding EEG signal in the EEGSignalIO according to the the given :obj:`key`.

        Args:
            record (str): The record id of the EEG signal to be queried.
            key (str): The index of the EEG signal to be queried.
            
        Returns:
            any: The EEG signal sample.
        '''
        eeg_io = self.eeg_io_router[record]
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
        result.__dict__.update({
            k: v
            for k, v in self.__dict__.items()
            if k not in ['eeg_io_router', 'info']
        })

        # shallow copy data
        result.eeg_io_router = {}
        for record, eeg_io in self.eeg_io_router.items():
            result.eeg_io_router[record] = eeg_io.__copy__()
        # deep copy info (for further modification)
        result.info = self.info

        return result

    @property
    def repr_body(self) -> Dict:
        return {
            'io_path': self.io_path,
            'io_size': self.io_size,
            'io_mode': self.io_mode
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
