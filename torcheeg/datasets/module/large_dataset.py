import logging
import os
import shutil
from typing import Any, Dict

import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from torcheeg.io import MetaInfoIO

MAX_QUEUE_SIZE = 1024

log = logging.getLogger(__name__)


class MockLock():
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class LargeDataset(Dataset):
    def __init__(self,
                 io_path: str = None,
                 num_worker: int = 0,
                 verbose: bool = True,
                 **kwargs):

        self.io_path = io_path
        self.num_worker = num_worker
        self.verbose = verbose

        # new IO
        if not self.exist(self.io_path):
            print(
                f'dataset does not exist at path {self.io_path}, generating files to path...'
            )
            # make the root dictionary
            os.makedirs(self.io_path, exist_ok=True)

            if self.num_worker == 0:
                try:
                    for file in tqdm(self.set_records(**kwargs),
                                     disable=not self.verbose,
                                     desc="[PROCESS]"):
                        self.save_record(io_path=self.io_path,
                                         file=file,
                                         process_record=self.process_record,
                                         **kwargs)
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e
            else:
                # catch the exception
                try:
                    Parallel(n_jobs=self.num_worker)(delayed(self.save_record)(
                        io_path=io_path,
                        file=file,
                        process_record=self.process_record,
                        **kwargs) for file in tqdm(self.set_records(**kwargs),
                                                   disable=not self.verbose,
                                                   desc="[PROCESS]"))
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e

        print(
            f'dataset already exists at path {self.io_path}, reading from path...'
        )

        # get the global io
        self.info = self.get_pointer(io_path=io_path)

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
            "Method set_records is not implemented in class LargeDataset")

    def get_pointer(self, io_path: str):
        '''
        The method for initializing the database. It is used to describe how to initialize the database. It is called in :obj:`__init__` of the class.
        '''
        # get all records
        records = os.listdir(io_path)
        # filter the records with the prefix '_record_'
        records = list(filter(lambda x: '_record_' in x, records))

        info_merged = []
        for record in records:
            meta_info_io_path = os.path.join(io_path, record, 'info.csv')
            info_io = MetaInfoIO(meta_info_io_path)
            info_df = info_io.read_all()

            assert '_record_id' not in info_df.columns, \
                "column '_record_id' is a forbidden reserved word and is used to index the corresponding IO. Please replace your '_record_id' with another name."
            info_df['_record_id'] = record

            info_merged.append(info_df)

        info_merged = pd.concat(info_merged, ignore_index=True)
        return info_merged

    @staticmethod
    def save_record(io_path: str = None,
                    file: Any = None,
                    process_record=None,
                    **kwargs):

        meta_info_io_path = os.path.join(io_path, f'_record_{str(file)}',
                                         'info.csv')
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
            "Method process_record is not implemented in class LargeDataset")

    def read_eeg(self, record: str) -> Any:
        r'''
        Query the corresponding EEG signal in the EEGSignalIO according to the the given :obj:`key`. If :obj:`self.in_memory` is set to :obj:`True`, then EEGSignalIO will be read into memory and directly index the specified EEG signal in memory with the given :obj:`key` on subsequent reads.

        Args:
            record (str): The record id of the EEG signal to be queried.
            
        Returns:
            any: The EEG signal sample.
        '''
        raise NotImplementedError(
            "Method process_record is not implemented in class LargeDataset")

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

        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record)

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

    def __copy__(self) -> 'LargeDataset':
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)

        result.info = self.get_pointer(io_path=result.io_path)
        return result

    @property
    def repr_body(self) -> Dict:
        return {'io_path': self.io_path}

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
