import os
import shutil
from multiprocessing import Manager
from typing import Any, Dict

from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from torcheeg.io import EEGSignalIO, MetaInfoIO

MAX_QUEUE_SIZE = 1024


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
                 **kwargs):

        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = io_mode
        self.in_memory = in_memory
        self.num_worker = num_worker
        self.verbose = verbose

        # new IO
        if not self.exist(self.io_path):
            print(
                f'dataset does not exist at path {self.io_path}, generating files to path...'
            )
            # make the root dictionary
            os.makedirs(self.io_path, exist_ok=True)

            # init sub-folders
            meta_info_io_path = os.path.join(self.io_path, 'info.csv')
            eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

            MetaInfoIO(meta_info_io_path)
            EEGSignalIO(eeg_signal_io_path,
                        io_size=self.io_size,
                        io_mode=self.io_mode)

            if self.num_worker == 0:
                lock = MockLock()  # do nothing, just for compatibility
                # if catch error, then delete the database
                try:
                    for file in tqdm(self._set_files(**kwargs),
                                    disable=not self.verbose,
                                    desc="[PROCESS]"):
                        self._process_file(io_path=self.io_path,
                                        io_size=self.io_size,
                                        io_mode=self.io_mode,
                                        file=file,
                                        lock=lock,
                                        _load_data=self._load_data,
                                        **kwargs)
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
                        delayed(self._process_file)(io_path=io_path,
                                                    io_size=io_size,
                                                    io_mode=io_mode,
                                                    file=file,
                                                    lock=lock,
                                                    _load_data=self._load_data,
                                                    **
                                                    kwargs)
                        for file in tqdm(self._set_files(**kwargs),
                                        disable=not self.verbose,
                                        desc="[PROCESS]"))
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e

        print(
            f'dataset already exists at path {self.io_path}, reading from path...'
        )

        meta_info_io_path = os.path.join(self.io_path, 'info.csv')
        eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

        info_io = MetaInfoIO(meta_info_io_path)
        self.eeg_io = EEGSignalIO(eeg_signal_io_path,
                                  io_size=self.io_size,
                                  io_mode=self.io_mode)

        self.info = info_io.read_all()

    @staticmethod
    def _set_files(**kwargs):
        '''
        The block method for generating the database. It is used to describe which data blocks need to be processed to generate the database. It is called in parallel by :obj:`joblib.Parallel` in :obj:`__init__` of the class.

        Args:
            lock (joblib.parallel.Lock): The lock for IO writter. (default: :obj:`None`)
            **kwargs: The arguments derived from __init__ of the class.

        .. code-block:: python

            def _set_files(root_path: str = None, **kwargs):
                # e.g., return file name list for _load_data to process
                return os.listdir(root_path)

        '''
        raise NotImplementedError(
            "Method _set_files is not implemented in class BaseDataset")

    @staticmethod
    def _process_file(io_path: str = None,
                      io_size: int = 10485760,
                      io_mode: str = 'lmdb',
                      file: Any = None,
                      lock: Any = None,
                      _load_data=None,
                      **kwargs):

        meta_info_io_path = os.path.join(io_path, 'info.csv')
        eeg_signal_io_path = os.path.join(io_path, 'eeg')

        info_io = MetaInfoIO(meta_info_io_path)
        eeg_io = EEGSignalIO(eeg_signal_io_path,
                             io_size=io_size,
                             io_mode=io_mode)

        gen = _load_data(file=file, **kwargs)
        # loop for data yield by _load_data, until to the end of the data
        while True:
            try:
                # call _load_data of the class
                # get the current class name
                obj = next(gen)

            except StopIteration:
                break

            with lock:
                if 'eeg' in obj and 'key' in obj:
                    eeg_io.write_eeg(obj['eeg'], obj['key'])
                if 'info' in obj:
                    info_io.write_info(obj['info'])

    @staticmethod
    def _load_data(file: Any = None, **kwargs):
        '''
        The IO method for generating the database. It is used to describe how files are processed to generate the database. It is called in parallel by :obj:`joblib.Parallel` in :obj:`__init__` of the class.

        Args:
            file (Any): The file to be processed. It is an element in the list returned by _set_files. (default: :obj:`Any`)
            **kwargs: The arguments derived from :obj:`__init__` of the class.

        .. code-block:: python

            def _load_data(file: Any = None, chunk_size: int = 128, **kwargs):
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
            "Method _load_data is not implemented in class BaseDataset")

    def read_eeg(self, key: str) -> Any:
        r'''
        Query the corresponding EEG signal in the EEGSignalIO according to the the given :obj:`key`. If :obj:`self.in_memory` is set to :obj:`True`, then EEGSignalIO will be read into memory and directly index the specified EEG signal in memory with the given :obj:`key` on subsequent reads.

        Args:
            key (str): The index of the EEG signal to be queried.
            
        Returns:
            any: The EEG signal sample.
        '''
        if self.in_memory:
            return self.eeg_io.read_eeg_in_memory(key)
        return self.eeg_io.read_eeg(key)

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
        meta_info_io_path = os.path.join(io_path, 'info.csv')
        eeg_signal_io_path = eeg_signal_io_path = os.path.join(io_path, 'eeg')

        return os.path.exists(meta_info_io_path) and os.path.exists(
            eeg_signal_io_path)

    def __getitem__(self, index: int) -> any:
        raise NotImplementedError(
            "Method __getitem__ is not implemented in class " +
            self.__class__.__name__)

    def __len__(self):
        return len(self.info)

    def __copy__(self) -> 'BaseDataset':
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

        result.eeg_io = EEGSignalIO(eeg_signal_io_path,
                                    io_size=self.io_size,
                                    io_mode=self.io_mode)
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
