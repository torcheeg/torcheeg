import math
import os
import shutil
from multiprocessing import Manager
from typing import Any, Callable, Union

from joblib import Parallel, delayed
from tqdm import tqdm

from torcheeg.io import EEGSignalIO, MetaInfoIO


def _set_files(**kwargs):
    root_path = kwargs.pop('root_path', '.')  # str
    num_samples_per_worker = kwargs.pop('num_samples_per_worker', '.')  # str

    existing_meta_info_io_path = os.path.join(root_path, 'info.csv')
    existing_info_io = MetaInfoIO(existing_meta_info_io_path)
    df = existing_info_io.read_all()

    block_list = list()
    num_block_list = math.ceil(len(df) / num_samples_per_worker)
    for i in range(num_block_list):
        block_list.append(
            [i * num_samples_per_worker, (i + 1) * num_samples_per_worker])
    return block_list


def _load_data(block,
           io_path: str = None,
           io_size: int = 10485760,
           io_mode: str = 'lmdb',
           lock: Any = None,
           **kwargs):
    start_id, end_id = block

    transform = kwargs.pop('transform', None)
    root_path = kwargs.pop('root_path', '.')

    eeg_signal_io_path = os.path.join(io_path, 'eeg')

    existing_meta_info_io_path = os.path.join(io_path, 'info.csv')
    existing_eeg_signal_io_path = os.path.join(root_path, 'eeg')

    existing_info_io = MetaInfoIO(existing_meta_info_io_path)
    existing_eeg_io = EEGSignalIO(existing_eeg_signal_io_path,
                                  io_size=io_size,
                                  io_mode=io_mode)

    df = existing_info_io.read_all()
    chunk = df[start_id:end_id]

    eeg_io = EEGSignalIO(eeg_signal_io_path, io_size=io_size, io_mode=io_mode)

    last_baseline_id = None
    last_baseline_sample = None

    for i, row in chunk.iterrows():
        clip_id = row['clip_id']
        clip_sample = existing_eeg_io.read_eeg(clip_id)

        if 'baseline_id' in row:
            baseline_id = row['baseline_id']

            if last_baseline_id == baseline_id:
                trial_baseline_sample = last_baseline_sample
            else:
                trial_baseline_sample = existing_eeg_io.read_eeg(baseline_id)

            if transform is not None:
                t = transform(eeg=clip_sample, baseline=trial_baseline_sample)
            else:
                t = dict(eeg=clip_sample, baseline=trial_baseline_sample)

            t_eeg = t['eeg']
            t_baseline = t['baseline']

            if not last_baseline_id == baseline_id:
                with lock:
                    eeg_io.write_eeg(t_baseline, baseline_id)

                last_baseline_id = baseline_id
                last_baseline_sample = trial_baseline_sample

            with lock:
                eeg_io.write_eeg(t_eeg, clip_id)

        else:
            if transform is not None:
                t = transform(eeg=clip_sample)
            else:
                t = dict(eeg=clip_sample)
            t_eeg = t['eeg']

            with lock:
                eeg_io.write_eeg(t_eeg, clip_id)


class MockLock():
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def from_existing(dataset: Any,
                  io_path: str,
                  transform: Union[None, Callable] = None,
                  num_worker: int = 0,
                  verbose: bool = True,
                  io_size: int = 10485760,
                  io_mode: str = 'lmdb',
                  num_samples_per_worker: int = 100,
                  in_memory: bool = False,
                  **kwargs):
    '''
    To create new datasets from existing databases, which is often used where new transformations are expected to be applied to already processed datasets. 

    Args:
        dataset (Basedataset): The existing database, which can be obtained through :obj:`DEAPDataset`, :obj:`SEEDDataset`, and so on.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/deap`)
        transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. It is executed before generating IO intermediate results. (default: :obj:`None`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_samples_per_worker (int): The number of samples processed by each process. Once the specified number of samples are processed, the process will be destroyed and new processes will be created to perform new tasks. (default: :obj:`100`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)    
    '''
    root_path = dataset.io_path
    params = {
        'root_path': root_path,
        'io_path': io_path,
        'transform': transform,
        'num_worker': num_worker,
        'verbose': verbose,
        'io_size': io_size,
        'io_mode': io_mode,
        'num_samples_per_worker': num_samples_per_worker,
        'in_memory': in_memory,
    }
    params.update(kwargs)

    meta_info_io_path = os.path.join(io_path, 'info.csv')
    eeg_signal_io_path = eeg_signal_io_path = os.path.join(io_path, 'eeg')

    exist_io = os.path.exists(meta_info_io_path) and os.path.exists(
        eeg_signal_io_path)

    if not exist_io:
        print(
            f'dataset does not exist at path {io_path}, generating files to path...'
        )
        os.makedirs(io_path, exist_ok=True)

        # # init sub-folders
        shutil.copy(os.path.join(root_path, 'info.csv'),
                    os.path.join(io_path, 'info.csv'))
        EEGSignalIO(eeg_signal_io_path, io_size=io_size, io_mode=io_mode)

        if num_worker == 0:
            lock = MockLock()  # do nothing, just for compatibility
            for block in tqdm(_set_files(**params),
                              disable=not verbose,
                              desc="[PROCESS]"):
                _load_data(block=block, lock=lock, **params)
        else:
            # lock for lmdb writter, LMDB only allows single-process writes
            manager = Manager()
            lock = manager.Lock()

            Parallel(n_jobs=num_worker)(
                delayed(_load_data)(block=block, lock=lock, **params)
                for block in tqdm(
                    _set_files(**params), disable=not verbose, desc="[PROCESS]"))
    else:
        print(f'dataset already exists at path {io_path}, reading from path...')

    return type(dataset)(io_path=io_path,
                         offline_transform=transform,
                         num_worker=num_worker,
                         verbose=verbose,
                         io_size=io_size,
                         in_memory=in_memory,
                         **kwargs)
