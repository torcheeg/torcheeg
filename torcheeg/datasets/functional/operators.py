import math
import os
import shutil
from multiprocessing import Manager
from typing import Any, Callable, Union

from joblib import Parallel, delayed
from tqdm import tqdm

from torcheeg.io import EEGSignalIO, MetaInfoIO


def _block(**kwargs):
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


def _io(block,
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


def from_existing_dataset(dataset,
                          io_path: str,
                          transform: Union[None, Callable] = None,
                          num_worker: int = 0,
                          verbose: bool = True,
                          io_size: int = 10485760,
                          io_mode: str = 'lmdb',
                          num_samples_per_worker: int = 100,
                          **kwargs):
    params = {
        'root_path': dataset.io_path,
        'io_path': io_path,
        'transform': transform,
        'num_worker': num_worker,
        'verbose': verbose,
        'io_size': io_size,
        'io_mode': io_mode,
        'num_samples_per_worker': num_samples_per_worker
    }
    meta_info_io_path = os.path.join(io_path, 'info.csv')
    eeg_signal_io_path = eeg_signal_io_path = os.path.join(io_path, 'eeg')

    exist_io = os.path.exists(meta_info_io_path) and os.path.exists(
        eeg_signal_io_path)

    if not exist_io:
        os.makedirs(io_path, exist_ok=True)

        # # init sub-folders
        shutil.copy(os.path.join(dataset.io_path, 'info.csv'),
                    os.path.join(io_path, 'info.csv'))
        EEGSignalIO(eeg_signal_io_path, io_size=io_size, io_mode=io_mode)

        if num_worker == 0:
            lock = MockLock()  # do nothing, just for compatibility
            for block in tqdm(_block(**params),
                              disable=not verbose,
                              desc="[PROCESS]"):
                _io(block=block, lock=lock, **params)
        else:
            # lock for lmdb writter, LMDB only allows single-process writes
            manager = Manager()
            lock = manager.Lock()

            Parallel(n_jobs=num_worker)(
                delayed(_io)(block=block, lock=lock, **params)
                for block in tqdm(
                    _block(**params), disable=not verbose, desc="[PROCESS]"))

    return type(dataset)(io_path=io_path,
                         offline_transform=transform,
                         num_worker=num_worker,
                         verbose=verbose,
                         io_size=io_size,
                         **kwargs)