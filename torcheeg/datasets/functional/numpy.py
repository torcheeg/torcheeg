import os
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, Dict, Tuple, Union

from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

import numpy as np

MAX_QUEUE_SIZE = 1024


def transform_producer(X_y_rank: Tuple[np.ndarray, Dict, int],
                       before_trial: Union[Callable,
                                           None], transform: Union[Callable,
                                                                   None],
                       after_trial: Union[Callable, None], queue: Queue):
    X, y, worker_rank = X_y_rank
    if before_trial:
        X = before_trial(X)

    trial_queue = []
    for write_pointer, clip_sample in enumerate(X):

        t_eeg = clip_sample

        if not transform is None:
            t = transform(eeg=clip_sample)
            t_eeg = t['eeg']

        clip_id = f'{worker_rank}_{write_pointer}'

        # record meta info for each signal
        record_info = {'clip_id': clip_id}
        record_info.update({k: v[write_pointer] for k, v in y.items()})
        if after_trial:
            trial_queue.append({
                'eeg': t_eeg,
                'key': clip_id,
                'info': record_info
            })
        else:
            queue.put({'eeg': t_eeg, 'key': clip_id, 'info': record_info})

    if len(trial_queue) and after_trial:
        trial_queue = after_trial(trial_queue)
        for obj in trial_queue:
            assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
            queue.put(obj)


def io_consumer(write_eeg_fn: Callable, write_info_fn: Callable, queue: Queue):
    while True:
        item = queue.get()
        if not item is None:
            eeg = item['eeg']
            key = item['key']
            write_eeg_fn(eeg, key)
            if 'info' in item:
                info = item['info']
                write_info_fn(info)
        else:
            break


class SingleProcessingQueue:
    def __init__(self, write_eeg_fn: Callable, write_info_fn: Callable):
        self.write_eeg_fn = write_eeg_fn
        self.write_info_fn = write_info_fn

    def put(self, item):
        eeg = item['eeg']
        key = item['key']
        self.write_eeg_fn(eeg, key)
        if 'info' in item:
            info = item['info']
            self.write_info_fn(info)


def numpy_constructor(
    X: np.ndarray,
    y: Dict,
    before_trial: Union[None, Callable] = None,
    transform: Union[None, Callable] = None,
    after_trial: Union[Callable, None] = None,
    num_samples_per_worker: int = 100,
    io_path: str = './io/numpy',
    io_size: int = 10485760,
    io_mode: str = 'lmdb',
    num_worker: int = 0,
    verbose: bool = True,
) -> None:

    for k, v in y.items():
        assert len(X) == len(
            v
        ), f'The number of labels {len(v)} does not match the number of samples {len(X)} for labels {k}.'

    # init IO
    meta_info_io_path = os.path.join(io_path, 'info.csv')
    eeg_signal_io_path = os.path.join(io_path, 'eeg')

    if os.path.exists(io_path) and not os.path.getsize(meta_info_io_path) == 0:
        print(
            f'The target folder already exists, if you need to regenerate the database IO, please delete the path {io_path}.'
        )
        return

    os.makedirs(io_path, exist_ok=True)

    info_io = MetaInfoIO(meta_info_io_path)
    eeg_io = EEGSignalIO(eeg_signal_io_path, io_size=io_size, io_mode=io_mode)

    # access the data
    indices = np.arange(len(X))
    worker_indices_list = np.array_split(indices,
                                         len(X) // num_samples_per_worker)

    X_y_rank_list = []
    for worker_rank, worker_indices in enumerate(worker_indices_list):
        X_worker = X[worker_indices]
        y_worker = {k: v[worker_indices] for k, v in y.items()}
        X_y_rank_list.append((X_worker, y_worker, worker_rank))

    if verbose:
        # show process bar
        pbar = tqdm(total=len(X_y_rank_list))
        pbar.set_description("[NUMPY]")

    if num_worker > 1:
        manager = Manager()
        queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
        io_consumer_process = Process(target=io_consumer,
                                      args=(eeg_io.write_eeg,
                                            info_io.write_info, queue),
                                      daemon=True)
        io_consumer_process.start()

        partial_mp_fn = partial(transform_producer,
                                before_trial=before_trial,
                                transform=transform,
                                after_trial=after_trial,
                                queue=queue)

        for _ in Pool(num_worker).imap(partial_mp_fn, X_y_rank_list):
            if verbose:
                pbar.update(1)

        queue.put(None)

        io_consumer_process.join()
        io_consumer_process.close()

    else:
        for X_y_rank in X_y_rank_list:
            transform_producer(X_y_rank=X_y_rank,
                               before_trial=before_trial,
                               transform=transform,
                               after_trial=after_trial,
                               queue=SingleProcessingQueue(
                                   eeg_io.write_eeg, info_io.write_info))
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
        print('Please wait for the writing process to complete...')
