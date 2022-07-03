import os
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, List, Union

import numpy as np
import pandas as pd
import scipy.io as scio
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024


def transform_producer(df: pd.DataFrame, root_path: str, chunk_size: int,
                       overlap: int, channel_num: int,
                       transform: Union[List[Callable], Callable, None],
                       write_info_fn: Callable, queue: Queue):

    # calculate moving step
    step = chunk_size - overlap
    write_pointer = 0

    start_epoch = None

    for _, epoch_info in df.iterrows():
        epoch_meta_info = {
            'epoch_id': epoch_info['EpochID'],
            'subject_id': epoch_info['SubjectID'],
            'session': epoch_info['Session'],
            'task': epoch_info['Task'],
            'usage': epoch_info['Usage'],
        }

        epoch_id = epoch_meta_info['epoch_id']

        if start_epoch is None:
            start_epoch = epoch_id

        samples = scio.loadmat(os.path.join(root_path, epoch_id))['epoch_data']

        # extract experimental signals
        start_at = 0
        end_at = chunk_size
        while end_at <= samples.shape[1]:
            clip_sample = samples[:channel_num, start_at:end_at]
            t_eeg = clip_sample

            if not transform is None:
                t_eeg = transform(eeg=clip_sample)['eeg']

            clip_id = f'after{start_epoch}_{write_pointer}'

            queue.put({'eeg': t_eeg, 'key': clip_id})
            write_pointer += 1

            # record meta info for each signal
            record_info = {
                'start_at': start_at,
                'end_at': end_at,
                'clip_id': clip_id
            }

            record_info.update(epoch_meta_info)
            write_info_fn(record_info)
            start_at = start_at + step
            end_at = start_at + chunk_size


def io_consumer(write_eeg_fn, queue):
    while True:
        item = queue.get()
        if not item is None:
            eeg = item['eeg']
            key = item['key']
            write_eeg_fn(eeg, key)
        else:
            break


class SingleProcessingQueue:
    def __init__(self, write_eeg_fn):
        self.write_eeg_fn = write_eeg_fn

    def put(self, item):
        eeg = item['eeg']
        key = item['key']
        self.write_eeg_fn(eeg, key)


def m3cv_constructor(root_path: str = './aistudio',
                     subset: str = 'Enrollment',
                     chunk_size: int = 1000,
                     overlap: int = 0,
                     channel_num: int = 65,
                     transform: Union[None, Callable] = None,
                     io_path: str = './io/m3cv',
                     num_worker: int = 0,
                     verbose: bool = True,
                     cache_size: int = 1024 * 1024 * 1024 * 1024) -> None:
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
    eeg_io = EEGSignalIO(eeg_signal_io_path, cache_size=cache_size)

    # loop to access the dataset files
    df = pd.read_csv(os.path.join(root_path, f'{subset}_Info.csv'))
    df_list_num = len(df) // 60
    df_list = np.array_split(df, df_list_num)

    if verbose:
        # show process bar
        pbar = tqdm(total=len(df_list))
        pbar.set_description("[M3CV]")

    if num_worker > 1:
        manager = Manager()
        queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
        io_consumer_process = Process(target=io_consumer,
                                      args=(eeg_io.write_eeg, queue),
                                      daemon=True)
        io_consumer_process.start()

        partial_mp_fn = partial(transform_producer,
                                root_path=os.path.join(root_path, subset),
                                chunk_size=chunk_size,
                                overlap=overlap,
                                channel_num=channel_num,
                                transform=transform,
                                write_info_fn=info_io.write_info,
                                queue=queue)

        for _ in Pool(num_worker).imap(partial_mp_fn, df_list):
            if verbose:
                pbar.update(1)

        queue.put(None)

        io_consumer_process.join()
        io_consumer_process.close()
    else:
        for df in df_list:
            transform_producer(df=df,
                               root_path=os.path.join(root_path, subset),
                               chunk_size=chunk_size,
                               overlap=overlap,
                               channel_num=channel_num,
                               transform=transform,
                               write_info_fn=info_io.write_info,
                               queue=SingleProcessingQueue(eeg_io.write_eeg))
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
        print('Please wait for the writing process to complete...')
