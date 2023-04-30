import os
import re
from functools import partial
from typing import Tuple, Callable
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, Union
from pathlib import Path
import scipy.io as scio
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm
import mne
import numpy as np
mne.set_log_level(40)#ERROR
MAX_QUEUE_SIZE = 1024


def transform_producer(file_path_id: Tuple[str, int], read_file: Callable, chunk_size: int, overlap: int,
                       num_channel: int, before_trial: Union[None, Callable],
                       transform: Union[None, Callable],
                       after_trial: Union[Callable, None], queue: Queue):
    
    file_path = file_path_id[0]
    subject_id = file_path_id[1]
    file_label= file_path.split(os.sep)
    label, file_name  = file_label[-2], file_label[-1]
    trial_samples = read_file(file_path, chunk_size)
    events = [i[0] for i in trial_samples.events]
    events.append(events[-1] + np.diff(events)[0])# time interval between all events are same
    if before_trial:
        raise NotImplementedError
 
    trial_queue = []
    write_pointer = 0
    for i, trial_signal in enumerate(trial_samples.get_data()):
        t_eeg = trial_signal[:num_channel, :]
        if not transform is None:
            t = transform(eeg=trial_signal[:num_channel, :])
            t_eeg = t['eeg']

        clip_id = f'{file_name}_{write_pointer}'
        write_pointer += 1


        record_info = {
            'subject_id': subject_id,
            'trial_id': i,
            'file_id': file_name,
            'start_at': events[i],
            'end_at': events[i+1],
            'clip_id': clip_id,
            'label': label
        }

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


def folder_constructor(
    root_path: str = './eeg_raw_data',
    chunk_size: int = 800,
    overlap: int = 0,
    num_channel: int = 62,
    before_trial: Union[None, Callable] = None,
    transform: Union[None, Callable] = None,
    after_trial: Union[Callable, None] = None,
    io_path: str = './io/seed_iv',
    io_size: int = 10485760,
    io_mode: str = 'lmdb',
    num_worker: int = 0,
    verbose: bool = True,
    read_func =None
    
) -> None:
    # init IO

    meta_info_io_path = os.path.join(io_path, 'info.csv')
    eeg_signal_io_path = os.path.join(io_path, 'eeg')

    if os.path.exists(io_path) and not os.path.getsize(meta_info_io_path) == 0:
        print(
            f'The target folder already exists, if you need to regenerate the database IO, please delete the path {io_path}.'
        )
        return

    os.makedirs(io_path, exist_ok=True)

    meta_info_io_path = os.path.join(io_path, 'info.csv')
    eeg_signal_io_path = os.path.join(io_path, 'eeg')

    info_io = MetaInfoIO(meta_info_io_path)
    eeg_io = EEGSignalIO(eeg_signal_io_path, io_size=io_size, io_mode=io_mode)

    # loop to access the dataset files
    
    folder_path = Path(root_path)
    file_paths = folder_path.glob('**/*.*')
    file_path_list = [str(i).replace('\\','/') for i in file_paths]   
    subjects = list(range(len(file_path_list))) 
    file_path_list_subject = zip(file_path_list,subjects)
    if verbose:
        # show process bar
        pbar = tqdm(total=len(file_path_list))
        pbar.set_description("[Folder Data]")

    if num_worker < 0:
        num_worker = os.cpu_count() + num_worker +1

    if num_worker > 1:
        manager = Manager()
        queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
        io_consumer_process = Process(target=io_consumer,
                                      args=(eeg_io.write_eeg,
                                            info_io.write_info, queue),
                                      daemon=True)
        io_consumer_process.start()

        partial_mp_fn = partial(transform_producer,
                                chunk_size=chunk_size,
                                read_file=read_func,
                                overlap=overlap,
                                num_channel=num_channel,
                                before_trial=before_trial,
                                transform=transform,
                                after_trial=after_trial,
                                queue=queue)

        for _ in Pool(num_worker).imap(partial_mp_fn, file_path_list_subject):
            if verbose:
                pbar.update(1)

        queue.put(None)

        io_consumer_process.join()
        io_consumer_process.close()

    else:
        for file_path in file_path_list_subject:
            transform_producer(file_path_id=file_path,
                               chunk_size=chunk_size,
                               read_file=read_func,
                               overlap=overlap,
                               num_channel=num_channel,
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
