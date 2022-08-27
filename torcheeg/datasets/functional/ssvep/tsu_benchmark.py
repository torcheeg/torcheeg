import os
import re
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue, set_start_method
from typing import Callable, List, Union

import scipy.io as scio
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024


def transform_producer(file_name: str, root_path: str, chunk_size: int,
                       overlap: int, num_channel: int,
                       transform: Union[Callable, None],
                       write_info_fn: Callable, queue: Queue):
    subject = int(re.findall(r'S(\d*).mat', file_name)[0])  # subject (35)
    freq_phase = scio.loadmat(os.path.join(root_path, 'Freq_Phase.mat'))
    freqs = freq_phase['freqs'][0]
    phases = freq_phase['phases'][0]

    samples = scio.loadmat(os.path.join(root_path,
                                        file_name))['data'].transpose(
                                            2, 3, 0, 1)
    # 40, 6, 64, 1500
    # Target number: 40
    # Block number: 6
    # Electrode number: 64
    # Time points: 1500

    step = chunk_size - overlap
    write_pointer = 0

    for trial_id in range(samples.shape[0]):
        trial_meta_info = {
            'subject_id': subject,
            'trial_id': trial_id,
            'phases': phases[trial_id],
            'freqs': freqs[trial_id]
        }
        trial_samples = samples[trial_id]

        for block_id in range(trial_samples.shape[0]):
            block_meta_info = {'block_id': block_id}
            block_meta_info.update(trial_meta_info)
            block_samples = trial_samples[block_id]

            start_at = 0
            end_at = chunk_size

            while end_at <= block_samples.shape[1]:
                clip_sample = block_samples[:num_channel, start_at:end_at]

                t_eeg = clip_sample
                if not transform is None:
                    t_eeg = transform(eeg=clip_sample)['eeg']

                clip_id = f'{file_name}_{write_pointer}'

                queue.put({'eeg': t_eeg, 'key': clip_id})
                write_pointer += 1

                # record meta info for each signal
                record_info = {
                    'start_at': start_at,
                    'end_at': end_at,
                    'clip_id': clip_id
                }
                record_info.update(block_meta_info)
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


def tsu_benchmark_constructor(
        root_path: str = './TSUBenchmark',
        chunk_size: int = 200,
        overlap: int = 0,
        num_channel: int = 64,
        transform: Union[None, Callable] = None,
        io_path: str = './io/tsu_benchmark',
        num_worker: int = 0,
        verbose: bool = True,
        cache_size: int = 64 * 1024 * 1024 * 1024) -> None:
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
    eeg_io = EEGSignalIO(eeg_signal_io_path, cache_size=cache_size)

    # loop to access the dataset files
    file_list = os.listdir(root_path)
    skip_set = [
        'Readme.txt', 'Sub_info.txt', '64-channels.loc', '64-channels.loc',
        'Freq_Phase.mat'
    ]
    file_list = [f for f in file_list if f not in skip_set]

    if verbose:
        # show process bar
        pbar = tqdm(total=len(file_list))
        pbar.set_description("[TSU Benchmark]")

    if num_worker > 1:
        manager = Manager()
        queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
        io_consumer_process = Process(target=io_consumer,
                                      args=(eeg_io.write_eeg, queue),
                                      daemon=True)
        io_consumer_process.start()

        partial_mp_fn = partial(transform_producer,
                                root_path=root_path,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                num_channel=num_channel,
                                transform=transform,
                                write_info_fn=info_io.write_info,
                                queue=queue)

        for _ in Pool(num_worker).imap(partial_mp_fn, file_list):
            if verbose:
                pbar.update(1)

        queue.put(None)

        io_consumer_process.join()
        io_consumer_process.close()

    else:
        for file_name in file_list:
            transform_producer(file_name=file_name,
                               root_path=root_path,
                               chunk_size=chunk_size,
                               overlap=overlap,
                               num_channel=num_channel,
                               transform=transform,
                               write_info_fn=info_io.write_info,
                               queue=SingleProcessingQueue(eeg_io.write_eeg))
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
        print('Please wait for the writing process to complete...')
