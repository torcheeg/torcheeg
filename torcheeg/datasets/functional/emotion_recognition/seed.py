import os
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, List, Union

import scipy.io as scio
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1099511627776


def transform_producer(file_name: str, root_path: str, chunk_size: int,
                       overlap: int, channel_num: int,
                       transform: Union[List[Callable], Callable, None],
                       write_info_fn: Callable, queue: Queue):
    subject = int(
        os.path.basename(file_name).split('.')[0].split('_')[0])  # subject (15)
    date = int(
        os.path.basename(file_name).split('.')[0].split('_')[1])  # period (3)

    samples = scio.loadmat(os.path.join(root_path, file_name),
                           verify_compressed_data_integrity=False
                           )  # trail (15), channel(62), timestep(n*200)
    # label file
    labels = scio.loadmat(os.path.join(root_path, 'label.mat'),
                          verify_compressed_data_integrity=False)['label'][0]

    trail_ids = [key for key in samples.keys() if 'eeg' in key]

    # calculate moving step
    step = chunk_size - overlap

    # prepare transform
    if transform is None:
        transform = lambda x: x

    write_pointer = 0
    # loop for each trial
    for trial_id in trail_ids:
        # extract baseline signals
        trail_samples = samples[trial_id]  # channel(62), timestep(n*200)

        # record the common meta info
        trail_meta_info = {
            'subject': subject,
            'trail_id': trial_id,
            'emotion': labels[int(trial_id.split('_')[-1][3:]) - 1],
            'date': date
        }

        # extract experimental signals
        start_at = 0
        end_at = chunk_size

        while end_at <= trail_samples.shape[1]:
            clip_sample = trail_samples[:channel_num, start_at:end_at]
            transformed_eeg = transform(clip_sample)
            clip_id = f'{file_name}_{write_pointer}'
            queue.put({'eeg': transformed_eeg, 'key': clip_id})
            write_pointer += 1

            # record meta info for each signal
            record_info = {
                'start_at': start_at,
                'end_at': end_at,
                'clip_id': clip_id
            }
            record_info.update(trail_meta_info)
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


def seed_constructor(root_path: str = './Preprocessed_EEG',
                     chunk_size: int = 200,
                     overlap: int = 0,
                     channel_num: int = 62,
                     transform: Union[None, Callable] = None,
                     io_path: str = './io/seed',
                     num_worker: int = 1,
                     verbose: bool = True) -> None:
    # init IO
    if os.path.exists(io_path):
        if verbose:
            print(
                f'The target folder already exists, if you need to regenerate the database IO, please delete the path {io_path}.'
            )
        return
    os.makedirs(io_path)

    meta_info_io_path = os.path.join(io_path, 'info.csv')
    eeg_signal_io_path = os.path.join(io_path, 'eeg')

    info_io = MetaInfoIO(meta_info_io_path)
    eeg_io = EEGSignalIO(eeg_signal_io_path)

    # loop to access the dataset files
    file_list = os.listdir(root_path)
    skip_set = ['label.mat', 'readme.txt']
    file_list = [f for f in file_list if f not in skip_set]

    if verbose:
        # show process bar
        pbar = tqdm(total=len(file_list))
        pbar.set_description("[SEED]")

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
                            channel_num=channel_num,
                            transform=transform,
                            write_info_fn=info_io.write_info,
                            queue=queue)

    for _ in Pool(num_worker).imap(partial_mp_fn, file_list):
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
        print('Please wait for the writing process to complete...')

    queue.put(None)

    io_consumer_process.join()
    io_consumer_process.close()
