import os
import re
import numpy as np

from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, Union

import scipy.io as scio
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024


def transform_producer(file_path: str, feature: list, num_channel: int,
                       before_trial: Union[None, Callable],
                       transform: Union[None, Callable],
                       after_trial: Union[Callable, None], queue: Queue):
    session_id = os.path.basename(os.path.dirname(file_path))
    _, file_name = os.path.split(file_path)

    subject = int(
        os.path.basename(file_name).split('.')[0].split('_')[0])  # subject (15)
    date = int(
        os.path.basename(file_name).split('.')[0].split('_')[1])  # period (3)

    samples = scio.loadmat(file_path, verify_compressed_data_integrity=False
                           )  # trial (15), channel(62), timestep(n*200)
    # label file
    labels = [[
        1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3
    ], [
        2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1
    ], [
        1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0
    ]]  # The labels with 0, 1, 2, and 3 denote the ground truth, neutral, sad, fear, and happy emotions, respectively.
    session_labels = labels[int(session_id) - 1]

    trial_ids = [
        int(re.findall(r"de_movingAve(\d+)", key)[0]) for key in samples.keys()
        if 'de_movingAve' in key
    ]

    write_pointer = 0
    # loop for each trial
    for trial_id in trial_ids:
        # extract baseline signals
        trial_samples = []
        for cur_feature in feature:
            trial_samples.append(
                samples[cur_feature +
                        str(trial_id)])  # channel(61), timestep(n), bands(5)
        trial_samples = np.concatenate(
            trial_samples,
            axis=-1)[:num_channel]  # channel(61), timestep(n), features(5*k)
        trial_samples = trial_samples.transpose((1, 0, 2))
        # timestep(n), channel(61), features(5*k)

        if before_trial:
            trial_samples = before_trial(trial_samples)

        # record the common meta info
        trial_meta_info = {
            'subject_id': subject,
            'trial_id': trial_id,
            'session_id': session_id,
            'emotion': int(session_labels[trial_id - 1]),
            'date': date
        }

        trial_queue = []
        for i, clip_sample in enumerate(trial_samples):
            t_eeg = clip_sample
            if not transform is None:
                t_eeg = transform(eeg=clip_sample)['eeg']

            clip_id = f'{file_name}_{write_pointer}'
            write_pointer += 1

            # record meta info for each signal
            record_info = {
                'start_at': i * 400,
                'end_at': (i + 1) *
                400,  # The size of the sliding time windows for feature extraction is 4 seconds.
                'clip_id': clip_id
            }
            record_info.update(trial_meta_info)
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


def seed_iv_feature_constructor(root_path: str = './eeg_feature_smooth',
                                feature: list = ['de_movingAve'],
                                num_channel: int = 61,
                                before_trial: Union[None, Callable] = None,
                                transform: Union[None, Callable] = None,
                                after_trial: Union[Callable, None] = None,
                                io_path: str = './io/seed_iv_feature',
                                io_size: int = 10485760,
                                io_mode: str = 'lmdb',
                                num_worker: int = 0,
                                verbose: bool = True) -> None:
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
    session_list = ['1', '2', '3']
    file_path_list = []
    for session in session_list:
        session_root_path = os.path.join(root_path, session)
        for file_name in os.listdir(session_root_path):
            file_path_list.append(os.path.join(session_root_path, file_name))

    if verbose:
        # show process bar
        pbar = tqdm(total=len(file_path_list))
        pbar.set_description("[SEED-IV FEATURE]")

    if num_worker > 1:
        manager = Manager()
        queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
        io_consumer_process = Process(target=io_consumer,
                                      args=(eeg_io.write_eeg,
                                            info_io.write_info, queue),
                                      daemon=True)
        io_consumer_process.start()

        partial_mp_fn = partial(transform_producer,
                                feature=feature,
                                num_channel=num_channel,
                                before_trial=before_trial,
                                transform=transform,
                                after_trial=after_trial,
                                queue=queue)

        for _ in Pool(num_worker).imap(partial_mp_fn, file_path_list):
            if verbose:
                pbar.update(1)

        queue.put(None)

        io_consumer_process.join()
        io_consumer_process.close()

    else:
        for file_path in file_path_list:
            transform_producer(file_path=file_path,
                               feature=feature,
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
