import os
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue, set_start_method
from typing import Callable, List, Union

import joblib
import numpy as np
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024

FIRST_BATCH_CHANNEL = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T7', 'T8', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ',
    'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2'
]

SECOND_BATCH_CHANNEL = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4',
    'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2', 'A2', 'A1'
]

FIRST_TO_SECOND_ORDER = [
    FIRST_BATCH_CHANNEL.index(c) for c in SECOND_BATCH_CHANNEL
]

VALENCE_DICT = {
    1: -1,  # negative
    2: -1,
    3: -1,
    4: -1,
    5: -1,
    6: -1,
    7: -1,
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: 0,  # neutral
    14: 0,
    15: 0,
    16: 0,
    17: 1,  # positive
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1
}

EMOTION_DICT = {
    1: 0,  # anger	
    2: 0,
    3: 0,
    4: 1,  # disgust
    5: 1,
    6: 1,
    7: 2,  # fear
    8: 2,
    9: 2,
    10: 3,  # sadness
    11: 3,
    12: 3,
    13: 4,  # neutral
    14: 4,
    15: 4,
    16: 4,
    17: 5,  # amusement
    18: 5,
    19: 5,
    20: 6,  # excitation
    21: 6,
    22: 6,
    23: 7,  # happy
    24: 7,
    25: 7,
    26: 8,  # warmth
    27: 8,
    28: 8
}


def transform_producer(file_name: str, root_path: str, chunk_size: int,
                       overlap: int, channel_num: int,
                       transform: Union[List[Callable], Callable, None],
                       reorder: bool, write_info_fn: Callable, queue: Queue):

    subject = file_name  # subject (54)
    samples = joblib.load(
        os.path.join(root_path, file_name,
                     f'{file_name}.pkl'))  # channel(33), timestep(n*250)

    events = samples[-1]
    if reorder:
        samples = samples.take(FIRST_TO_SECOND_ORDER, axis=0)

    # initial pointers
    trial_id = 0
    write_pointer = 0

    video_id = None
    start_at = None
    end_at = None

    # loop for each trial
    for i, event in enumerate(events):

        if event in list(range(1, 29)):
            # Video events 1-28: Different events correspond to different experimental video materials
            video_id = event
        elif event == 240:
            # Current trial video start event 240: This event appears 0.1s after the video event, indicating that the video starts to play.
            start_at = i
        elif event == 241:
            # Current trial video end event 241: This event indicates that the video ends playing. Block end event 243: This event indicates the end of the block.
            end_at = i
            assert (not video_id is None) and (
                not start_at is None
            ), f'Parse event fail for trial {trial_id} with video_id={video_id}, start_at={start_at}, end_at={end_at}!'

            trial_meta_info = {
                'trial_id': trial_id,
                'video_id': video_id,
                'subject_id': subject,
                'valence': VALENCE_DICT[video_id],
                'emotion': EMOTION_DICT[video_id],
            }

            step = chunk_size - overlap
            cur_start_at = start_at
            cur_end_at = start_at + chunk_size

            while cur_end_at <= end_at:
                t_eeg = samples[:channel_num, cur_start_at:cur_end_at]
                if not transform is None:
                    t_eeg = transform(eeg=t_eeg)['eeg']

                clip_id = f'{subject}_{write_pointer}'
                queue.put({'eeg': t_eeg, 'key': clip_id})
                write_pointer += 1

                record_info = {
                    'start_at': cur_start_at,
                    'end_at': cur_end_at,
                    'clip_id': clip_id
                }
                record_info.update(trial_meta_info)
                write_info_fn(record_info)

                cur_start_at = cur_start_at + step
                cur_end_at = cur_start_at + chunk_size

            # prepare for the next trial
            trial_id += 1
            video_id = None
            start_at = None
            end_at = None


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


def bci2022_constructor(root_path: str = './2022EmotionPublic/TrainSet/',
                        chunk_size: int = 250,
                        overlap: int = 0,
                        channel_num: int = 30,
                        transform: Union[None, Callable] = None,
                        io_path: str = './io/bci2022',
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

    for train_set_batch_idx, train_set_batch in enumerate(
        ['TrainSet_first_batch', 'TrainSet_second_batch']):
        # loop to access the first batch files
        file_list = os.listdir(os.path.join(root_path, train_set_batch))

        if verbose:
            # show process bar
            pbar = tqdm(total=len(file_list))
            pbar.set_description(f"[BCI2022 BATCH {train_set_batch_idx+1}/2]")

        if num_worker > 1:
            manager = Manager()
            queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
            io_consumer_process = Process(target=io_consumer,
                                          args=(eeg_io.write_eeg, queue),
                                          daemon=True)
            io_consumer_process.start()

            partial_mp_fn = partial(
                transform_producer,
                root_path=os.path.join(root_path, train_set_batch),
                chunk_size=chunk_size,
                overlap=overlap,
                channel_num=channel_num,
                transform=transform,
                reorder=train_set_batch == 'TrainSet_first_batch',
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
                transform_producer(
                    file_name=file_name,
                    root_path=os.path.join(root_path, train_set_batch),
                    chunk_size=chunk_size,
                    overlap=overlap,
                    channel_num=channel_num,
                    transform=transform,
                    reorder=train_set_batch == 'TrainSet_first_batch',
                    write_info_fn=info_io.write_info,
                    queue=SingleProcessingQueue(eeg_io.write_eeg))
                if verbose:
                    pbar.update(1)

        if verbose:
            pbar.close()
            print('Please wait for the writing process to complete...')
