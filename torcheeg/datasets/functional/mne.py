import os
import mne
import numpy as np

from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, Dict, Tuple, Union, List

from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024


def transform_producer(epochs_metadata_rank: Tuple[mne.Epochs, Dict, int],
                       chunk_size: int, overlap: int, num_channel: int,
                       before_trial: Union[Callable,
                                           None], transform: Union[Callable,
                                                                   None],
                       after_trial: Union[Callable, None], queue: Queue):

    epochs, metadata, rank = epochs_metadata_rank

    if chunk_size <= 0:
        chunk_size = len(epochs.times)

    if num_channel == -1:
        num_channel = len(epochs.info['chs'])

    trial_event_index = epochs.events[:, 2]
    trial_start_at_list = epochs.events[:, 0]

    trial_end_at = len(epochs.times) - chunk_size

    clip_sample_start_at_list = np.arange(0, trial_end_at + 1,
                                          chunk_size - overlap)

    sample_events = [[clip_sample_start_at, chunk_size, -1]
                     for clip_sample_start_at in clip_sample_start_at_list]

    epoch_meta_info = metadata

    write_pointer = 0

    # for loop of trials
    for trial_id, trial in enumerate(epochs):
        # split sample from epochs
        start_at_list = clip_sample_start_at_list + trial_start_at_list[trial_id]
        end_at_list = clip_sample_start_at_list + trial_start_at_list[
            trial_id] + chunk_size
        event_index_list = len(sample_events) * [trial_event_index[trial_id]]

        trial_samples = mne.Epochs(mne.io.RawArray(trial, epochs.info),
                                   sample_events,
                                   baseline=None,
                                   tmin=0,
                                   tmax=(chunk_size - 1) / epochs.info["sfreq"])
        trial_samples.drop_bad(reject=None, flat=None)
        if before_trial:
            trial_samples = before_trial(trial_samples)

        # for loop of samples
        trial_queue = []
        for i, trial_signal in enumerate(trial_samples.get_data()):
            t_eeg = trial_signal[:num_channel, :]
            if not transform is None:
                t = transform(eeg=trial_signal[:num_channel, :])
                t_eeg = t['eeg']

            clip_id = f'{rank}_{write_pointer}'
            write_pointer += 1

            record_info = {
                'trial_id': trial_id,
                'start_at': start_at_list[i],
                'end_at': end_at_list[i],
                'event': event_index_list[i],
                'clip_id': clip_id
            }
            record_info.update(epoch_meta_info)
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


def mne_constructor(epochs_list: List[mne.Epochs],
                    metadata_list: List[Dict],
                    chunk_size: int = -1,
                    overlap: int = 0,
                    num_channel: int = -1,
                    before_trial: Union[None, Callable] = None,
                    transform: Union[None, Callable] = None,
                    after_trial: Union[Callable, None] = None,
                    io_path: str = './io/mne',
                    io_size: int = 10485760,
                    io_mode: str = 'lmdb',
                    num_worker: int = 0,
                    verbose: bool = True) -> None:

    mne.set_log_level('CRITICAL')

    assert len(epochs_list) == len(
        metadata_list
    ), f'The number of mne.Epochs {len(epochs_list)} does not match the number of metadata {len(metadata_list)}.'

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

    epochs_metadata_rank_list = []
    for rank, (epochs, metadata) in enumerate(zip(epochs_list, metadata_list)):
        assert (epochs.tmax - epochs.tmin) * epochs.info[
            "sfreq"] >= chunk_size, f'chunk_size cannot be larger than (tmax - tmin) * sfreq. Here, tmax is set to {epochs.tmax}, tmin is set to {epochs.tmin}, and sfreq is {epochs.info["sfreq"]}. In the current configuration, chunk_size {chunk_size} is greater than {(epochs.tmax - epochs.tmin) * epochs.info["sfreq"]}!'
        epochs_metadata_rank_list.append((epochs, metadata, rank))

    if verbose:
        # show process bar
        pbar = tqdm(total=len(epochs_metadata_rank_list))
        pbar.set_description("[MNE]")

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
                                overlap=overlap,
                                num_channel=num_channel,
                                before_trial=before_trial,
                                transform=transform,
                                after_trial=after_trial,
                                queue=queue)

        for _ in Pool(num_worker).imap(partial_mp_fn,
                                       epochs_metadata_rank_list):
            if verbose:
                pbar.update(1)

        queue.put(None)

        io_consumer_process.join()
        io_consumer_process.close()

    else:
        for epochs_metadata_rank in epochs_metadata_rank_list:
            transform_producer(epochs_metadata_rank=epochs_metadata_rank,
                               chunk_size=chunk_size,
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
