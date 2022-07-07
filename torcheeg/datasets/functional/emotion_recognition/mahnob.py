import glob
import json
import os
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue, set_start_method
from typing import Callable, List, Union

import mne
import xmltodict
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024


def transform_producer(file_name: str, root_path: str, chunk_size: int,
                       sampling_rate: int, overlap: int, channel_num: int,
                       baseline_num: int, baseline_chunk_size: int,
                       trial_sample_num: int, transform: Union[List[Callable],
                                                               Callable, None],
                       write_info_fn: Callable, queue: Queue):
    trial_dir = os.path.join(root_path, file_name)

    mne.set_log_level('CRITICAL')
    # record the common meta info for the trial
    label_file = os.path.join(trial_dir, 'session.xml')
    emodims = [
        '@feltArsl', '@feltCtrl', '@feltEmo', '@feltPred', '@feltVlnc',
        '@isStim'
    ]
    with open(label_file) as f:
        label_info = xmltodict.parse('\n'.join(f.readlines()))
    label_info = json.loads(json.dumps(label_info))['session']

    if not '@feltArsl' in label_info:
        # skip label_info['@isStim'] == '0' and other exception
        return

    trial_meta_info = {
        'subject_id': label_info['subject']['@id'],
        'trial_id': label_info['@mediaFile'],
        'duration': float(label_info['@cutLenSec'])
    }
    # feltArsl, feltCtrl, feltEmo, feltPred, feltVlnc, isStim
    trial_meta_info.update({k[1:]: int(label_info[k]) for k in emodims})

    write_pointer = 0

    # extract signals
    sample_file = glob.glob(str(os.path.join(trial_dir, '*.bdf')))[0]

    raw = mne.io.read_raw_bdf(sample_file, preload=True, stim_channel='Status')
    events = mne.find_events(raw, stim_channel='Status')

    montage = mne.channels.make_standard_montage(kind='biosemi32')
    raw.set_montage(montage, on_missing='ignore')

    # pick channels
    raw.pick_channels(raw.ch_names[:channel_num])

    start_samp, end_samp = events[0][0] + 1, events[1][0] - 1

    # extract baseline signals
    trial_baseline_raw = raw.copy().crop(raw.times[0], raw.times[end_samp])
    trial_baseline_raw = trial_baseline_raw.resample(sampling_rate)

    trial_baseline_sample = trial_baseline_raw.to_data_frame().to_numpy(
    )[:, 1:].swapaxes(1, 0)  # channel(32), timestep(30 * 128)
    trial_baseline_sample = trial_baseline_sample[:, :baseline_num *
                                                  baseline_chunk_size]
    trial_baseline_sample = trial_baseline_sample.reshape(
        channel_num, baseline_num,
        baseline_chunk_size).mean(axis=1)  # channel(32), timestep(128)

    # extract experimental signals
    trial_raw = raw.copy().crop(raw.times[start_samp], raw.times[end_samp])
    trial_raw = trial_raw.resample(sampling_rate)
    trial_samples = trial_raw.to_data_frame().to_numpy()[:, 1:].swapaxes(1, 0)

    # calculate moving step
    step = chunk_size - overlap

    start_at = 0
    end_at = start_at + chunk_size

    max_len = trial_samples.shape[1]
    if not (trial_sample_num <= 0):
        max_len = min(trial_sample_num * chunk_size, trial_samples.shape[1])

    while end_at <= max_len:
        clip_sample = trial_samples[:, start_at:end_at]

        t_eeg = clip_sample
        t_baseline = trial_baseline_sample
        if not transform is None:
            t = transform(eeg=clip_sample, baseline=trial_baseline_sample)
            t_eeg = t['eeg']
            t_baseline = t['baseline']

        # put baseline signal into IO
        if not 'baseline_id' in trial_meta_info:
            trial_base_id = f'{file_name}_{write_pointer}'

            queue.put({'eeg': t_baseline, 'key': trial_base_id})
            write_pointer += 1
            trial_meta_info['baseline_id'] = trial_base_id

        clip_id = f'{file_name}_{write_pointer}'
        queue.put({'eeg': t_eeg, 'key': clip_id})
        write_pointer += 1

        # record meta info for each signal
        record_info = {
            'start_at': start_at,
            'end_at': end_at,
            'clip_id': clip_id
        }
        record_info.update(trial_meta_info)
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


def mahnob_constructor(root_path: str = './Sessions',
                       chunk_size: int = 128,
                       sampling_rate: int = 128,
                       overlap: int = 0,
                       channel_num: int = 32,
                       baseline_num: int = 30,
                       baseline_chunk_size: int = 128,
                       trial_sample_num: int = 30,
                       transform: Union[None, Callable] = None,
                       io_path: str = './io/mahnob',
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

    info_io = MetaInfoIO(meta_info_io_path)
    eeg_io = EEGSignalIO(eeg_signal_io_path, cache_size=cache_size)

    # loop to access the dataset files
    file_list = os.listdir(root_path)

    if verbose:
        # show process bar
        pbar = tqdm(total=len(file_list))
        pbar.set_description("[MAHNOB]")

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
                                sampling_rate=sampling_rate,
                                overlap=overlap,
                                channel_num=channel_num,
                                baseline_num=baseline_num,
                                baseline_chunk_size=baseline_chunk_size,
                                trial_sample_num=trial_sample_num,
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
                               sampling_rate=sampling_rate,
                               overlap=overlap,
                               channel_num=channel_num,
                               baseline_num=baseline_num,
                               baseline_chunk_size=baseline_chunk_size,
                               trial_sample_num=trial_sample_num,
                               transform=transform,
                               write_info_fn=info_io.write_info,
                               queue=SingleProcessingQueue(eeg_io.write_eeg))
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
        print('Please wait for the writing process to complete...')
