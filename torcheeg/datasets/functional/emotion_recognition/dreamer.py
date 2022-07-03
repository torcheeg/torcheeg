import os
from typing import Callable, List, Union
from functools import partial
import scipy.io as scio
from multiprocessing import Manager, Pool, Process, Queue, set_start_method
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024


def transform_producer(subject: int, trial_len: int, mat_data: any,
                       chunk_size: int, overlap: int, channel_num: int,
                       baseline_num: int, baseline_chunk_size: int,
                       transform: Union[List[Callable], Callable, None],
                       write_info_fn: Callable, queue: Queue):
    # calculate moving step
    step = chunk_size - overlap

    write_pointer = 0
    # loop for each trial
    for trial_id in range(trial_len):
        # extract baseline signals
        trial_baseline_sample = mat_data['DREAMER'][0, 0]['Data'][
            0, subject]['EEG'][0, 0]['baseline'][0, 0][trial_id, 0]
        trial_baseline_sample = trial_baseline_sample[:, :channel_num].swapaxes(
            1, 0)  # channel(14), timestep(61*128)
        trial_baseline_sample = trial_baseline_sample[:, :baseline_num *
                                                      baseline_chunk_size].reshape(
                                                          channel_num,
                                                          baseline_num,
                                                          baseline_chunk_size
                                                      ).mean(
                                                          axis=1
                                                      )  # channel(14), timestep(128)

        # record the common meta info
        trial_meta_info = {'subject_id': subject, 'trial_id': trial_id}

        trial_meta_info['valence'] = mat_data['DREAMER'][0, 0]['Data'][
            0, subject]['ScoreValence'][0, 0][trial_id, 0]
        trial_meta_info['arousal'] = mat_data['DREAMER'][0, 0]['Data'][
            0, subject]['ScoreArousal'][0, 0][trial_id, 0]
        trial_meta_info['dominance'] = mat_data['DREAMER'][0, 0]['Data'][
            0, subject]['ScoreDominance'][0, 0][trial_id, 0]

        # extract experimental signals
        start_at = 0
        end_at = chunk_size

        trial_samples = mat_data['DREAMER'][0, 0]['Data'][0, subject]['EEG'][
            0, 0]['stimuli'][0, 0][trial_id, 0]
        trial_samples = trial_samples[:, :channel_num].swapaxes(
            1, 0)  # channel(14), timestep(n*128)

        while end_at <= trial_samples.shape[1]:
            clip_sample = trial_samples[:, start_at:end_at]

            t_eeg = clip_sample
            t_baseline = trial_baseline_sample

            if not transform is None:
                t = transform(eeg=clip_sample, baseline=trial_baseline_sample)
                t_eeg = t['eeg']
                t_baseline = t['baseline']

            # put baseline signal into IO
            if not 'baseline_id' in trial_meta_info:
                trial_base_id = f'{subject}_{write_pointer}'
                queue.put({'eeg': t_baseline, 'key': trial_base_id})
                write_pointer += 1
                trial_meta_info['baseline_id'] = trial_base_id

            clip_id = f'{subject}_{write_pointer}'

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


def dreamer_constructor(mat_path: str = './DREAMER.mat',
                        chunk_size: int = 128,
                        overlap: int = 0,
                        channel_num: int = 14,
                        baseline_num: int = 61,
                        baseline_chunk_size: int = 128,
                        transform: Union[None, Callable] = None,
                        io_path: str = './io/dreamer',
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

    # access the dataset files
    mat_data = scio.loadmat(mat_path, verify_compressed_data_integrity=False)

    subject_len = len(mat_data['DREAMER'][0, 0]['Data'][0])  # 23
    trial_len = len(
        mat_data['DREAMER'][0, 0]['Data'][0, 0]['EEG'][0,
                                                       0]['stimuli'][0,
                                                                     0])  # 18

    if verbose:
        # show process bar
        pbar = tqdm(total=subject_len)
        pbar.set_description("[DREAMER]")

    if num_worker > 1:
        manager = Manager()
        queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
        io_consumer_process = Process(target=io_consumer,
                                      args=(eeg_io.write_eeg, queue),
                                      daemon=True)
        io_consumer_process.start()

        partial_mp_fn = partial(transform_producer,
                                trial_len=trial_len,
                                mat_data=mat_data,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                channel_num=channel_num,
                                baseline_num=baseline_num,
                                baseline_chunk_size=baseline_chunk_size,
                                transform=transform,
                                write_info_fn=info_io.write_info,
                                queue=queue)

        for _ in Pool(num_worker).imap(partial_mp_fn, list(range(subject_len))):
            if verbose:
                pbar.update(1)

        queue.put(None)

        io_consumer_process.join()
        io_consumer_process.close()

    else:
        for subject in list(range(subject_len)):
            transform_producer(subject=subject,
                               trial_len=trial_len,
                               mat_data=mat_data,
                               chunk_size=chunk_size,
                               overlap=overlap,
                               channel_num=channel_num,
                               baseline_num=baseline_num,
                               baseline_chunk_size=baseline_chunk_size,
                               transform=transform,
                               write_info_fn=info_io.write_info,
                               queue=SingleProcessingQueue(eeg_io.write_eeg))
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
        print('Please wait for the writing process to complete...')
