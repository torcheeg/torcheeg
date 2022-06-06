import os
from typing import Callable, List, Union
from functools import partial
import scipy.io as scio
from multiprocessing import Manager, Pool, Process, Queue
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1099511627776


def transform_producer(subject: int, trial_len: int, mat_data: any,
                       chunk_size: int, overlap: int, channel_num: int,
                       baseline_num: int, baseline_chunk_size: int,
                       transform: Union[List[Callable], Callable, None],
                       write_info_fn: Callable, queue: Queue):
    # calculate moving step
    step = chunk_size - overlap

    # prepare transform
    if transform is None:
        transform = lambda x: x

    write_pointer = 0
    # loop for each trial
    for trial_id in range(trial_len):
        # extract baseline signals
        trail_baseline_sample = mat_data['DREAMER'][0, 0]['Data'][
            0, subject]['EEG'][0, 0]['baseline'][0, 0][trial_id, 0]
        trail_baseline_sample = trail_baseline_sample[:, :channel_num].swapaxes(
            1, 0)  # channel(14), timestep(61*128)
        trail_baseline_sample = trail_baseline_sample[:, :baseline_num *
                                                      baseline_chunk_size].reshape(
                                                          channel_num,
                                                          baseline_num,
                                                          baseline_chunk_size
                                                      ).mean(
                                                          axis=1
                                                      )  # channel(14), timestep(128)

        # put baseline signal into IO
        transformed_eeg = transform(trail_baseline_sample)
        trail_base_id = f'{subject}_{write_pointer}'
        queue.put({'eeg': transformed_eeg, 'key': trail_base_id})
        write_pointer += 1

        # record the common meta info
        trail_meta_info = {
            'subject': subject,
            'trail_id': trial_id,
            'baseline_id': trail_base_id
        }

        trail_meta_info['valence'] = mat_data['DREAMER'][0, 0]['Data'][
            0, subject]['ScoreValence'][0, 0][trial_id, 0]
        trail_meta_info['arousal'] = mat_data['DREAMER'][0, 0]['Data'][
            0, subject]['ScoreArousal'][0, 0][trial_id, 0]
        trail_meta_info['dominance'] = mat_data['DREAMER'][0, 0]['Data'][
            0, subject]['ScoreDominance'][0, 0][trial_id, 0]

        # extract experimental signals
        start_at = 0
        end_at = chunk_size

        trail_samples = mat_data['DREAMER'][0, 0]['Data'][0, subject]['EEG'][
            0, 0]['stimuli'][0, 0][trial_id, 0]
        trail_samples = trail_samples[:, :channel_num].swapaxes(
            1, 0)  # channel(14), timestep(n*128)

        while end_at <= trail_samples.shape[1]:
            clip_sample = trail_samples[:, start_at:end_at]
            transformed_eeg = transform(clip_sample)
            clip_id = f'{subject}_{write_pointer}'
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


def dreamer_constructor(mat_path: str = './DREAMER.mat',
                        chunk_size: int = 128,
                        overlap: int = 0,
                        channel_num: int = 14,
                        baseline_num: int = 61,
                        baseline_chunk_size: int = 128,
                        transform: Union[None, Callable] = None,
                        io_path: str = './io/dreamer',
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

    if verbose:
        pbar.close()
        print('Please wait for the writing process to complete...')

    queue.put(None)

    io_consumer_process.join()
    io_consumer_process.close()
