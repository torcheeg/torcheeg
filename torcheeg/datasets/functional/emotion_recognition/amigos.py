import os
import re
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, List, Union

import scipy.io as scio
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1099511627776


def transform_producer(file_name: str, root_path: str, chunk_size: int, overlap: int, channel_num: int, trial_num: int,
                       skipped_subjects: List, baseline_num: int, baseline_chunk_size: int,
                       transform: Union[List[Callable], Callable, None], write_info_fn: Callable, queue: Queue):
    subject = int(re.findall(r'Data_Preprocessed_P(\d*).mat', file_name)[0])  # subject (40)

    if subject in skipped_subjects:
        return

    data = scio.loadmat(os.path.join(root_path, file_name), verify_compressed_data_integrity=False)
    samples = data['joined_data'][0]  # trail (20), timestep(n*128), channel(17) (14 channels are EEGs)
    # label file
    labels = data['labels_selfassessment'][0]  # trail (20), label of different dimensions ((1, 12))

    # calculate moving step
    step = chunk_size - overlap

    # prepare transform
    if transform is None:
        transform = lambda x: x

    write_pointer = 0

    max_len = len(samples)
    if not (trial_num <= 0):
        max_len = min(len(samples), trial_num)

    # loop for each trial
    for trial_id in range(max_len):
        # extract baseline signals
        trail_samples = samples[trial_id]  # timestep(n*128), channel(17)

        # record the common meta info
        trail_meta_info = {'subject': subject, 'trail_id': trial_id}
        trail_rating = labels[trial_id][0]  # label of different dimensions (12)

        # missing values
        if (not sum(trail_samples.shape)) or (not sum(trail_rating.shape)):
            # 3 of the participants (08,24,28<->32) of the previous experiment did not watch a set of 4 long affective
            if sum(trail_samples.shape) != sum(trail_rating.shape):
                print(
                    f'[WARNING] Find EEG signals without labels, or labels without EEG signals. Please check the {trial_id + 1}-th experiment of the {subject}-th subject in the file {file_name}. TorchEEG currently skipped the mismatched data.'
                )
            continue

        for label_idx, label_name in enumerate([
                'arousal', 'valence', 'dominance', 'liking', 'familiarity', 'neutral', 'disgust', 'happiness',
                'surprise', 'anger', 'fear', 'sadness'
        ]):
            trail_meta_info[label_name] = trail_rating[label_idx]

        # extract baseline signals
        trail_baseline_sample = trail_samples[:baseline_chunk_size *
                                              baseline_num, :channel_num]  # timestep(5*128), channel(14)
        trail_baseline_sample = trail_baseline_sample.reshape(baseline_num,
                                                              baseline_chunk_size, channel_num).mean(axis=0).swapaxes(
                                                                  1, 0)  # channel(14), timestep(128)

        # put baseline signal into IO
        transformed_eeg = transform(trail_baseline_sample)
        trail_base_id = f'{file_name}_{write_pointer}'
        queue.put({'eeg': transformed_eeg, 'key': trail_base_id})
        write_pointer += 1
        trail_meta_info['baseline_id'] = trail_base_id

        # extract experimental signals
        start_at = baseline_chunk_size * baseline_num
        end_at = start_at + chunk_size

        while end_at <= trail_samples.shape[0]:
            clip_sample = trail_samples[start_at:end_at, :channel_num].swapaxes(1, 0)
            transformed_eeg = transform(clip_sample)
            clip_id = f'{file_name}_{write_pointer}'
            queue.put({'eeg': transformed_eeg, 'key': clip_id})
            write_pointer += 1

            # record meta info for each signal
            record_info = {'start_at': start_at, 'end_at': end_at, 'clip_id': clip_id}
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


def amigos_constructor(root_path: str = './data_preprocessed',
                       chunk_size: int = 128,
                       overlap: int = 0,
                       channel_num: int = 14,
                       trial_num: int = 16,
                       skipped_subjects: List[int] = [9, 12, 21, 22, 23, 24, 33],
                       baseline_num: int = 5,
                       baseline_chunk_size: int = 128,
                       transform: Union[None, Callable] = None,
                       io_path: str = './io/amigos',
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

    if verbose:
        # show process bar
        pbar = tqdm(total=len(file_list))
        pbar.set_description("[AMIGOS]")

    manager = Manager()
    queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
    io_consumer_process = Process(target=io_consumer, args=(eeg_io.write_eeg, queue), daemon=True)
    io_consumer_process.start()

    partial_mp_fn = partial(transform_producer,
                            root_path=root_path,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            channel_num=channel_num,
                            trial_num=trial_num,
                            skipped_subjects=skipped_subjects,
                            baseline_num=baseline_num,
                            baseline_chunk_size=baseline_chunk_size,
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
