import os
import pickle as pkl
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, List, Union

from sklearn import preprocessing
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1099511627776


def transform_producer(file_name: str, root_path: str, chunk_size: int,
                       overlap: int, channel_num: int, baseline_num: int,
                       baseline_chunk_size: int,
                       transform: Union[List[Callable], Callable,
                                        None], write_info_fn: Callable,
                       label_encoder: preprocessing.LabelEncoder, queue: Queue):
    with open(os.path.join(root_path, file_name), 'rb') as f:
        pkl_data = pkl.load(f, encoding='iso-8859-1')

    samples = pkl_data['data']  # trail(40), channel(32), timestep(63*128)
    labels = pkl_data['labels']
    subject_id = label_encoder.transform([file_name])[0]

    # calculate moving step
    step = chunk_size - overlap

    # prepare transform
    if transform is None:
        transform = lambda x: x

    write_pointer = 0
    # loop for each trial
    for trial_id in range(len(samples)):
        # extract baseline signals
        trail_samples = samples[trial_id, :
                                channel_num]  # channel(32), timestep(63*128)
        trail_baseline_sample = trail_samples[:, :baseline_chunk_size *
                                              baseline_num]  # channel(32), timestep(3*128)
        trail_baseline_sample = trail_baseline_sample.reshape(
            channel_num, baseline_num,
            baseline_chunk_size).mean(axis=1)  # channel(32), timestep(128)

        # put baseline signal into IO
        transformed_eeg = transform(trail_baseline_sample)
        trail_base_id = f'{file_name}_{write_pointer}'
        queue.put({
            'eeg': transformed_eeg,
            'key': trail_base_id
        })
        write_pointer += 1

        # record the common meta info
        trail_meta_info = {
            'subject_id': subject_id,
            'trail_id': trial_id,
            'baseline_id': trail_base_id
        }
        trail_rating = labels[trial_id]

        for label_idx, label_name in enumerate(
            ['valence', 'arousal', 'dominance', 'liking']):
            trail_meta_info[label_name] = trail_rating[label_idx]

        # extract experimental signals
        start_at = baseline_chunk_size * baseline_num
        end_at = start_at + chunk_size

        while end_at <= trail_samples.shape[1]:
            clip_sample = trail_samples[:, start_at:end_at]
            transformed_eeg = transform(clip_sample)

            clip_id = f'{file_name}_{write_pointer}'
            queue.put({
                'eeg': transformed_eeg,
                'key': clip_id
            })
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


def deap_constructor(root_path: str = './data_preprocessed_python',
                     chunk_size: int = 128,
                     overlap: int = 0,
                     channel_num: int = 32,
                     baseline_num: int = 3,
                     baseline_chunk_size: int = 128,
                     transform: Union[None, Callable] = None,
                     io_path: str = './io/deap',
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
        pbar.set_description("[DEAP]")

    # label subject files
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(file_list)

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
                            baseline_num=baseline_num,
                            baseline_chunk_size=baseline_chunk_size,
                            transform=transform,
                            write_info_fn=info_io.write_info,
                            label_encoder=label_encoder,
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
