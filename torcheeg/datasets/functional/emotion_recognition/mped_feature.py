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


def transform_producer(file_name: str, root_path: str, feature: list,
                       num_channel: int, before_trial: Union[None, Callable],
                       transform: Union[None, Callable],
                       after_trial: Union[Callable, None], queue: Queue):
    labels = [
        0, 2, 1, 4, 3, 5, 6, 7, 1, 2, 3, 6, 7, 4, 5, 0, 1, 5, 6, 2, 2, 1, 7, 6,
        4, 4, 3, 5, 3, 7
    ]  # 0-resting status, 1-neutral, 2-joy, 3-funny, 4-angry, 5-fear, 6-disgust, 7-sadness

    subject = os.path.basename(file_name).split('.')[0].split('_')[0]

    samples_dict = {}
    for cur_feature in feature:
        samples_dict[cur_feature] = scio.loadmat(
            os.path.join(root_path, cur_feature, file_name),
            verify_compressed_data_integrity=False)  # (1, 30)
    write_pointer = 0

    for trial_id in range(30):
        trial_samples = []
        for cur_feature, samples in samples_dict.items():
            for sub_band in samples.keys():
                # HHS: hhs_A, hhs_E
                # Hjorth/HOC: alpha, beta, delta, gamma, theta, whole
                # STFT: STFT
                # PSD: PSD
                if not str.startswith(sub_band, '__'):
                    # not '__header__', '__version__', '__globals__'
                    trial_samples += [
                        samples[sub_band][0][trial_id][:num_channel],
                    ]
                    # PSD: (62, 120, 5)
                    # Hjorth: (62, 120, 3)
                    # HOC: (62, 120, 20)
                    # HHS: (62, 120, 5)
                    # STFT: (62, 120, 5)
        trial_samples = np.concatenate(trial_samples,
                                       axis=-1)  # (62, 120, num_features)
        if before_trial:
            trial_samples = before_trial(trial_samples)

        # record the common meta info
        trial_meta_info = {
            'subject_id': subject,
            'trial_id': trial_id,
            'emotion': labels[trial_id]
        }
        num_clips = trial_samples.shape[1]

        trial_queue = []
        for clip_id in range(num_clips):
            # PSD: (62, 5)
            # Hjorth: (62, 3 * 6)
            # HOC: (62, 20 * 6)
            # STFT: (62, 5)
            # HHS: (62, 5 * 2)
            clip_sample = trial_samples[:, clip_id]

            t_eeg = clip_sample
            if not transform is None:
                t_eeg = transform(eeg=clip_sample)['eeg']

            clip_id = f'{file_name}_{write_pointer}'
            write_pointer += 1

            # record meta info for each signal
            record_info = {'clip_id': clip_id}
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


def mped_feature_constructor(root_path: str = './EEG_feature',
                             feature: list = ['PSD'],
                             num_channel: int = 62,
                             before_trial: Union[None, Callable] = None,
                             transform: Union[None, Callable] = None,
                             after_trial: Union[Callable, None] = None,
                             io_path: str = './io/mped_feature',
                             io_size: int = 10485760,
                             io_mode: str = 'lmdb',
                             num_worker: int = 0,
                             verbose: bool = True) -> None:
    avaliable_features = os.listdir(
        root_path)  # ['HHS', 'Hjorth', 'PSD', 'STFT', 'HOC']
    assert set(feature).issubset(
        set(avaliable_features)
    ), 'The features supported by the MPEDFeature dataset are HHS, Hjorth, PSD, STFT, HOC. The input features are not a subset of the list of supported features.'

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
    file_list = os.listdir(os.path.join(root_path, avaliable_features[0]))

    if verbose:
        # show process bar
        pbar = tqdm(total=len(file_list))
        pbar.set_description("[MPED FEATURE]")

    if num_worker > 1:
        manager = Manager()
        queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
        io_consumer_process = Process(target=io_consumer,
                                      args=(eeg_io.write_eeg,
                                            info_io.write_info, queue),
                                      daemon=True)
        io_consumer_process.start()

        partial_mp_fn = partial(transform_producer,
                                root_path=root_path,
                                feature=feature,
                                num_channel=num_channel,
                                before_trial=before_trial,
                                transform=transform,
                                after_trial=after_trial,
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
