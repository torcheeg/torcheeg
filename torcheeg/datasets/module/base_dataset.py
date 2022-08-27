import math
import os
import shutil
import pandas as pd
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, Dict, List, Union

from torch.utils.data import Dataset
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024


def copyfile(src, dst, *, follow_symlinks=True):
    if shutil._samefile(src, dst):
        raise shutil.SameFileError("{!r} and {!r} are the same file".format(
            src, dst))

    for fn in [src, dst]:
        try:
            st = os.stat(fn)
        except OSError:
            pass
        else:
            if shutil.stat.S_ISFIFO(st.st_mode):
                raise shutil.SpecialFileError("`%s` is a named pipe" % fn)

    if not follow_symlinks and os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        size = os.stat(src).st_size
        with open(src, 'rb') as fsrc:
            with open(dst, 'wb') as fdst:
                copyfileobj(fsrc, fdst, total=size)
    return dst


def copyfileobj(fsrc, fdst, total, length=16 * 1024):
    pbar = tqdm(total=total)
    pbar.set_description("[FROM EXISTING]")
    while True:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        pbar.update(len(buf))


def copy_with_progress(src, dst, *, follow_symlinks=True):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    copyfile(src, dst, follow_symlinks=follow_symlinks)
    shutil.copymode(src, dst)
    return dst


def split_df_by_size(df, chunk_size=100):
    chunks = list()
    num_chunks = math.ceil(len(df) / chunk_size)
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def split_df_by_column(df, column_name='baseline_id'):
    chunks = list()
    column_uniques = df[column_name].unique()
    for column in column_uniques:
        chunks.append(df[df[column_name].isin([column])])
    return chunks


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


def basic_producer(chunk: pd.DataFrame, transform: Callable,
                   read_eeg_fn: Callable, queue: Queue):

    last_baseline_id = None
    last_baseline_sample = None

    for i, row in chunk.iterrows():
        clip_id = row['clip_id']
        clip_sample = read_eeg_fn(clip_id)

        if 'baseline_id' in row:
            baseline_id = row['baseline_id']

            if last_baseline_id == baseline_id:
                trial_baseline_sample = last_baseline_sample
            else:
                trial_baseline_sample = read_eeg_fn(baseline_id)

            t = transform(eeg=clip_sample, baseline=trial_baseline_sample)
            t_eeg = t['eeg']
            t_baseline = t['baseline']

            if not last_baseline_id == baseline_id:
                queue.put({'eeg': t_baseline, 'key': baseline_id})

                last_baseline_id = baseline_id
                last_baseline_sample = trial_baseline_sample

            queue.put({'eeg': t_eeg, 'key': clip_id})

        else:
            t = transform(eeg=clip_sample)
            t_eeg = t['eeg']
            queue.put({'eeg': t_eeg, 'key': clip_id})


def reduce_producer(chunk: List[pd.DataFrame], transform: Callable,
                    read_eeg_fn: Callable, write_info_fn: Callable,
                    queue: Queue):

    for reduce_list in chunk:

        reduce_info = None

        reduce_clip_list = []
        reduce_baseline_list = []

        reduce_list = reduce_list.sort_values("clip_id", inplace=False)

        for i, row in reduce_list.iterrows():
            if reduce_info is None:
                reduce_info = row.to_dict()
            reduce_clip_list.append(read_eeg_fn(row['clip_id']))
            if 'baseline_id' in row:
                reduce_baseline_list.append(read_eeg_fn(row['baseline_id']))

        reduce_clip = transform(reduce_clip_list)
        queue.put({'eeg': reduce_clip, 'key': reduce_info['clip_id']})

        if len(reduce_baseline_list):
            reduce_baseline = transform(reduce_baseline_list)
            queue.put({
                'eeg': reduce_baseline,
                'key': reduce_info['baseline_id']
            })

        write_info_fn(reduce_info)


class BaseDataset(Dataset):
    channel_location_dict = {}
    adjacency_matrix = []

    def __init__(self, io_path: str):
        if not self.exist(io_path):
            raise RuntimeError(
                'Database IO does not exist, please regenerate database IO.')
        self.io_path = io_path

        meta_info_io_path = os.path.join(self.io_path, 'info.csv')
        eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

        info_io = MetaInfoIO(meta_info_io_path)
        self.eeg_io = EEGSignalIO(eeg_signal_io_path)

        self.info = info_io.read_all()

    def exist(self, io_path: str) -> bool:
        meta_info_io_path = os.path.join(io_path, 'info.csv')
        eeg_signal_io_path = eeg_signal_io_path = os.path.join(io_path, 'eeg')

        return os.path.exists(meta_info_io_path) and os.path.exists(
            eeg_signal_io_path)

    def __getitem__(self, index: int) -> any:
        raise NotImplementedError(
            "Method __getitem__ is not implemented in class " +
            self.__class__.__name__)

    def __len__(self):
        return len(self.info)

    def __copy__(self) -> 'BaseDataset':
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

        result.eeg_io = EEGSignalIO(eeg_signal_io_path)
        return result

    @property
    def repr_body(self) -> Dict:
        return {'io_path': self.io_path}

    @property
    def repr_tail(self) -> Dict:
        return {'length': self.__len__()}

    def __repr__(self) -> str:
        # init info
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            # line end
            if i:
                format_string += ','
            format_string += '\n'
            # str param
            if isinstance(v, str):
                format_string += f"    {k}='{v}'"
            else:
                format_string += f"    {k}={v}"
        format_string += '\n)'
        # other info
        format_string += '\n'
        for i, (k, v) in enumerate(self.repr_tail.items()):
            if i:
                format_string += ', '
            format_string += f'{k}={v}'
        return format_string

    @staticmethod
    def from_existing(dataset,
                      io_path: str,
                      offline_transform: Union[None, Callable] = None,
                      num_worker: int = 0,
                      verbose: bool = True,
                      cache_size: int = 64 * 1024 * 1024 * 1024,
                      chunk_size_for_worker: int = 100,
                      chunk_column_for_worker: str = 'baseline_id',
                      **args):
        meta_info_io_path = os.path.join(io_path, 'info.csv')
        eeg_signal_io_path = eeg_signal_io_path = os.path.join(io_path, 'eeg')

        exist_io = os.path.exists(meta_info_io_path) and os.path.exists(
            eeg_signal_io_path)

        if not exist_io:
            os.makedirs(io_path, exist_ok=True)

            # copy info file
            shutil.copy(os.path.join(dataset.io_path, 'info.csv'),
                        os.path.join(io_path, 'info.csv'))
            # copy dataset
            if not offline_transform is None:
                eeg_io = EEGSignalIO(eeg_signal_io_path, cache_size=cache_size)
                if chunk_column_for_worker in dataset.info.columns:
                    chunk_for_worker = split_df_by_column(
                        dataset.info, chunk_column_for_worker)
                else:
                    chunk_for_worker = split_df_by_size(dataset.info,
                                                        chunk_size_for_worker)

                if verbose:
                    # show process bar
                    pbar = tqdm(total=len(chunk_for_worker))
                    pbar.set_description("[FROM EXISTING]")
                if num_worker > 1:
                    manager = Manager()
                    queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)

                    partial_mp_fn = partial(basic_producer,
                                            transform=offline_transform,
                                            read_eeg_fn=dataset.eeg_io.read_eeg,
                                            queue=queue)

                    io_consumer_process = Process(target=io_consumer,
                                                  args=(eeg_io.write_eeg,
                                                        queue),
                                                  daemon=True)
                    io_consumer_process.start()

                    for _ in Pool(num_worker).imap(partial_mp_fn,
                                                   chunk_for_worker):
                        if verbose:
                            pbar.update(1)

                    queue.put(None)

                    io_consumer_process.join()
                    io_consumer_process.close()

                else:
                    for chunk in chunk_for_worker:
                        basic_producer(chunk=chunk,
                                       transform=offline_transform,
                                       read_eeg_fn=dataset.eeg_io.read_eeg,
                                       queue=SingleProcessingQueue(
                                           eeg_io.write_eeg))
                        if verbose:
                            pbar.update(1)

                if verbose:
                    pbar.close()
                    print('Please wait for the writing process to complete...')
            else:
                if verbose:
                    os.makedirs(os.path.join(io_path, 'eeg'), exist_ok=True)
                    copy_with_progress(
                        os.path.join(dataset.io_path, 'eeg', 'data.mdb'),
                        os.path.join(io_path, 'eeg', 'data.mdb'))
                else:
                    shutil.copytree(os.path.join(dataset.io_path, 'eeg'),
                                    os.path.join(io_path, 'eeg'))

        return type(dataset)(io_path=io_path,
                             offline_transform=offline_transform,
                             num_worker=num_worker,
                             verbose=verbose,
                             cache_size=cache_size,
                             **args)

    @staticmethod
    def reduce_from_existing(dataset,
                             io_path: str,
                             reduce_fn: Callable,
                             reduce_by: str = 'epoch_id',
                             verbose: bool = True,
                             cache_size: int = 64 * 1024 * 1024 * 1024,
                             num_worker: int = 0,
                             chunk_size_for_worker: int = 100,
                             **args):
        meta_info_io_path = os.path.join(io_path, 'info.csv')
        eeg_signal_io_path = eeg_signal_io_path = os.path.join(io_path, 'eeg')

        exist_io = os.path.exists(meta_info_io_path) and os.path.exists(
            eeg_signal_io_path)

        if not exist_io:
            os.makedirs(io_path, exist_ok=True)

            info_io = MetaInfoIO(meta_info_io_path)
            eeg_io = EEGSignalIO(eeg_signal_io_path, cache_size=cache_size)

            reduce_list = split_df_by_column(dataset.info, reduce_by)
            chunk_for_worker = [
                reduce_list[i:i + chunk_size_for_worker]
                for i in range(0, len(reduce_list), chunk_size_for_worker)
            ]

            if verbose:
                # show process bar
                pbar = tqdm(total=len(chunk_for_worker))
                pbar.set_description("[REDUCE]")

            if num_worker > 1:
                manager = Manager()
                queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
                io_consumer_process = Process(target=io_consumer,
                                              args=(eeg_io.write_eeg, queue),
                                              daemon=True)
                io_consumer_process.start()

                partial_mp_fn = partial(reduce_producer,
                                        transform=reduce_fn,
                                        read_eeg_fn=dataset.eeg_io.read_eeg,
                                        write_info_fn=info_io.write_info,
                                        queue=queue)

                for _ in Pool(num_worker).imap(partial_mp_fn, chunk_for_worker):
                    if verbose:
                        pbar.update(1)

                queue.put(None)

                io_consumer_process.join()
                io_consumer_process.close()
            else:
                for chunk in chunk_for_worker:
                    reduce_producer(chunk=chunk,
                                    transform=reduce_fn,
                                    read_eeg_fn=dataset.eeg_io.read_eeg,
                                    write_info_fn=info_io.write_info,
                                    queue=SingleProcessingQueue(
                                        eeg_io.write_eeg))
                    if verbose:
                        pbar.update(1)

            if verbose:
                pbar.close()

        return type(dataset)(io_path=io_path,
                             verbose=verbose,
                             cache_size=cache_size,
                             num_worker=num_worker,
                             **args)
