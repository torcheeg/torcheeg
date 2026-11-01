import itertools
import logging
import os
from typing import Any, Callable, Dict, Tuple, Union

import mne
import numpy as np
import pandas as pd

from ...utils import LazyLoader
from ..base_data import BaseData
from ..cached_dataset import CachedDataset

_MOABBDataset = LazyLoader(
    'moabb.datasets.base', install_hint='Install with `pip install moabb`.').BaseDataset
_MOABBParadigm = LazyLoader(
    'moabb.paradigms.base', install_hint='Install with `pip install moabb`.').BaseParadigm

_log = logging.getLogger('torcheeg')


class MOABBData(BaseData):
    """Read and preprocess records for :class:`MOABBDataset`."""

    def __init__(self,
                 dataset: _MOABBDataset,
                 paradigm: _MOABBParadigm,
                 chunk_size: int = -1,
                 overlap: int = 0,
                 download_path: str = None,
                 transform: Union[None, Callable] = None,
                 **kwargs):

        if download_path is None:
            from ...utils import get_random_dir_path
            download_path = get_random_dir_path(dir_prefix='moabb_raw')
        if not paradigm.is_valid(dataset):
            raise ValueError(
                f'Dataset {dataset.code} is not valid for paradigm')

        super().__init__(
            transform=transform,
            dataset=dataset,
            paradigm=paradigm,
            chunk_size=chunk_size,
            overlap=overlap,
            download_path=download_path,
            **kwargs
        )

    def process_record(self, file: Any = None, chunk_size: int = None, overlap: int = None, dataset: _MOABBDataset = None, paradigm: _MOABBParadigm = None, **kwargs):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if dataset is None:
            dataset = self.dataset
        if paradigm is None:
            paradigm = self.paradigm
        subject_id, session_id = file
        session_signal = dataset.get_data(subjects=[subject_id])[
            subject_id][session_id]
        for run_id, run_signal in session_signal.items():
            proc = paradigm.process_raw(
                run_signal, dataset, return_epochs=False)
            if proc is None:
                continue
            trial_queue = []
            roi_signals, labels, metadatas = proc
            for roi_id, (roi_signal, label, metadata) in enumerate(zip(roi_signals, labels, metadatas.iterrows())):
                start_at = 0
                if chunk_size <= 0:
                    chunk_size = roi_signal.shape[1] - start_at
                end_at = start_at + chunk_size
                step = chunk_size - overlap
                while end_at <= roi_signal.shape[1]:
                    clip_sample = roi_signal[:, start_at:end_at]
                    t_eeg = clip_sample
                    record_info = {'subject_id': subject_id, 'session_id': session_id, 'trial_id': run_id, 'roi_id': roi_id,
                                   'label': label, 'start_at': start_at, 'end_at': end_at, 'metadata': metadata[1].to_dict()}
                    start_at = start_at + step
                    end_at = start_at + chunk_size
                    yield {'eeg': t_eeg, 'info': record_info}

    def set_records(self,
                    dataset: _MOABBDataset = None,
                    io_path: str = '.torcheeg/io/moabb',
                    download_path: str = None,
                    **kwargs):
        if dataset is None:
            dataset = self.dataset
        if download_path is None:
            download_path = self.download_path
        subject_id_list = dataset.subject_list
        if download_path is None:
            download_path = os.path.join(io_path, 'raw')
        if not os.path.exists(download_path):
            dataset.download(subject_list=subject_id_list,
                             path=download_path,
                             verbose=False)

        subject_id = subject_id_list[0]
        session_id_list = list(
            dataset.get_data(subjects=[subject_id])[subject_id].keys())

        # prod of subject_list and session_id_list
        subject_id_session_id_list = list(
            itertools.product(subject_id_list, session_id_list))
        return subject_id_session_id_list

    def read_record(self, record, **kwargs):
        return {'file': record}


class MOABBDataset(CachedDataset):

    r"""
    Mother of all BCI Benchmarks (MoABB) aims at building a comprehensive benchmark of popular Brain-Computer Interface (BCI) algorithms applied on an extensive list of freely availabel EEG datasets. This class implements the conversion of MOABB datasets to TorchEEG datasets, allowing to import any MOABB datasets to use them like TorchEEG datasets.

    A tiny case shows the use of :obj:`MOABBDataset`:

    .. code-block:: python

        from moabb.datasets import BNCI2014001
        from moabb.paradigms import LeftRightImagery

        from torcheeg import transforms
        import torcheeg.datasets.moabb as moabb_dataset

        dataset = BNCI2014001()
        dataset.subject_list = [1, 2, 3]
        paradigm = LeftRightImagery()


        def transform_fn(sample):
            eeg_transform = transforms.Compose([
                transforms.BandDifferentialEntropy(),
                transforms.ToTensor(),
            ])
            label_transform = transforms.Compose([
                transforms.Select('label'),
            ])
            return {
                "eeg": eeg_transform(sample["eeg"]),
                "label": label_transform(sample["info"]),
            }


        dataset = moabb_dataset.MOABBDataset(
            dataset=dataset, paradigm=paradigm, io_path='./io/moabb').with_transform(transform_fn)

    Args:
        dataset (MOABBDataset): an instance of :obj:`BaseDataset` defined by moabb.
        paradigm (MOABBParadigm): an instance of :obj:`BaseParadigm` defined by moabb.
        chunk_size (int): The length of each EEG sample. If set to :obj:`-1`, the length of each EEG sample is the same as the length of the EEG signal in event. (default: :obj:`-1`)
        overlap (int): The overlap between two adjacent EEG samples. (default: :obj:`0`)



        transform (Callable, optional): Sample dictionary transform applied before storage. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`./io/moabb`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        download_path (str): The path to download raw data. If set to :obj:`None`, the raw data will be downloaded to :obj:`f'{self.io_path}/raw'`. If the path already exists, the download will be skipped. (default: :obj:`None`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
    """

    data_class = MOABBData

    def __init__(self, dataset: _MOABBDataset, paradigm: _MOABBParadigm, chunk_size: int = -1, overlap: int = 0, transform: Union[None, Callable] = None, io_path: Union[None, str] = None, io_size: int = 1048576, io_mode: str = 'lmdb', download_path: str = None, num_worker: int = 0, verbose: bool = True, **kwargs):
        if download_path is None:
            from ...utils import get_random_dir_path
            if io_path is None:
                io_path = get_random_dir_path(dir_prefix='datasets')
            download_path = os.fspath(io_path) + '_raw'
        params = {'dataset': dataset, 'paradigm': paradigm, 'chunk_size': chunk_size, 'overlap': overlap, 'transform': transform,
                  'io_path': io_path, 'io_size': io_size, 'io_mode': io_mode, 'download_path': download_path, 'num_worker': num_worker, 'verbose': verbose}
        params.update(kwargs)
        super().__init__(**params)
        self.__dict__.update(params)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'dataset': self.dataset, 'paradigm': self.paradigm, 'transform': self.transform, 'verbose': self.verbose, 'io_size': self.io_size})
