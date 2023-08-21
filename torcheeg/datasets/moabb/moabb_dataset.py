import os
import mne
import logging
import numpy as np
import itertools
from typing import Any, Callable, Dict, Tuple, Union
import pandas as pd

from ..module.base_dataset import BaseDataset

from moabb.datasets.base import BaseDataset as _MOABBDataset
from moabb.paradigms.base import BaseParadigm as _MOABBParadigm

log = logging.getLogger(__name__)

class MOABBDataset(BaseDataset):
    '''
    Mother of all BCI Benchmarks (MoABB) aims at building a comprehensive benchmark of popular Brain-Computer Interface (BCI) algorithms applied on an extensive list of freely available EEG datasets. This class implements the conversion of MOABB datasets to TorchEEG datasets, allowing to import any MOABB datasets to use them like TorchEEG datasets.

    A tiny case shows the use of :obj:`MOABBDataset`:

    .. code-block:: python
        
        import torcheeg.datasets.moabb as moabb_dataset

        dataset = BNCI2014001()
        dataset.subject_list = [1, 2, 3]
        paradigm = LeftRightImagery()
        dataset = moabb_dataset.MOABBDataset(dataset=dataset,
                               paradigm=paradigm,
                               io_path='./io/moabb',
                               offline_transform=transforms.Compose(
                                   [transforms.BandDifferentialEntropy()]),
                               online_transform=transforms.ToTensor(),
                               label_transform=transforms.Compose([
                                   transforms.Select('label')
                               ]))
    
    Args:
        dataset (MOABBDataset): an instance of :obj:`BaseDataset` defined by moabb.
        paradigm (MOABBParadigm): an instance of :obj:`BaseParadigm` defined by moabb.
        chunk_size (int): The length of each EEG sample. If set to :obj:`-1`, the length of each EEG sample is the same as the length of the EEG signal in event. (default: :obj:`-1`)
        overlap (int): The overlap between two adjacent EEG samples. (default: :obj:`0`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as filter, reference, moving average and etc. Its input is a :obj:`mne.io.Raw` object, and its output should also be a :obj:`mne.io.Raw` object. (default: :obj:`None`)
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/moabb`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        download_path (str): The path to download raw data. If set to :obj:`None`, the raw data will be downloaded to :obj:`f'{self.io_path}/raw'`. If the path already exists, the download will be skipped. (default: :obj:`None`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)
    '''
    def __init__(self,
                 dataset: _MOABBDataset,
                 paradigm: _MOABBParadigm,
                 chunk_size: int = -1,
                 overlap: int = 0,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: str = './io/moabb',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 download_path: str = None,
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False,
                 **kwargs):
        # pass all arguments to super class
        params = {
            'dataset': dataset,
            'paradigm': paradigm,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
            'after_session': after_session,
            'after_subject': after_subject,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'download_path': download_path,
            'num_worker': num_worker,
            'verbose': verbose,
            'in_memory': in_memory
        }
        params.update(kwargs)

        if not paradigm.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)
        
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def process_record(file: Any = None,
                       chunk_size: int = -1,
                       overlap: int = 0,
                       offline_transform: Union[None, Callable] = None,
                       dataset: _MOABBDataset = None,
                       paradigm: _MOABBParadigm = None,
                       before_trial: Union[None, Callable] = None,
                       after_trial: Union[None, Callable] = None,
                       **kwargs):

        subject_id, session_id = file

        write_pointer = 0
        session_signal = dataset.get_data(
            subjects=[subject_id])[subject_id][session_id]

        for run_id, run_signal in session_signal.items():
            if before_trial is not None:
                run_signal = before_trial(run_signal)

            proc = paradigm.process_raw(run_signal,
                                        dataset,
                                        return_epochs=False)
            if proc is None:
                # this mean the run did not contain any selected event
                # go to next
                continue

            trial_queue = []
            roi_signals, labels, metadatas = proc
            for roi_id, (roi_signal, label, metadata) in enumerate(
                    zip(roi_signals, labels, metadatas.iterrows())):

                start_at = 0
                if chunk_size <= 0:
                    chunk_size = roi_signal.shape[1] - start_at
                end_at = start_at + chunk_size

                step = chunk_size - overlap

                while end_at <= roi_signal.shape[1]:
                    clip_sample = roi_signal[:, start_at:end_at]

                    t_eeg = clip_sample
                    if not offline_transform is None:
                        t = offline_transform(eeg=t_eeg)
                        t_eeg = t['eeg']

                    clip_id = f'{subject_id}_{session_id}_{write_pointer}'
                    write_pointer += 1

                    record_info = {
                        'subject_id': subject_id,
                        'session_id': session_id,
                        'trial_id': run_id,
                        'roi_id': roi_id,
                        'clip_id': clip_id,
                        'label': label,
                        'start_at': start_at,
                        'end_at': end_at,
                        'metadata': metadata[1].to_dict()
                    }
                    write_pointer += 1
                    start_at = start_at + step
                    end_at = start_at + chunk_size

                    if after_trial is not None:
                        trial_queue.append({
                            'eeg': t_eeg,
                            'key': clip_id,
                            'info': record_info
                        })
                    else:
                        yield {
                            'eeg': t_eeg,
                            'key': clip_id,
                            'info': record_info
                        }

            if len(trial_queue) and after_trial is not None:
                trial_queue = after_trial(trial_queue)
                for obj in trial_queue:
                    assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
                    yield obj

    def set_records(self,
                    dataset: _MOABBDataset,
                    io_path: str = './io/moabb',
                    download_path: str = None,
                    **kwargs):
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

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        signal = eeg
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=eeg)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'dataset': self.dataset,
                'paradigm': self.paradigm,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
