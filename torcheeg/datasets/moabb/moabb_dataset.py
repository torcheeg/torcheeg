import os
import mne
import moabb
from copy import copy
from numbers import Integral
from pathlib import Path
import logging
import numpy as np
from typing import Any, Callable, Dict, Tuple, Union

from ..module.base_dataset import BaseDataset
from ...utils import get_random_dir_path

from moabb.datasets.base import BaseDataset as _MOABBDataset
from moabb.paradigms.base import BaseParadigm as _MOABBParadigm

log = logging.getLogger('torcheeg')

class MOABBDataset(BaseDataset):
    '''
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
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`./io/moabb`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        download_path (str): The path for MOABB raw downloads. It must be outside the processed cache. If None, MOABB uses its configured data location. An explicit path is registered through MOABB's download API; existing source files are reused rather than removed. (default: :obj:`None`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
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
                 io_path: str = '.torcheeg/io/moabb',
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 download_path: str = None,
                 num_worker: int = 0,
                 verbose: bool = True,
                                  **kwargs):
        if (not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool)
                or chunk_size == 0 or chunk_size < -1):
            raise ValueError('chunk_size must be a positive integer or -1.')
        if (not isinstance(overlap, Integral) or isinstance(overlap, bool)
                or overlap < 0 or (chunk_size > 0 and overlap >= chunk_size)
                or (chunk_size == -1 and overlap != 0)):
            raise ValueError('overlap must be nonnegative and smaller than chunk_size; use 0 for full epochs.')
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        if download_path is not None:
            cache_path = Path(io_path).resolve()
            raw_path = Path(download_path).resolve()
            if raw_path == cache_path or cache_path in raw_path.parents:
                raise ValueError('download_path must be separate from io_path.')
        # Keep downloads outside the cache generation and preserve existing files.
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
            'verbose': verbose
        }
        params.update(kwargs)
        params['_backend_versions'] = {'moabb': moabb.__version__, 'mne': mne.__version__}

        if not paradigm.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)
        
        if download_path is not None and not os.path.exists(io_path):
            dataset.download(subject_list=list(dataset.subject_list),
                             path=download_path, update_path=True, verbose=False)
        super().__init__(**params)
        if 'run_id' not in self.info:
            # Legacy caches used trial_id for the run identifier.
            self.info['run_id'] = self.info['trial_id']
        for column in ('session_id', 'run_id'):
            self.info[column] = self.info[column].map(str)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def read_record(record: Any, dataset: _MOABBDataset,
                    paradigm: _MOABBParadigm,
                    before_trial: Union[None, Callable] = None,
                    **kwargs) -> Dict:
        source = dataset
        if before_trial is not None:
            # Preserve the source object and use MOABB's raw-dataset extension point.
            source = copy(dataset)

            def read_subject(subject):
                sessions = dataset.get_data(subjects=[subject])[subject]
                result = {}
                for session, runs in sessions.items():
                    result[session] = {}
                    for run, raw in runs.items():
                        transformed = before_trial(raw.copy())
                        if not isinstance(transformed, mne.io.BaseRaw):
                            raise TypeError('before_trial must return an MNE Raw object.')
                        result[session][run] = transformed
                return result

            source._get_single_subject_data = read_subject
        samples, labels, metadata = paradigm.get_data(
            source, subjects=[record], return_epochs=False)
        if not isinstance(samples, np.ndarray) or samples.ndim not in (3, 4):
            raise ValueError('MOABB must return epochs shaped (epochs, channels, samples[, filters]).')
        if len(samples) != len(labels) or len(samples) != len(metadata):
            raise ValueError('MOABB epochs, labels and metadata must have matching lengths.')
        if not {'subject', 'session', 'run'}.issubset(metadata.columns):
            raise ValueError('MOABB metadata must contain subject, session and run.')
        if metadata[['subject', 'session', 'run']].isna().any().any():
            raise ValueError('MOABB subject, session and run must not be missing.')
        if any(str(subject) != str(record) for subject in metadata['subject']):
            raise ValueError('MOABB returned metadata from an unexpected subject.')
        return {'samples': samples, 'labels': labels,
                'metadata': metadata.reset_index(drop=True)}

    @staticmethod
    def process_record(record: Any, samples: np.ndarray,
                       labels: np.ndarray, metadata: Any,
                       chunk_size: int = -1, overlap: int = 0,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        for roi_id, (signal, label) in enumerate(zip(samples, labels)):
            row = metadata.iloc[roi_id]
            size = signal.shape[1] if chunk_size == -1 else chunk_size
            step = size - overlap
            if size <= 0 or step <= 0:
                raise ValueError('An epoch must contain samples and a positive window step.')
            for start_at in range(0, signal.shape[1] - size + 1, step):
                end_at = start_at + size
                eeg = signal[:, start_at:end_at, ...]
                if offline_transform is not None:
                    eeg = offline_transform(eeg=eeg)['eeg']
                clip_id = f'{record}_{roi_id}_{start_at}'
                info = {
                    'subject_id': record,
                    'session_id': str(row['session']),
                    'run_id': str(row['run']),
                    'trial_id': f'{record}_{roi_id}',
                    'roi_id': roi_id,
                    'clip_id': clip_id,
                    'label': label,
                    'start_at': start_at,
                    'end_at': end_at,
                    'metadata': row.to_dict()
                }
                yield {'eeg': eeg, 'key': clip_id, 'info': info}

    def set_records(self, dataset: _MOABBDataset, **kwargs):
        # Public paradigm.get_data supplies each subject's actual sessions and runs.
        subjects = list(dataset.subject_list)
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError('Select a nonempty list of distinct MOABB subjects.')
        return subjects

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
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
