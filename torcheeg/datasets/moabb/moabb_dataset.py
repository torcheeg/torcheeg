import os
import itertools
from typing import Any, Callable, Dict, Tuple, Union

from ..module.base_dataset import BaseDataset

from moabb.datasets.base import BaseDataset as _MOABBDataset
from moabb.paradigms.base import BaseParadigm as _MOABBParadigm


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
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/moabb`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)
    '''
    def __init__(self,
                 dataset: _MOABBDataset,
                 paradigm: _MOABBParadigm,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/moabb',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False,
                 **kwargs):
        # pass all arguments to super class
        params = {
            'dataset': dataset,
            'paradigm': paradigm,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose,
            'in_memory': in_memory
        }
        params.update(kwargs)
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def _load_data(file: Any = None,
                   offline_transform: Union[None, Callable] = None,
                   dataset: _MOABBDataset = None,
                   paradigm: _MOABBParadigm = None,
                   **kwargs):

        if not paradigm.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)

        subject_id, session_id = file

        write_pointer = 0
        session_signal = dataset.get_data(
            subjects=[subject_id])[subject_id][session_id]

        for run_id, run_signal in session_signal.items():

            proc = paradigm.process_raw(run_signal,
                                        dataset,
                                        return_epochs=False)
            if proc is None:
                # this mean the run did not contain any selected event
                # go to next
                continue

            clip_signals, labels, metadatas = proc
            for clip_id, (clip_signal, label, metadata) in enumerate(
                    zip(clip_signals, labels, metadatas.iterrows())):
                t_eeg = clip_signal
                if not offline_transform is None:
                    t = offline_transform(eeg=clip_signal)
                    t_eeg = t['eeg']

                clip_id = f'{subject_id}_{session_id}_{write_pointer}'
                write_pointer += 1

                record_info = {
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'run_id': run_id,
                    'clip_id': clip_id,
                    'label': label,
                    'metadata': metadata[1].to_dict()
                }

                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

    @staticmethod
    def _set_files(dataset: _MOABBDataset,
                   io_path: str = './io/moabb',
                   **kwargs):
        subject_id_list = dataset.subject_list
        dataset.download(subject_list=subject_id_list,
                         path=os.path.join(io_path, 'raw'),
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
        eeg = self.read_eeg(eeg_index)

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
