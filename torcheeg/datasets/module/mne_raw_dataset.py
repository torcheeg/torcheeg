from typing import Any, Callable, Dict, Tuple, Union, List
import mne
import numpy as np

from .base_dataset import BaseDataset
from ...utils import get_random_dir_path

class MNERawDataset(BaseDataset):
    '''
    Process a list of MNE Raw objects and corresponding information dictionaries. This dataset is particularly useful for working with pre-loaded MNE Raw objects, such as those obtained from various EEG datasets like Physionet EEG Motor Movement/Imagery Dataset.

    The dataset splits the continuous EEG data into epochs based on the specified chunk size and overlap. Each epoch is associated with the corresponding information from the info_list.

    .. code-block:: python

        import mne
        from torcheeg.datasets import MNERawDataset
        from torcheeg import transforms

        subject_id = 22
        event_codes = [5, 6, 9, 10, 13, 14]

        physionet_paths = mne.datasets.eegbci.load_data(
            subject_id, event_codes, update_path=False)

        # Load each of the files
        raw_list = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto')
                 for path in physionet_paths]

        info_list = [{"trial_id": event_code, "subject_id": subject_id}
                        for event_code in event_codes]

        dataset = MNERawDataset(raw_list=raw_list,
                                info_list=info_list,
                                chunk_size=500,
                                overlap=0,
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Select('trial_id'))

    Args:
        raw_list (List): A list of MNE Raw objects containing the EEG data.
        info_list (List): A list of dictionaries containing metadata for each Raw object. Each dictionary should correspond to the Raw object at the same index in raw_list.
        chunk_size (int): The size of each epoch in samples. (default: :obj:`3000`)
        overlap (int): The number of overlapping samples between consecutive epochs. (default: :obj:`0`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    '''
    def __init__(self,
                 raw_list: List,
                 info_list: List,
                 chunk_size: int = 3000,
                 overlap: int = 0,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 **kwargs):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        params = {
            'raw_list': raw_list,
            'info_list': info_list,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }
        params.update(kwargs)
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def process_record(file: Any = None,
                       offline_transform: Union[None, Callable] = None,
                       chunk_size: int = 500,
                       overlap: int = 0,
                       **kwargs):
        raw, info = file
        data, times = raw[:, :]
        
        # Calculate the step size
        step = chunk_size - overlap
        
        # Generate chunks
        for i in range(0, data.shape[1] - chunk_size + 1, step):
            chunk = data[:, i:i+chunk_size]
            
            if offline_transform is not None:
                chunk = offline_transform(eeg=chunk)['eeg']
            
            clip_id = f"{info['subject_id']}_{info['trial_id']}_{i}"
            
            record_info = {
                **info,
                'start_at': times[i],
                'end_at': times[i+chunk_size-1],
                'clip_id': clip_id
            }
            
            yield {'eeg': chunk, 'key': clip_id, 'info': record_info}

    def set_records(self, raw_list: List[mne.io.BaseRaw], info_list: List[Dict[str, Any]], **kwargs):
        return list(zip(raw_list, info_list))

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
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })