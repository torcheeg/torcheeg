from typing import Callable, Dict, List, Tuple, Union

import mne

from ..functional.mne import mne_constructor
from .base_dataset import BaseDataset


class MNEDataset(BaseDataset):
    r'''
    A generic EEG analysis dataset that allows creating datasets from :obj:`mne.Epochs`, and caches the generated results in a unified input and output format (IO). It is generally used to support custom datasets or datasets not yet provided by TorchEEG.

    :obj:`MNEDataset` allows a list of :obj:`mne.Epochs` and the corresponding description dictionary as input, and divides the signals in :obj:`mne.Epochs` into several segments according to the configuration information provided by the user. These segments will be annotated by the description dictionary elements and the event type annotated in :obj:`mne.Epochs`. Here is an example case shows the use of :obj:`MNEDataset`:

    .. code-block:: python

        # subject index and run index of mne.Epochs
        metadata_list = [{
            'subject': 1,
            'run': 3
        }, {
            'subject': 1,
            'run': 7
        }, {
            'subject': 1,
            'run': 11
        }]

        epochs_list = []
        for metadata in metadata_list:
            physionet_path = mne.datasets.eegbci.load_data(metadata['subject'],
                                                        metadata['run'],
                                                        update_path=False)[0]

            raw = mne.io.read_raw_edf(physionet_path, preload=True, stim_channel='auto')
            events, _ = mne.events_from_annotations(raw)
            picks = mne.pick_types(raw.info,
                                meg=False,
                                eeg=True,
                                stim=False,
                                eog=False,
                                exclude='bads')
            # init Epochs with raw EEG signals and corresponding event annotations
            epochs_list.append(mne.Epochs(raw, events, picks=picks))

        # split into chunks of 160 data points (1s)
        dataset = MNEDataset(epochs_list=epochs_list,
                            metadata_list=metadata_list,
                            chunk_size=160,
                            overlap=0,
                            num_channel=60,
                            io_path=io_path,
                            offline_transform=transforms.Compose(
                                [transforms.BandDifferentialEntropy()]),
                            online_transform=transforms.ToTensor(),
                            label_transform=transforms.Compose([
                                transforms.Select('event')
                            ]),
                            num_worker=2)
        print(dataset[0])
        # EEG signal (torch.Tensor[60, 4]),
        # coresponding baseline signal (torch.Tensor[60, 4]),
        # label (int)

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            # subject index and run index of mne.Epochs
            metadata_list = [{
                'subject': 1,
                'run': 3
            }, {
                'subject': 1,
                'run': 7
            }, {
                'subject': 1,
                'run': 11
            }]

            epochs_list = []
            for metadata in metadata_list:
                physionet_path = mne.datasets.eegbci.load_data(metadata['subject'],
                                                            metadata['run'],
                                                            update_path=False)[0]

                raw = mne.io.read_raw_edf(physionet_path, preload=True, stim_channel='auto')
                events, _ = mne.events_from_annotations(raw)
                picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    stim=False,
                                    eog=False,
                                    exclude='bads')
                # init Epochs with raw EEG signals and corresponding event annotations
                epochs_list.append(mne.Epochs(raw, events, picks=picks))

            # split into chunks of 160 data points (1s)
            dataset = MNEDataset(epochs_list=epochs_list,
                                metadata_list=metadata_list,
                                chunk_size=160,
                                overlap=0,
                                num_channel=60,
                                io_path=io_path,
                                offline_transform=transforms.Compose(
                                    [transforms.BandDifferentialEntropy()]),
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Compose([
                                    transforms.Select('event')
                                ]),
                                num_worker=2)
            print(dataset[0])
            # EEG signal (torch.Tensor[60, 4]),
            # coresponding baseline signal (torch.Tensor[60, 4]),
            # label (int)

    Args:
        epochs_list (list): A list of :obj:`mne.Epochs`. :obj:`MNEDataset` will divide the signals in :obj:`mne.Epochs` into several segments according to the :obj:`chunk_size` and :obj:`overlap` information provided by the user. The divided segments will be transformed and cached in a unified input and output format (IO) for accessing.
        metadata_list (list): A list of dictionaries of the same length as :obj:`epochs_list`. Each of these dictionaries is annotated with meta-information about :obj:`mne.Epochs`, such as subject index, experimental dates, etc. These annotated meta-information will be added to the element corresponding to :obj:`mne.Epochs` for use as labels for the sample.
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. If set to -1, the EEG signal is not segmented, and the length of the chunk is the length of the event. (default: :obj:`-1`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. If set to -1, all electrodes are used (default: :obj:`-1`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a :obj:`mne.Epoch`.
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/deap`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval. Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)    
    '''
    def __init__(self,
                 epochs_list: List[mne.Epochs],
                 metadata_list: List[Dict],
                 chunk_size: int = -1,
                 overlap: int = 0,
                 num_channel: int = -1,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 io_path: str = './io/mne',
                 io_size: int = 10485760,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):
        mne_constructor(epochs_list=epochs_list,
                        metadata_list=metadata_list,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        num_channel=num_channel,
                        before_trial=before_trial,
                        transform=offline_transform,
                        after_trial=after_trial,
                        io_path=io_path,
                        io_size=io_size,
                        io_mode=io_mode,
                        num_worker=num_worker,
                        verbose=verbose)
        super().__init__(io_path=io_path,
                         io_size=io_size,
                         io_mode=io_mode,
                         in_memory=in_memory)

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.num_channel = num_channel
        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.before_trial = before_trial
        self.after_trial = after_trial
        self.num_worker = num_worker
        self.verbose = verbose

    def __getitem__(self, index: int) -> Tuple:
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
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
