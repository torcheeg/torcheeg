import os
from multiprocessing import Manager
from typing import Any, Callable, Dict, List, Tuple, Union

import mne
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from torcheeg.io import EEGSignalIO, MetaInfoIO

from .base_dataset import BaseDataset

MAX_QUEUE_SIZE = 1024


class MockLock():
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


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
        epochs_list (list): A list of :obj:`mne.Epochs`. :obj:`MNEDataset` will divide the signals in :obj:`mne.Epochs` into several segments according to the :obj:`chunk_size` and :obj:`overlap` information provided by the user. The divided segments will be transformed and cached in a unified input and output format (IO) for accessing. You can also pass through the path list of the mne.Epochs file (can be obtained by .save()).
        metadata_list (list): A list of dictionaries of the same length as :obj:`epochs_list`. Each of these dictionaries is annotated with meta-information about :obj:`mne.Epochs`, such as subject index, experimental dates, etc. These annotated meta-information will be added to the element corresponding to :obj:`mne.Epochs` for use as labels for the sample.
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. If set to -1, the EEG signal of a trial is used as a sample of a chunk. If set to -1, the EEG signal is not segmented, and the length of the chunk is the length of the event. (default: :obj:`-1`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used. If set to -1, all electrodes are used (default: :obj:`-1`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a :obj:`mne.Epoch`.
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/mne`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
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
        # pass all arguments to super class
        params = {
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial
        }
        self.__dict__.update(params)

        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = io_mode
        self.in_memory = in_memory
        self.num_worker = num_worker
        self.verbose = verbose

        # new IO
        if not self.exist(self.io_path):
            print(
                f'dataset does not exist at path {self.io_path}, generating files to path...'
            )
            # make the root dictionary
            os.makedirs(self.io_path, exist_ok=True)

            # init sub-folders
            meta_info_io_path = os.path.join(self.io_path, 'info.csv')
            eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

            MetaInfoIO(meta_info_io_path)
            EEGSignalIO(eeg_signal_io_path,
                        io_size=self.io_size,
                        io_mode=self.io_mode)

            if self.num_worker == 0:
                lock = MockLock()  # do nothing, just for compatibility
                for file in tqdm(self._set_files(epochs_list=epochs_list,
                                                 metadata_list=metadata_list,
                                                 io_path=io_path,
                                                 **params),
                                 disable=not self.verbose,
                                 desc="[PROCESS]"):
                    self._process_file(io_path=self.io_path,
                                       io_size=self.io_size,
                                       io_mode=self.io_mode,
                                       file=file,
                                       lock=lock,
                                       _load_data=self._load_data,
                                       **params)
            else:
                # lock for lmdb writter, LMDB only allows single-process writes
                manager = Manager()
                lock = manager.Lock()

                Parallel(n_jobs=self.num_worker)(
                    delayed(self._process_file)(io_path=io_path,
                                                io_size=io_size,
                                                io_mode=io_mode,
                                                file=file,
                                                lock=lock,
                                                _load_data=self._load_data,
                                                **params)
                    for file in tqdm(self._set_files(
                        epochs_list=epochs_list,
                        metadata_list=metadata_list,
                        io_path=io_path,
                        **params),
                                     disable=not self.verbose,
                                     desc="[PROCESS]"))

        print(
            f'dataset already exists at path {self.io_path}, reading from path...'
        )

        meta_info_io_path = os.path.join(self.io_path, 'info.csv')
        eeg_signal_io_path = os.path.join(self.io_path, 'eeg')

        info_io = MetaInfoIO(meta_info_io_path)
        self.eeg_io = EEGSignalIO(eeg_signal_io_path,
                                  io_size=self.io_size,
                                  io_mode=self.io_mode)

        self.info = info_io.read_all()

    @staticmethod
    def _load_data(file: Any = None,
                   chunk_size: int = -1,
                   overlap: int = 0,
                   num_channel: int = -1,
                   before_trial: Union[None, Callable] = None,
                   offline_transform: Union[None, Callable] = None,
                   after_trial: Union[None, Callable] = None,
                   **kwargs):
        mne.set_log_level('CRITICAL')

        tmp_path, metadata, block_id = file
        epochs = mne.read_epochs(tmp_path, preload=True)

        assert (epochs.tmax - epochs.tmin) * epochs.info[
            "sfreq"] >= chunk_size, f'chunk_size cannot be larger than (tmax - tmin) * sfreq. Here, tmax is set to {epochs.tmax}, tmin is set to {epochs.tmin}, and sfreq is {epochs.info["sfreq"]}. In the current configuration, chunk_size {chunk_size} is greater than {(epochs.tmax - epochs.tmin) * epochs.info["sfreq"]}!'

        if chunk_size <= 0:
            chunk_size = len(epochs.times)

        if num_channel == -1:
            num_channel = len(epochs.info['chs'])

        trial_event_index = epochs.events[:, 2]
        trial_start_at_list = epochs.events[:, 0]

        trial_end_at = len(epochs.times) - chunk_size

        clip_sample_start_at_list = np.arange(0, trial_end_at + 1,
                                              chunk_size - overlap)

        sample_events = [[clip_sample_start_at, chunk_size, -1]
                         for clip_sample_start_at in clip_sample_start_at_list]

        epoch_meta_info = metadata

        write_pointer = 0

        # for loop of trials
        for trial_id, trial in enumerate(epochs):
            # split sample from epochs
            start_at_list = clip_sample_start_at_list + trial_start_at_list[
                trial_id]
            end_at_list = clip_sample_start_at_list + trial_start_at_list[
                trial_id] + chunk_size
            event_index_list = len(sample_events) * [
                trial_event_index[trial_id]
            ]

            trial_samples = mne.Epochs(mne.io.RawArray(trial, epochs.info),
                                       sample_events,
                                       baseline=None,
                                       tmin=0,
                                       tmax=(chunk_size - 1) /
                                       epochs.info["sfreq"])
            trial_samples.drop_bad(reject=None, flat=None)
            if before_trial:
                trial_samples = before_trial(trial_samples)

            # for loop of samples
            trial_queue = []
            for i, trial_signal in enumerate(trial_samples.get_data()):
                t_eeg = trial_signal[:num_channel, :]
                if not offline_transform is None:
                    t = offline_transform(eeg=trial_signal[:num_channel, :])
                    t_eeg = t['eeg']

                clip_id = f'{block_id}_{write_pointer}'
                write_pointer += 1

                record_info = {
                    'trial_id': trial_id,
                    'start_at': start_at_list[i],
                    'end_at': end_at_list[i],
                    'event': event_index_list[i],
                    'clip_id': clip_id
                }
                record_info.update(epoch_meta_info)
                if after_trial:
                    trial_queue.append({
                        'eeg': t_eeg,
                        'key': clip_id,
                        'info': record_info
                    })
                else:
                    yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

            if len(trial_queue) and after_trial:
                trial_queue = after_trial(trial_queue)
                for obj in trial_queue:
                    assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
                    yield obj

    @staticmethod
    def _set_files(epochs_list: Union[List[str], List[mne.Epochs]],
                   metadata_list: List[Dict[str, Any]], io_path: str, **kwargs):
        epochs_metadata_block_id_list = []
        for block_id, (epochs,
                       metadata) in enumerate(zip(epochs_list, metadata_list)):
            if isinstance(epochs, str):
                epochs_path = epochs
            else:
                if not os.path.exists(os.path.join(io_path, 'tmp')):
                    os.makedirs(os.path.join(io_path, 'tmp'))
                epochs_path = os.path.join(io_path, 'tmp', f'{block_id}.epochs')
                epochs.save(epochs_path)

            epochs_metadata_block_id_list.append(
                (epochs_path, metadata, block_id))

        return epochs_metadata_block_id_list

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
