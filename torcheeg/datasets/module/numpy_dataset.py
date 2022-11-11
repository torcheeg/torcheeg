import numpy as np

from typing import Callable, Dict, Tuple, Union

from ..functional.numpy import numpy_constructor
from .base_dataset import BaseDataset


class NumpyDataset(BaseDataset):
    r'''
    A general dataset, this class converts EEG signals and annotations in Numpy format into dataset types, and caches the generated results in a unified input and output format (IO).

    A tiny case shows the use of :obj:`NumpyDataset`:

    .. code-block:: python

        # Mock 100 EEG samples. Each EEG signal contains a signal of length 1 s at a sampling rate of 128 sampled by 32 electrodes.
        X = np.random.randn(100, 32, 128)

        # Mock 100 labels, denoting valence and arousal of subjects during EEG recording.
        y = {
            'valence': np.random.randint(10, size=100),
            'arousal': np.random.randint(10, size=100)
        }
        dataset = NumpyDataset(X=X,
                               y=y,
                               io_path=io_path,
                               offline_transform=transforms.Compose(
                                   [transforms.BandDifferentialEntropy()]),
                               online_transform=transforms.ToTensor(),
                               label_transform=transforms.Compose([
                                   transforms.Select('valence'),
                                   transforms.Binary(5.0),
                               ]),
                               num_worker=2,
                               num_samples_per_trial=50)
        print(dataset[0])
        # EEG signal (torch.Tensor[32, 4]),
        # coresponding baseline signal (torch.Tensor[32, 4]),
        # label (int)

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            # Mock 100 EEG samples. Each EEG signal contains a signal of length 1 s at a sampling rate of 128 sampled by 32 electrodes.
            X = np.random.randn(100, 32, 128)

            # Mock 100 labels, denoting valence and arousal of subjects during EEG recording.
            y = {
                'valence': np.random.randint(10, size=100),
                'arousal': np.random.randint(10, size=100)
            }
            dataset = NumpyDataset(X=X,
                                y=y,
                                io_path=io_path,
                                offline_transform=transforms.Compose(
                                    [transforms.BandDifferentialEntropy()]),
                                online_transform=transforms.ToTensor(),
                                label_transform=transforms.Compose([
                                    transforms.Select('valence'),
                                    transforms.Binary(5.0),
                                ]),
                                num_worker=2,
                                num_samples_per_trial=50)
            print(dataset[0])
            # EEG signal (torch.Tensor[32, 4]),
            # coresponding baseline signal (torch.Tensor[32, 4]),
            # label (int)

    Args:
        X (np.ndarray): An array in :obj:`numpy.ndarray` format representing the EEG signal samples in the dataset. The shape of the array is :obj:`[num_sample, ...]` where :obj:`num_sample` is the number of samples.
        y (dict): A dictionary that records the labels corresponding to EEG samples. The keys of the dictionary represent the names of the labels, and the values are lists of labels whose length is consistent with the EEG signal samples.
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a :obj:`np.ndarray`, whose shape is (number of EEG samples per trial, ...).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/deap`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        num_samples_per_trial (str): The number of samples processed by each process. Once the specified number of samples are processed, the process will be destroyed and new processes will be created to perform new tasks. (default: :obj:`100`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        cache_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`64 * 1024 * 1024 * 1024`)
    
    '''
    def __init__(self,
                 X: np.ndarray,
                 y: Dict,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 io_path: str = './io/numpy',
                 num_worker: int = 0,
                 num_samples_per_trial: int = 100,
                 verbose: bool = True,
                 cache_size: int = 64 * 1024 * 1024 * 1024):
        numpy_constructor(X=X,
                          y=y,
                          before_trial=before_trial,
                          transform=offline_transform,
                          after_trial=after_trial,
                          io_path=io_path,
                          num_worker=num_worker,
                          num_samples_per_trial=num_samples_per_trial,
                          verbose=verbose,
                          cache_size=cache_size)
        super().__init__(io_path)

        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.before_trial = before_trial
        self.after_trial = after_trial
        self.num_worker = num_worker
        self.num_samples_per_trial = num_samples_per_trial
        self.verbose = verbose
        self.cache_size = cache_size

    def __getitem__(self, index: int) -> Tuple:
        info = self.info.iloc[index].to_dict()

        eeg_index = str(info['clip_id'])
        eeg = self.eeg_io.read_eeg(eeg_index)

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
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'num_samples_per_trial': self.num_samples_per_trial,
                'verbose': self.verbose,
                'cache_size': self.cache_size
            })
