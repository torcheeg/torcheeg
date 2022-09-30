import numpy as np

from typing import Callable, Dict, Tuple, Union

from ..functional.numpy import numpy_constructor
from .base_dataset import BaseDataset


class NumpyDataset(BaseDataset):
    r'''
    A general dataset, this class converts EEG signals and annotations in Numpy format into dataset types, and caches the generated results in a unified input and output format (IO).

    A tiny case shows the use of :obj:`NumpyDataset`:

    .. code-block:: python

        # Mock 100 EEG samples. Each EEG signal contains a signal of length 1 s at a frequency of 128 sampled by 32 electrodes.
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
                               num_samples_per_worker=50)
        print(dataset[0])
        # EEG signal (torch.Tensor[32, 4]),
        # coresponding baseline signal (torch.Tensor[32, 4]),
        # label (int)

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            # Mock 100 EEG samples. Each EEG signal contains a signal of length 1 s at a frequency of 128 sampled by 32 electrodes.
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
                                num_samples_per_worker=50)
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
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/deap`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        num_samples_per_worker (str): The number of samples processed by each process. Once the specified number of samples are processed, the process will be destroyed and new processes will be created to perform new tasks. (default: :obj:`100`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        cache_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`64 * 1024 * 1024 * 1024`)
    
    '''
    def __init__(self,
                 X: np.ndarray,
                 y: Dict,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/numpy',
                 num_worker: int = 0,
                 num_samples_per_worker: int = 100,
                 verbose: bool = True,
                 cache_size: int = 64 * 1024 * 1024 * 1024):
        numpy_constructor(X=X,
                          y=y,
                          transform=offline_transform,
                          io_path=io_path,
                          num_worker=num_worker,
                          num_samples_per_worker=num_samples_per_worker,
                          verbose=verbose,
                          cache_size=cache_size)
        super().__init__(io_path)

        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.num_worker = num_worker
        self.num_samples_per_worker = num_samples_per_worker
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
                'num_worker': self.num_worker,
                'num_samples_per_worker': self.num_samples_per_worker,
                'verbose': self.verbose,
                'cache_size': self.cache_size
            })
