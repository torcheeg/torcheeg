from typing import Callable, Union, Tuple, Dict

from ..base_dataset import BaseDataset
from ...functional.personal_identification.m3cv import m3cv_constructor


class M3CVDataset(BaseDataset):
    r'''
    A reliable EEG-based biometric system should be able to withstand changes in an individual's mental state (cross-task test) and still be able to successfully identify an individual after several days (cross-session test). The authors built an EEG dataset M3CV with 106 subjects, two sessions of experiment on different days, and multiple paradigms. Ninety-five of the subjects participated in two sessions of the experiments, separated by more than 6 days. The experiment includes 6 common EEG experimental paradigms including resting state, sensory and cognitive task, and brain-computer interface.
    
    - Author: Huang et al.
    - Year: 2022
    - Download URL: https://aistudio.baidu.com/aistudio/datasetdetail/151025/0
    - Signals: Electroencephalogram (64 channels and one marker channel at 250Hz).

    In order to use this dataset, the download dataset folder :obj:`aistudio` is required, containing the following files:
    
    - Calibration_Info.csv
    - Enrollment_Info.csv
    - Testing_Info.csv
    - Calibration (unzipped Calibration.zip)
    - Testing (unzipped Testing.zip)
    - Enrollment (unzipped Enrollment.zip)

    An example dataset for CNN-based methods:

    .. code-block:: python
    
        dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(M3CV_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('SubjectID'),
                                  transforms.StringToNumber()
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1000, 9, 9]),
        # coresponding baseline signal (torch.Tensor[1000, 9, 9]),
        # label (int)

    Another example dataset for CNN-based methods:

    .. code-block:: python

        dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('SubjectID'),
                                  transforms.StringToNumber()
                              ]))
        print(dataset[0])
        # EEG signal (torch.Tensor[1, 65, 1000]),
        # coresponding baseline signal (torch.Tensor[1, 65, 1000]),
        # label (int)

    An example dataset for GNN-based methods:

    .. code-block:: python
    
        dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              online_transform=transforms.Compose([
                                  transforms.ToG(M3CV_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('SubjectID'),
                                  transforms.StringToNumber()
                              ]))
        print(dataset[0])
        # EEG signal (torch_geometric.data.Data),
        # coresponding baseline signal (torch_geometric.data.Data),
        # label (int)

    In particular, TorchEEG utilizes the producer-consumer model to allow multi-process data preprocessing. If your data preprocessing is time consuming, consider increasing :obj:`num_worker` for higher speedup. If running under Windows, please use the proper idiom in the main module:

    .. code-block:: python
    
        if __name__ == '__main__':
            dataset = M3CVDataset(io_path=f'./m3cv',
                              root_path='./aistudio',
                              online_transform=transforms.Compose([
                                  transforms.ToG(M3CV_ADJACENCY_MATRIX)
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('SubjectID'),
                                  transforms.StringToNumber()
                              ]),
                              num_worker=4)
            print(dataset[0])
            # EEG signal (torch_geometric.data.Data),
            # coresponding baseline signal (torch_geometric.data.Data),
            # label (int)

    Args:
        root_path (str): Downloaded data files in pickled python/numpy (unzipped aistudio.zip) formats (default: :obj:`'./aistudio'`)
        subset (str): In the competition, the M3CV dataset is splited into the Enrollment set, Calibration set, and Testing set. Please specify the subset to use, options include Enrollment, Calibration and Testing. (default: :obj:`'Enrollment'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`1000`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        channel_num (int): Number of channels used, of which the first 32 channels are EEG signals. (default: :obj:`64`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        io_path (str): The path to generated unified data IO, cached as an intermediate result. (default: :obj:`./io/m3cv`)
        num_worker (str): How many subprocesses to use for data processing. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        cache_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`64 * 1024 * 1024 * 1024`)
    
    '''
    def __init__(self,
                 root_path: str = './aistudio',
                 subset: str = 'Enrollment',
                 chunk_size: int = 1000,
                 overlap: int = 0,
                 channel_num: int = 64,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: str = './io/m3cv',
                 num_worker: int = 0,
                 verbose: bool = True,
                 cache_size: int = 64 * 1024 * 1024 * 1024):
        m3cv_constructor(root_path=root_path,
                         subset=subset,
                         chunk_size=chunk_size,
                         overlap=overlap,
                         channel_num=channel_num,
                         transform=offline_transform,
                         io_path=io_path,
                         num_worker=num_worker,
                         verbose=verbose,
                         cache_size=cache_size)
        super().__init__(io_path)

        self.root_path = root_path
        self.subset = subset
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.channel_num = channel_num
        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.num_worker = num_worker
        self.verbose = verbose
        self.cache_size = cache_size

        assert subset in [
            'Enrollment', 'Calibration', 'Testing'
        ], f"Unavailable subset name {subset}, and available options include 'Enrollment', 'Calibration', and 'Testing'."

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
                'root_path': self.root_path,
                'subset': self.subset,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'channel_num': self.channel_num,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'cache_size': self.cache_size
            })
