import os
from typing import Callable, Dict, Tuple, Union
from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path
from scipy.io import loadmat
import re

class  SanDiegoSSVEPDataset(BaseDataset):
    '''
    San Diego Square Joint Frequnecy-Phase Modulation SSVEP Dataset: lightweight dataset for studying SSVEP. For more information, please refer to the details below.


    - Author: Masaki Nakanishi et al.
    - Year: 2015
    - Download URL: https://www.kaggle.com/datasets/lzyuuu/ssvep-sandiego
    - Reference: Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung,"A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials," PLoS One, vol.10, no.10, e140703, 2015.
    - Stimulus: 12 different frequencies and phases of visual stimuli.
    - Signals: Electroencephalogram (8 channels at 256Hz). Training and testing sets have been divided for each participant (totally 10 participants) in original datasets .
    - label: The order of the stimulus frequencies in the EEG data is [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75] Hz which are labeled to range(0,12).
    
    
    In order to use this dataset, the download folder is required, containing the following files:
    
    - S01testEEG.mat
    - S01trainEEG.mat
    - S02testEEG.mat
    - ...
    - S010testEEG.mat
    - S010trainEEG.mat

    An example:
    
    .. code-block:: python

        from torcheeg.transforms import Select,BandSignal
        dataset = SanDiegoSSVEPDataset(root_path=r'D:\datasets\archive',
                                chunk_size=512,  #2 second
                                io_path= r'D:\datasets\tmp_out\sandiego',
                                offline_transform=BandSignal(sampling_rate=256,band_dict={'frequency_range':[8,16]}),
                                label_transform=Select('label')
        )
        print(dataset[0][0].shape) #EEG shape(1,8,512)
        print(dataset[0][1])  # label (int)

    Args:
        root_path (str): Downloaded data files (unzipped) dir.  (default: :obj:`'./archive'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`256`)
        online_transform (Callable, optional): The transformation of the EEG signals and baseline EEG signals. The input is a :obj:`np.ndarray`, and the ouput is used as the first and second value of each element in the dataset. (default: :obj:`None`)
        offline_transform (Callable, optional): The usage is the same as :obj:`online_transform`, but executed before generating IO intermediate results. (default: :obj:`None`)
        label_transform (Callable, optional): The transformation of the label. The input is an information dictionary, and the ouput is used as the third value of each element in the dataset. (default: :obj:`None`)
        before_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed before the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input of this hook function is a 2D EEG signal with shape (number of electrodes, number of data points), whose ideal output shape is also (number of electrodes, number of data points).
        after_trial (Callable, optional): The hook performed on the trial to which the sample belongs. It is performed after the offline transformation and thus typically used to implement context-dependent sample transformations, such as moving averages, etc. The input and output of this hook function should be a sequence of dictionaries representing a sequence of EEG samples. Each dictionary contains two key-value pairs, indexed by :obj:`eeg` (the EEG signal matrix) and :obj:`key` (the index in the database) respectively.
        io_path (str): The path to generated unified data IO, cached as an intermediate result. If set to None, a random path will be generated. (default: :obj:`None`)
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``, an exception will be raised and the user must close and reopen. (default: :obj:`1048576`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals. LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided. When io_mode is set to :obj:`pickle`, pickle-based persistence files are used. When io_mode is set to :obj:`memory`, memory are used. (default: :obj:`lmdb`)
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
    '''
    def __init__(self,
                 root_path = './archive',
                 chunk_size:int =256,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')
        
        
        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'chunk_size':chunk_size,
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
            'num_worker': num_worker,
            'verbose': verbose
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)


    @staticmethod
    def process_record(file,
                       root_path,
                       chunk_size,
                       offline_transform:Union[None, Callable],
                       before_trial: Union[None, Callable] = None,
                       **kwargs):
        
        file_path = os.path.join(root_path,file)
        subject_id = int( re.findall(r"S(\d+).*\.mat",file)[0] )
        train = True if re.findall(r"S\d+(train)EEG\.mat",file) else False
        
        data= loadmat(file_path)
        eeg = data['X'].transpose(-1,-2,-3)
        y = data['y'][0]
        record_global_info = {
            'subject_id':subject_id,
            'train':train,
        }
        for trial_id,eeg_clip in enumerate(eeg):
            if before_trial:
                before_trial(eeg_clip)
            eeg_clip = eeg_clip[:,:chunk_size]
            if not offline_transform is None:
                eeg_clip = offline_transform(eeg= eeg_clip)['eeg']
            label = int(y[trial_id]-1)
            trial_id = f"train_{trial_id}" if train else f"test_{trial_id}"
            clip_id = f"S{subject_id}_{trial_id}"
            record_info= {"clip_id":clip_id,'label':label,'trial_id':trial_id}
            record_info.update(record_global_info)
            yield {'eeg':eeg_clip,'key':clip_id,"info":record_info}
        
    def set_records(self,
                    root_path,
                    **kwargs):
        file_names = os.listdir(root_path)
        return file_names

    def __getitem__(self, index: int) -> Tuple:
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
            'root_path': self.root_path,
            'chunk_size':self.chunk_size,
            'online_transform': self.online_transform,
            'offline_transform': self.offline_transform,
            'label_transform': self.label_transform,
            'before_trial': self.before_trial,
            'after_trial': self.after_trial,
            'after_session':self.after_session,
            'after_subject': self.after_subject,
            'io_path': self.io_path,
            'io_size': self.io_size,
            'io_mode': self.io_mode,
            'num_worker': self.num_worker,
            'verbose':self.verbose
            })
