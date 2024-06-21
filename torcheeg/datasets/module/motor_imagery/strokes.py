import os
from typing import Any, Callable, Dict, Tuple, Union
from ..base_dataset import BaseDataset
from ....utils import get_random_dir_path
from scipy.io import loadmat
import re
import pandas as pd
import mne


class StrokePatientsMIDataset(BaseDataset):
    '''
    An EEG motor imagery dataset for brain computer interface in acute stroke patients. For more detail, please refer to following information.

    - Author: Haijie Liu et al.
    - Year: 2024
    - Download URL: https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035/5
    - Reference: Liu, Haijie, et al. "An EEG motor imagery dataset for brain computer interface in acute stroke patients." Scientific Data 11.1 (2024): 131.
    - Stimulus: A video of gripping motion is played on the computer, which leads the patient imagine grabbing the ball. This video stays playing for 4 s. Patients only imagine one hand movement.
    - Signals: Electroencephalogram (30 channels at 500Hz sample rate,. Electrooculography (including horizontal and vertical EOG) (totally 50 participants).
    - label: left hand and right hand.
    
    In order to use this dataset, the downlowd root path is required, containing the following files and directories:
    
    - sourcedata (dir)
    - edffile (dir)
    - task-motor-imagery_events.tsv
    - ...
    
    An example:
    
    .. code-block:: python

        from torcheeg.transforms import Select,BandSignal
        dataset = StrokePatientsMIDataset(root_path='your unzipped root path',
                                chunk_size=500,  # 1 second
                                overlap = 0,
                                io_path= './stroke_dataset_cache',
                                offline_transform=BandSignal(sampling_rate=500,band_dict={'frequency_range':[8,40]}),
                                label_transform=Select('label')
        )
        print(dataset[0][0].shape) #EEG shape(1,30,500)
        print(dataset[0][1])  # label (int)

    Args:
        root_path (str): Downloaded data files (unzipped) dir.  (default: :obj:`'./StrokePatientsMIDataset'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`500`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
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
                 root_path='./StrokePatientsMIDataset',
                 chunk_size: int = 500,
                 overlap: int = 0,
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
                 verbose: bool = True,
                ):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        self.subjects_info = pd.read_csv(os.path.join(root_path,
                                                      'participants.tsv'),
                                         sep='\t')
        self.electodes_info = pd.read_csv(os.path.join(
            root_path, "task-motor-imagery_electrodes.tsv"),
                                          sep='\t')
        electodes_info2 = pd.read_csv(os.path.join(
            root_path, "task-motor-imagery_channels.tsv"),
                                      sep='\t')
        self.electodes_info = pd.merge(self.electodes_info,
                                       electodes_info2,
                                       on='name',
                                       how='outer')
        refence = {
            'name': 'CPz',
            'type': 'EEG',
            'status': 'good',
            'status_description': 'refence'
        }

        insert_index = self.electodes_info.index[
            self.electodes_info.index.get_loc(17)]
        self.electodes_info = pd.concat([
            self.electodes_info.iloc[:insert_index],
            pd.DataFrame([refence], index=[insert_index]),
            self.electodes_info.iloc[insert_index:]
        ])
        self.electodes_info.index = range(len(self.electodes_info))

        self.events_info = pd.read_csv(os.path.join(
            root_path, 'task-motor-imagery_events.tsv'),
                                       sep='\t')
        # pass all arguments to super class
        params = {
            'root_path': root_path,
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
            'num_worker': num_worker,
            'verbose': verbose,
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    

    @staticmethod
    def process_record_edf(file,
                           chunk_size: int,
                           overlap: int,
                           offline_transform: Union[None, Callable] = None,
                           **kwargs):
        subject_id = int(
            re.findall("sub-(\d\d)_task-motor-imagery_eeg.edf", file)[0])
        edf_reader = mne.io.read_raw_edf(file, preload=True)
        epochs = mne.make_fixed_length_epochs(edf_reader,
                                              duration=8,
                                              preload=True)
        data = epochs.get_data(
        )  # shape(40,33,4000) -num_trial, channels, T—duration（8s）

                  
        eeg = data[:, :30, :]
        #eog = data[:, 30:32, :]


        for trial_id, eeg_trial in enumerate(eeg):
            eeg_baseline = eeg_trial[:, :1000]
            #eog_baseline = eog_trial[:, :1000]
            label = 1 if trial_id % 2 else 0

            assert chunk_size > overlap, f"Arg 'chunk_size' must be larger than arg 'overlap'.Current chunksize is {chunk_size},overlap is {overlap}"
            start = 1000
            step = chunk_size - overlap
            end = start + step
            end_time_point = 3000

            write_pointer = 0
            #PUT baseline into io
            baseline_id = f"{trial_id}_{write_pointer}"
            yield_dict = {'key': baseline_id,'eeg':eeg_baseline}
            yield yield_dict
            write_pointer += 1

            while end <= end_time_point:
                eeg_clip = eeg_trial[:, start:end]
                if (not offline_transform is None):
                    eeg_clip = offline_transform(eeg=eeg_clip,
                                                 baseline=eeg_baseline)['eeg']
                clip_id = f"{trial_id}_{write_pointer}"
                record_info = {
                    "clip_id": clip_id,
                    'label': label,
                    'trial_id': trial_id,
                    'baseline_id': baseline_id,
                    'subject_id': subject_id
                }
                yield {'eeg':eeg_clip,'key': clip_id, "info": record_info}
                start, end = start + step, end + step
                write_pointer += 1

    @staticmethod
    def process_record(file,
                           chunk_size: int,
                           overlap: int,
                           offline_transform: Union[None, Callable] = None,
                           **kwargs):
        subject_id = int(
            re.findall("sub-(\d\d)_task-motor-imagery_eeg.mat", file)[0])
        fdata = loadmat(os.path.join(file))
        X, Y = fdata['eeg'][0][
            0]  # X.shape = [40trials, 33channels,4000timepoints]
        Y = Y[:, 0]
        eeg = X[:, :30, :]


        for trial_id, eeg_trial in enumerate(eeg):
            eeg_baseline = eeg_trial[:, :1000]
            #eog_baseline = eog_trial[:, :1000]
            label = 1 if trial_id % 2 else 0

            assert chunk_size > overlap, f"Arg 'chunk_size' must be larger than arg 'overlap'.Current chunksize is {chunk_size},overlap is {overlap}"
            start = 1000
            step = chunk_size - overlap
            end = start + step
            end_time_point = 3000

            write_pointer = 0
            #PUT baseline into io
            baseline_id = f"{trial_id}_{write_pointer}"
            baseline_yield_dict = {'key': baseline_id,'eeg':eeg_baseline}
            yield baseline_yield_dict
            write_pointer += 1

            while end <= end_time_point:
                eeg_clip = eeg_trial[:, start:end]
                if (not offline_transform is None):
                    eeg_clip = offline_transform(eeg=eeg_clip,
                                                 baseline=eeg_baseline)['eeg']
                clip_id = f"{trial_id}_{write_pointer}"
                record_info = {
                    "clip_id": clip_id,
                    'label': label,
                    'trial_id': trial_id,
                    'baseline_id': baseline_id,
                    'subject_id': subject_id
                }
        
                yield {'eeg':eeg_clip,'key': clip_id, "info": record_info}
                start, end = start + step, end + step
                write_pointer += 1


   

    def set_records(self, root_path, **kwargs):
        subject_dir = os.path.join(root_path, 'sourcedata')
        return [
            os.path.join(os.path.join(subject_dir, sub),
                         os.listdir(os.path.join(subject_dir, sub))[0])
            for sub in os.listdir(subject_dir)
        ]

    def __getitem__(self, index: int) -> Tuple:
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        baseline_index = str(info['baseline_id'])
        signal = self.read_eeg(eeg_record, eeg_index)
        baseline = self.read_eeg(eeg_record, baseline_index)
        if self.online_transform:
            signal = self.online_transform(eeg=signal,
                                            baseline=baseline)['eeg']
    
        if self.label_transform:
            info = self.label_transform(y=info)['y']
        
        return signal, info

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'root_path': self.root_path,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'after_session': self.after_session,
                'after_subject': self.after_subject,
                'io_path': self.io_path,
                'io_size': self.io_size,
                'io_mode': self.io_mode,
                'num_worker': self.num_worker,
                'verbose': self.verbose
            })





class StrokePatientsMIProcessedDataset(StrokePatientsMIDataset):
    '''
    An EEG motor imagery dataset for brain computer interface in acute stroke patients. This version is the processed version provided from authors, in which data have undergone baseline removal and 8-40 bandpass operation. For more detail, please refer to following information.
    
    - Author: Haijie Liu et al.
    - Year: 2024
    - Download URL: https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035/5
    - Reference: Liu, Haijie, et al. "An EEG motor imagery dataset for brain computer interface in acute stroke patients." Scientific Data 11.1 (2024): 131.
    - Stimulus: A video of gripping motion is played on the computer, which leads the patient imagine grabbing the ball. This video stays playing for 4 s. Patients only imagine one hand movement.
    - Signals: Electroencephalogram (30 channels at 500Hz sample rate,. Electrooculography (including horizontal and vertical EOG) (totally 50 participants).
    - label: left hand and right hand.
    
    In order to use this dataset, the downlowd root path is required, containing the following files and directories:
    
    - sourcedata (dir)
    - edffile (dir)
    - task-motor-imagery_events.tsv
    - ...
    
    An example:
    
    .. code-block:: python

        from torcheeg.transforms import Select,BandSignal
        dataset = StrokePatientsMIProcessedDataset(root_path='your unzipped root path',
                                chunk_size=500,  # 1 second
                                overlap = 0,
                                io_path= './stroke_dataset_cache',
                                offline_transform=BandSignal(sampling_rate=500,band_dict={'frequency_range':[8,40]}),
                                label_transform=Select('label')
        )
        print(dataset[0][0].shape) #EEG shape(1,30,500)
        print(dataset[0][1])  # label (int)

    Args:
        root_path (str): Downloaded data files (unzipped) dir.  (default: :obj:`'./StrokePatientsMIDataset'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples. (default: :obj:`500`)
        overlap (int): The number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
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
                 root_path='./StrokePatientsMIDataset',
                 chunk_size: int = 500,
                 overlap: int = 0,
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
                 verbose: bool = True,
                ):
        super().__init__(root_path, chunk_size, overlap, online_transform, 
                         offline_transform, label_transform, before_trial, after_trial, after_session, 
                         after_subject, io_path, io_size, io_mode, num_worker, verbose)
    
    @staticmethod
    def process_record(file,
                           chunk_size: int,
                           overlap: int,
                           offline_transform: Union[None, Callable] = None,
                           **kwargs):
        subject_id = int(
            re.findall("sub-(\d\d)_task-motor-imagery_eeg.edf", file)[0])
        edf_reader = mne.io.read_raw_edf(file, preload=True)
        epochs = mne.make_fixed_length_epochs(edf_reader,
                                              duration=8,
                                              preload=True)
        data = epochs.get_data(
        )  # shape(40,33,4000) -num_trial, channels, T—duration（8s）

                  
        eeg = data[:, :30, :]
        #eog = data[:, 30:32, :]


        for trial_id, eeg_trial in enumerate(eeg):
            eeg_baseline = eeg_trial[:, :1000]
            #eog_baseline = eog_trial[:, :1000]
            label = 1 if trial_id % 2 else 0

            assert chunk_size > overlap, f"Arg 'chunk_size' must be larger than arg 'overlap'.Current chunksize is {chunk_size},overlap is {overlap}"
            start = 1000
            step = chunk_size - overlap
            end = start + step
            end_time_point = 3000

            write_pointer = 0
            #PUT baseline into io
            baseline_id = f"{trial_id}_{write_pointer}"
            yield_dict = {'key': baseline_id,'eeg':eeg_baseline}
            yield yield_dict
            write_pointer += 1

            while end <= end_time_point:
                eeg_clip = eeg_trial[:, start:end]
                if (not offline_transform is None):
                    eeg_clip = offline_transform(eeg=eeg_clip,
                                                 baseline=eeg_baseline)['eeg']
                clip_id = f"{trial_id}_{write_pointer}"
                record_info = {
                    "clip_id": clip_id,
                    'label': label,
                    'trial_id': trial_id,
                    'baseline_id': baseline_id,
                    'subject_id': subject_id
                }
        
                yield {'eeg':eeg_clip,'key': clip_id, "info": record_info}
                start, end = start + step, end + step
                write_pointer += 1

    

    def set_records(self, root_path, **kwargs):
        subject_dir = os.path.join(root_path, 'edffile')
        return [
                os.path.join(
                    os.path.join(subject_dir, sub, 'eeg'),
                    os.listdir(os.path.join(subject_dir, sub, 'eeg'))[0])
                for sub in os.listdir(subject_dir)
            ]
        



