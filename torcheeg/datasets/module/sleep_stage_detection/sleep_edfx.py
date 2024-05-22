import os
from typing import Any, Callable, Dict, Tuple, Union

import mne
import pandas as pd

from ....utils import get_random_dir_path
from ..base_dataset import BaseDataset


class SleepEDFxDataset(BaseDataset):
    r'''
    A dataset for studying human sleep stages (expanded version), of which a small subset was previously contributed in 2002, is now available in PhysioNet. The database now includes 61 full-night polysomnograms of healthy subjects and of subjects with mild difficulty falling asleep, with accompanying expert annotations of sleep stages.
    This class generates training samples and test samples according to the given parameters, and caches the generated results in a unified input and output format (IO). The relevant information of the dataset is as follows:
    
    - Author:B Kemp.
    - Year: 2002
    - Download URL: https://www.physionet.org/content/sleep-edfx/1.0.0/
    - Reference: B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000)
    - Signals: 197 whole-night PolySomnoGraphic sleep recordings, containing EEG, EOG, chin EMG, and event markers. Corresponding hypnograms (sleep patterns) were manually scored by well-trained technicians according to the Rechtschaffen and Kales manual, and are also available. 
    - Sleep stages: W, R, 1, 2, 3, 4, M (Movement time) and ? (not scored).
    
    In order to use this dataset, the download folder :obj:`sleep-edf-database-expanded-1.0.0` is required, containing the following files and directories:
    
    - sleep-cassette (dir)
    - sleep-telemetry (dir)
    - SC-subjects.xls (file)
    - ST-subjects.xls (file)
    - ...

    An example dataset for CNN-based methods:

    .. code-block:: python

        from torcheeg.datasets import SleepEDFxDataset
        from torcheeg import transforms
        
        dataset = SleepEDFxDataset(root_path="./sleep-edf-database-expanded-1.0.0",
                           num_channels=4,
                           chunk_size=3000,
                           remove_unclear_example=True,
                           online_transform=transforms.ToTensor(),
                           label_transform=transforms.Compose([
                               transforms.Select(key="stage"),
                               transforms.Mapping(map_dict={
                                   "W": 0,
                                   "1": 1,
                                   "2": 2,
                                   "3": 3,
                                   "4": 4,
                                   "R": 5
                               })
                           ]))
        
        print(dataset[0])
        # EEG signal (torch.Tensor[2, 3000]),
        # label (int)
            
    Args:
        root_path (str): Downloaded data folder (unzipped sleep-edf-database-expanded-1.0.0.zip) (default: :obj:`'./sleep-edf-database-expanded-1.0.0'`)
        chunk_size (int): Number of data points included in each EEG chunk as training or test samples.  (default: :obj:`3000`)
        overlap (int): Number of overlapping data points between different chunks when dividing EEG chunks. (default: :obj:`0`)
        num_channel (int): Number of channels used, of which the first 4 channels are EEG signals. (default: :obj:`2`)
        version (str): There are two studies corresponding to two different dataset called "cassette" and "Telemetry" in the downloaded data folder. Available options are ['cassette','Telemetry'] (default: :obj:`"cattesse"`)
        remove_unclear_example (bool): Whether to remove the examples which are labels as "?". (default: :obj:`True`)
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
                 root_path: str = './sleep-edf-database-expanded-1.0.0',
                 chunk_size: int = 3000,
                 overlap: int = 0,
                 num_channel: int = 2,
                 version: str = "cassette",
                 remove_unclear_example: bool = True,
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

        assert version in [
            "cassette", "Telemetry"
        ], f"please choose \"version\" in ['cassette','Telemetry'].{version}(current setting) is not available. "
        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'version': version,
            'remove_unclear_example': remove_unclear_example,
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
    def process_record(file: Any = None,
                       root_path: str = './sleep-edf-database-expanded-1.0.0',
                       version: str = "cassette",
                       chunk_size: int = 3000,
                       overlap: int = 0,
                       num_channel: int = 2,
                       remove_unclear_example: bool = True,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        file_name = file  # an element from file name list
        # derive the given arguments (kwargs)
        eeg_dir = "sleep-" + version
        file_path = os.path.join(root_path, eeg_dir, file_name)

        eeg = mne.io.read_raw_edf(file_path).get_data()
        eeg = eeg[:num_channel]

        ann_file_name = list(
            filter(
                lambda x: x[-13:] == "Hypnogram.edf" and x[:7] == file_name[:7],
                os.listdir(os.path.join(root_path, eeg_dir))))[0]
        ann = mne.read_annotations(
            os.path.join(root_path, eeg_dir, ann_file_name))

        subject_id = int(file_name[3:5])
        trial_id = int(file_name[5]) - 1
        meta_info = {'subject_id': subject_id, 'trial_id': trial_id}
        if eeg_dir == "sleep-cassette":
            suject_info = pd.read_excel(
                os.path.join(root_path, "SC-subjects.xls"))
            query = suject_info[(suject_info["subject"] == subject_id)
                                & (suject_info["night"] == trial_id + 1)]
            meta_info["age"], meta_info["sex"], meta_info[
                "light_off"] = query.values[0][-3:]
        else:
            subject_info = pd.read_excel(
                os.path.join(root_path, "ST-subjects.xls"))
            query = subject_info[subject_info["Subject - age - sex"] ==
                                 subject_id]
            meta_info["age"], meta_info["sex"] = query.values[0, 1:3]
            Placebo_night = query[['Placebo night']].values[0, 0]
            meta_info[
                "night"] = 'Placebo night' if trial_id + 1 == Placebo_night else 'Temazepam night'
            meta_info["light_off"] = query['Unnamed: 4'].values[
                0] if trial_id + 1 == Placebo_night else query[
                    'Unnamed: 6'].values[0]

        write_pointer = 0

        if before_trial:
            eeg = before_trial(eeg)
            # record the common meta info

        start_at = 0
        assert chunk_size > 0 and chunk_size <= eeg.shape[
            1], f"please set the chunk_size orrectly.(current chunk_size = {chunk_size})"
        end_at = start_at + chunk_size
        step = chunk_size - overlap

        ann_cur = 0
        cur_stage_time = 0
        while ann_cur < len(ann.duration):
            stage = ann.description[ann_cur][-1]
            if cur_stage_time + chunk_size > ann.duration[ann_cur] * 100:
                start_at += int(ann.duration[ann_cur]) - cur_stage_time
                end_at = start_at + chunk_size
                ann_cur += 1
                cur_stage_time = 0
                continue

            if stage == "?" and remove_unclear_example:
                start_at += step
                end_at = start_at + chunk_size
                cur_stage_time += step
                continue

            clip_sample = eeg[:, start_at:end_at]
            t_eeg = clip_sample

            if not offline_transform is None:
                t = offline_transform(eeg=clip_sample)
                t_eeg = t['eeg']

            clip_id = f'subject{subject_id}_night{trial_id}_{write_pointer}'
            write_pointer += 1
            # record meta info for each signal
            record_info = {
                'start_at': start_at,
                'end_at': end_at,
                'clip_id': clip_id,
                'stage': stage
            }
            record_info.update(meta_info)

            yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}

            start_at += step
            cur_stage_time += step
            end_at = start_at + chunk_size

    def set_records(
            self,
            root_path: str = './sleep-edf-database-expanded-1.0.0',
            **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        self.eeg_dir = "sleep-" + kwargs["version"]
        return list(
            filter(lambda item: item[-7:] == "PSG.edf",
                   os.listdir(os.path.join(root_path, self.eeg_dir))))

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
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'num_channel': self.num_channel,
                'version': self.version,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
