from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .base_dataset import BaseDataset


class PairDataset(Dataset):
    r'''
    A dataset class for pairing multiple datasets. This class combines multiple datasets based on a specified join key and join type. It is particularly useful for constructing multimodal datasets, such as merging dataset A and dataset B to simultaneously access both modalities for the same subject during training.

    Below is a quick start example:

    .. code-block:: python

        from torcheeg.datasets import PairDataset, HMCDataset

        dataset_eeg = HMCDataset(root_path='./HMC/recordings', channels=['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2'])
        dataset_ecg = HMCDataset(root_path='./HMC/recordings', channels=['ECG'])
        dataset = PairDataset(datasets=[dataset_eeg, dataset_ecg], join_key='clip_id')

        # Returns a tuple containing both EEG and ECG data:
        # (dataset_eeg[0][0], dataset_eeg[0][1], dataset_ecg[0][0], dataset_ecg[0][1])
        dataset[0]

    Args:
        datasets (List[BaseDataset]): A list of datasets to be paired. Each dataset should inherit from BaseDataset.
        join_key (str): The key used to join the datasets. This should be a column name present in all datasets' info DataFrames. Common join keys could be 'subject_id', 'trial_id', 'clip_id', etc. (default: :obj:`'subject_id'`)
        join_type (str): The type of join to perform. Valid options are:
            - 'inner': Only keeps matching records from all datasets
            - 'outer': Keeps all records, filling missing matches with None
            - 'left': Keeps all records from the first dataset
            - 'right': Keeps all records from the last dataset
            (default: :obj:`'inner'`)
        pair_info_fn (Callable, optional): A custom function to pair the datasets. If provided, this function will be used instead of the default pairing logic. 
                                         The function should take a list of info DataFrames as input and return a DataFrame with appropriate index columns.
                                         This is useful when you need custom pairing logic beyond simple joins. (default: :obj:`None`)
        distinct_key (Optional[str]): A key to ensure distinct pairs based on a specific column. This is useful when you want to avoid duplicate pairs
                                    based on certain criteria. For example, using 'trial_id' would ensure no duplicate trials are paired. (default: :obj:`None`)
    '''

    def __init__(self,
                 datasets: List[BaseDataset],
                 join_key: str = 'subject_id',
                 join_type: str = 'inner',
                 pair_info_fn: Optional[Callable] = None,
                 distinct_key: Optional[str] = None):
        self.datasets = datasets

        if pair_info_fn is None:
            info_list = [dataset.info.copy() for dataset in self.datasets]
            for i, info in enumerate(info_list):
                # print(len(info))
                info[f'index_{i}'] = np.arange(len(info))

            # Merge all datasets based on the join key
            self.info = info_list[0]
            for i, info in enumerate(info_list[1:], start=1):
                columns_to_drop = set(info.columns).intersection(
                    set(self.info.columns)) - {join_key}
                info = info.drop(columns=columns_to_drop)

                self.info = pd.merge(
                    self.info, info, on=join_key, how=join_type)

        else:
            self.info = pair_info_fn([dataset.info.copy()
                                     for dataset in self.datasets])

            if not all(f'index_{i}' in self.info.columns for i in range(len(datasets))):
                raise ValueError(
                    "The custom pair_info_fn must return a DataFrame with appropriate index columns.")

        if distinct_key is not None:
            if distinct_key not in self.info.columns:
                raise ValueError(
                    f"distinct_key '{distinct_key}' not found in merged DataFrame columns")

            new_info = self.info.sample(frac=1, random_state=42)
            new_info = new_info.drop_duplicates(
                subset=distinct_key, keep='first')
            new_info = new_info.reset_index(drop=True)
            self.info = new_info

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        info = self.info.iloc[index].to_dict()

        # Retrieve the signals and labels from all datasets
        signals = []
        labels = []
        for i, dataset in enumerate(self.datasets):
            index_i = info[f'index_{i}']
            signal, label = dataset[index_i]
            signals.append(signal)
            labels.append(label)

        return (*signals, *labels)

    def get_labels(self) -> List:
        labels = []
        for i in range(len(self)):
            *_, label = self.__getitem__(i)
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.info)
