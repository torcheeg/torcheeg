from typing import Any, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .base_dataset import BaseDataset


class ConcatDataset(Dataset):
    """
    A dataset class that vertically concatenates two datasets. This class is particularly useful for combining multiple datasets to create a large-scale dataset for pre-training. The class combines datasets by appending their information DataFrames and provides unified access to samples from both datasets.

    An example usage for combining sleep EEG datasets:

    .. code-block:: python

        isruc_dataset = ISRUCDataset(root_path='./ISRUC-SLEEP',
                       sfreq=100,
                       channels=['F3-M2', 'C3-M2', 'O1-M2',
                                 'F4-M1', 'C4-M1', 'O2-M1'],
                       label_transform=transforms.Compose([
                           transforms.Select('label'),
                           transforms.Mapping({'Sleep stage W': 0,
                                               'Sleep stage N1': 1,
                                               'Sleep stage N2': 2,
                                               'Sleep stage N3': 3,
                                               'Sleep stage R': 4,
                                               'Lights off@@EEG F4-A1': 0})
                       ]),
                       online_transform=transforms.Compose([
                           transforms.MeanStdNormalize(),
                           OrderElectrode(source_electrodes=['F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1'],
                                          target_electrodes=['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'])
                       ]),
                       )

        hmc_dataset = HMCDataset(root_path='./HMC/recordings',
                                sfreq=100,
                                channels=['EEG F4-M1', 'EEG C4-M1',
                                            'EEG O2-M1', 'EEG C3-M2'],
                                label_transform=transforms.Compose([
                                    transforms.Select('label'),
                                    transforms.Mapping({'Sleep stage W': 0,
                                                        'Sleep stage N1': 1,
                                                        'Sleep stage N2': 2,
                                                        'Sleep stage N3': 3,
                                                        'Sleep stage R': 4,
                                                        'Lights off@@EEG F4-A1': 0})
                                ]),
                                online_transform=transforms.Compose([
                                    transforms.MeanStdNormalize(),
                                    OrderElectrode(source_electrodes=['EEG F4-M1', 'EEG C4-M1',
                                                                        'EEG O2-M1', 'EEG C3-M2'],
                                                    target_electrodes=['F3', 'EEG F4-M1', 'EEG C3-M2', 'EEG C4-M1', 'O1', 'EEG O2-M1'])
                                ]),
                                )

        p2018_dataset = P2018Dataset(root_path='./P2018/training/', sfreq=100, channels=['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'],
                                    label_transform=transforms.Compose([
                                        transforms.Select('label'),
                                        transforms.Mapping({'Sleep stage W': 0,
                                                            'Sleep stage N1': 1,
                                                            'Sleep stage N2': 2,
                                                            'Sleep stage N3': 3,
                                                            'Sleep stage R': 4,
                                                            'Lights off@@EEG F4-A1': 0})
                                    ]),
                                    online_transform=transforms.Compose([
                                        transforms.MeanStdNormalize(),
                                        OrderElectrode(source_electrodes=['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'],
                                                        target_electrodes=['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'])
                                    ]),
                                    )

        sleep_dataset = ConcatDataset(
            isruc_dataset, ConcatDataset(hmc_dataset, p2018_dataset))

    Args:
        dataset1 (BaseDataset): The first dataset to be concatenated.
        dataset2 (BaseDataset): The second dataset to be concatenated.
    """

    def __init__(self, dataset1: BaseDataset, dataset2: BaseDataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # Combine info DataFrames
        info1 = dataset1.info.copy()
        info2 = dataset2.info.copy()

        # Add prefixes to subject_id and trial_id columns
        if 'subject_id' in info1.columns:
            info1['subject_id'] = 'dataset1_' + info1['subject_id'].astype(str)
        if 'subject_id' in info2.columns:
            info2['subject_id'] = 'dataset2_' + info2['subject_id'].astype(str)
        if 'trial_id' in info1.columns:
            info1['trial_id'] = 'dataset1_' + info1['trial_id'].astype(str)
        if 'trial_id' in info2.columns:
            info2['trial_id'] = 'dataset2_' + info2['trial_id'].astype(str)

        # Add a source column to identify the origin of each sample
        info1['dataset_source'] = 'dataset1'
        info2['dataset_source'] = 'dataset2'

        # Add an index column for each dataset
        info1['original_index'] = np.arange(len(info1))
        info2['original_index'] = np.arange(len(info2))

        # Combine the DataFrames
        self.info = pd.concat([info1, info2], ignore_index=True)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns the item at the specified index from the concatenated dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[Any, Any]: A tuple containing the signal and label of the item.
        """
        info = self.info.iloc[index]

        if info['dataset_source'] == 'dataset1':
            return self.dataset1[info['original_index']]
        else:
            return self.dataset2[info['original_index']]

    def __len__(self) -> int:
        """
        Returns the total number of samples in the concatenated dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.info)

    def get_labels(self) -> list:
        """
        Returns a list of labels for all samples in the concatenated dataset.

        Returns:
            list: A list of labels for all samples.
        """
        labels = []
        for i in range(len(self)):
            _, label = self.__getitem__(i)
            labels.append(label)
        return labels