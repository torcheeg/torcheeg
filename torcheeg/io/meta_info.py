import os
import csv

import pandas as pd

from typing import Dict


class MetaInfoIO:
    r'''
    Use with torcheeg.io.EEGSignalIO to store description information for EEG signals in the form of a table, so that the user can still analyze, insert, delete and modify the corresponding information after the generation is completed.

    .. code-block:: python

        info_io = MetaInfoIO('YOUR_PATH')
        key = info_io.write_info({
            'clip_id': 0,
            'baseline_id': 1,
            'valence': 1.0,
            'arousal': 9.0
        })
        info = info_io.read_info(key).to_dict()
        >>> {
                'clip_id': 0,
                'baseline_id': 1,
                'valence': 1.0,
                'arousal': 9.0
            }

    Args:
        io_path (str): Where the table is stored.
    '''
    def __init__(self, io_path: str) -> None:
        self.io_path = io_path
        if not os.path.exists(self.io_path):
            open(self.io_path, 'x').close()
            self.write_pointer = 0
        else:
            self.write_pointer = len(self)

    def __len__(self):
        if os.path.getsize(self.io_path) == 0:
            return 0
        info_list = pd.read_csv(self.io_path)
        return len(info_list)

    def write_info(self, obj: Dict) -> int:
        r'''
        Insert a description of the EEG signal.

        Args:
            obj (dict): The description to be written into the table.
        
        Returns:
            int: The index of written EEG description in the table.
        '''
        with open(self.io_path, 'a+') as f:
            require_head = os.path.getsize(self.io_path) == 0
            writer = csv.DictWriter(f, fieldnames=list(obj.keys()))
            if require_head:
                writer.writeheader()
            writer.writerow(obj)
        key = self.write_pointer
        self.write_pointer += 1
        return key

    def read_info(self, key) -> pd.DataFrame:
        r'''
        Query the corresponding EEG description in the table according to the index.

        Args:
            key (int): The index of the EEG description to be queried.
        Returns:
            pd.DataFrame: The EEG description.
        '''
        return pd.read_csv(self.io_path).iloc[key]

    def read_all(self) -> pd.DataFrame:
        r'''
        Get all EEG descriptions in the database in tabular form.

        Returns:
            pd.DataFrame: The EEG descriptions.
        '''
        return pd.read_csv(self.io_path)