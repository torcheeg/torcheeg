import os
from typing import Any, Dict, Union

import pandas as pd

from .eeg_signal import EEGSignalIO
from .meta_info import MetaInfoIO


class IORouter:
    def __init__(self,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb') -> None:
        # Get all records
        records = os.listdir(io_path)
        # Filter records with prefix '_record_'
        records = list(filter(lambda x: '_record_' in x, records))
        # Sort records
        records = sorted(records, key=lambda x: int(x.split('_')[2]))

        assert len(
            records) > 0, f"The io_path, {io_path}, is corrupted. Please delete this folder and try again."

        self.eeg_signal_io_router = {}
        info_merged = []

        for record in records:
            meta_info_io_path = os.path.join(io_path, record, 'info.csv')
            eeg_signal_io_path = os.path.join(io_path, record, 'eeg')

            info_io = MetaInfoIO(meta_info_io_path)
            eeg_io = EEGSignalIO(eeg_signal_io_path,
                                 io_size=io_size,
                                 io_mode=io_mode)

            self.eeg_signal_io_router[record] = eeg_io
            info_df = info_io.read_all()

            assert '_record_id' not in info_df.columns, \
                "column '_record_id' is a forbidden reserved word and is used to index the corresponding IO. Please replace your '_record_id' with another name."

            info_df['_record_id'] = record
            info_merged.append(info_df)

        self.info_io_router = pd.concat(info_merged, ignore_index=True)

    def read_info(self, index: int) -> Dict:
        """
        Query meta information by index.

        Args:
            index (int): Index of the meta information

        Returns:
            dict: Meta information
        """
        return self.info_io_router.iloc[index].to_dict()

    def read_eeg(self, record: str, key: str) -> Any:
        """
        Query EEG signal by record and key.

        Args:
            record (str): Record ID
            key (str): Index of the EEG signal

        Returns:
            any: EEG signal sample
        """
        eeg_io = self.eeg_signal_io_router[record]
        return eeg_io.read_eeg(key)

    def write_eeg(self, record: str, key: str, eeg: Any) -> None:
        """
        Update EEG signal by record and key.

        Args:
            record (str): Record ID
            key (str): Index of the EEG signal
            eeg (any): The EEG signal sample to be updated
        """
        eeg_io = self.eeg_signal_io_router[record]
        eeg_io.write_eeg(eeg=eeg, key=key)

    def __getitem__(self, index: int) -> tuple:
        """
        Read both EEG signal and its meta information by index.

        Args:
            index (int): Index of the sample

        Returns:
            tuple: (eeg_signal, meta_information)
        """
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)

        return eeg, info

    def __len__(self) -> int:
        """
        Get the total number of samples in the router.

        Returns:
            int: Number of samples
        """
        return len(self.info_io_router)

    def __copy__(self) -> 'IORouter':
        """
        Create a shallow copy of the router.

        Returns:
            IORouter: A new router instance with copied data
        """
        result = IORouter.__new__(IORouter)
        # Shallow copy data
        result.eeg_signal_io_router = {}
        for record, eeg_io in self.eeg_signal_io_router.items():
            result.eeg_signal_io_router[record] = eeg_io.__copy__()
        # Deep copy info (for further modification)
        result.info_io_router = self.info_io_router.__copy__()
        return result


class LazyIORouter:
    def __init__(self,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb') -> None:
        self.io_size = io_size
        self.io_mode = io_mode
        
        # Get all records
        records = os.listdir(io_path)
        records = list(filter(lambda x: '_record_' in x, records))
        records = sorted(records, key=lambda x: int(x.split('_')[2]))

        assert len(records) > 0, f"The io_path, {io_path}, is corrupted."

        # Store paths instead of EEGSignalIO instances
        self.eeg_signal_io_paths = {}
        info_merged = []

        for record in records:
            meta_info_io_path = os.path.join(io_path, record, 'info.csv')
            eeg_signal_io_path = os.path.join(io_path, record, 'eeg')

            info_io = MetaInfoIO(meta_info_io_path)
            # Store path only
            self.eeg_signal_io_paths[record] = eeg_signal_io_path
            
            info_df = info_io.read_all()
            assert '_record_id' not in info_df.columns, \
                "column '_record_id' is a forbidden reserved word."
                
            info_df['_record_id'] = record
            info_merged.append(info_df)

        self.info_io_router = pd.concat(info_merged, ignore_index=True)

    def read_info(self, index: int) -> Dict:
        return self.info_io_router.iloc[index].to_dict()

    def read_eeg(self, record: str, key: str) -> Any:
        # Create temporary EEGSignalIO instance
        eeg_signal_io = EEGSignalIO(
            self.eeg_signal_io_paths[record],
            io_size=self.io_size,
            io_mode=self.io_mode
        )
        # Read data and let the instance be garbage collected
        return eeg_signal_io.read_eeg(key)

    def write_eeg(self, record: str, key: str, eeg: Any) -> None:
        # Create temporary EEGSignalIO instance
        eeg_signal_io = EEGSignalIO(
            self.eeg_signal_io_paths[record],
            io_size=self.io_size,
            io_mode=self.io_mode
        )
        # Write data and let the instance be garbage collected
        eeg_signal_io.write_eeg(eeg=eeg, key=key)

    def __getitem__(self, index: int) -> tuple:
        info = self.read_info(index)
        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index)
        return eeg, info

    def __len__(self) -> int:
        return len(self.info_io_router)

    def __copy__(self) -> 'LazyIORouter':
        result = LazyIORouter.__new__(LazyIORouter)
        # Copy paths
        result.eeg_signal_io_paths = self.eeg_signal_io_paths.copy()
        result.io_size = self.io_size
        result.io_mode = self.io_mode
        # Deep copy info
        result.info_io_router = self.info_io_router.__copy__()
        return result