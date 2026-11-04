import numbers
from copy import copy, deepcopy
from uuid import uuid4

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..io import BaseEEGIO, BaseInfoIO


class _RequiredMetadata:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        raise NotImplementedError(
            "Attribute " + self._name + " is not implemented in class "
            + owner.__name__)


class BaseDataset(Dataset):
    """Store sample dictionaries and transform selected samples on reading."""

    channels = _RequiredMetadata()
    grid = _RequiredMetadata()
    adjacency = _RequiredMetadata()
    _STORAGE_COLUMNS = frozenset(('_record_id', 'clip_id', 'baseline_id'))

    def __init__(self, eeg_io, info_io, transform=None):
        if not isinstance(eeg_io, BaseEEGIO):
            raise TypeError('eeg_io must be a BaseEEGIO.')
        if not isinstance(info_io, BaseInfoIO):
            raise TypeError('info_io must be a BaseInfoIO.')
        self.eeg_io = eeg_io
        self.info_io = info_io
        self._info = None
        if transform is not None and not callable(transform):
            raise TypeError('transform must be callable.')
        self.transform = transform
        self._is_view = False

    @property
    def info(self):
        """Sample metadata in the current selection order."""
        if self._info is None:
            self._info = self.info_io.read_all()
        return self._info

    @info.setter
    def info(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError('info must be a pandas DataFrame.')
        self._info = value.reset_index(drop=True).copy(deep=True)
        for name in self._info.select_dtypes(include='object'):
            self._info[name] = self._info[name].map(deepcopy)
        self._is_view = True

    def append(self, sample):
        """Append one raw sample; storage identities are assigned internally."""
        if self._is_view:
            raise ValueError(
                'Append to a storage dataset, not a selection or transform view.')
        if not isinstance(sample, dict):
            raise TypeError('sample must be a dictionary.')
        if not {'eeg', 'info'} <= sample.keys():
            raise ValueError('sample must contain eeg and info.')
        if set(sample) - {'eeg', 'baseline', 'info'}:
            raise ValueError(
                'Stored samples support eeg, info and optional baseline.')
        if not isinstance(sample['info'], dict):
            raise TypeError('sample info must be a dictionary.')
        if self._STORAGE_COLUMNS.intersection(sample['info']):
            raise ValueError(
                'Sample metadata must not contain storage identifiers.')
        if sample['eeg'] is None or ('baseline' in sample and sample['baseline'] is None):
            raise ValueError('Stored signals must not be None.')
        info = deepcopy(sample['info'])
        record = uuid4().hex
        self.eeg_io.write_eeg(deepcopy(sample['eeg']), 'eeg', record=record)
        info.update(_record_id=record, clip_id='eeg')
        if 'baseline' in sample:
            self.eeg_io.write_eeg(
                deepcopy(sample['baseline']), 'baseline', record=record)
            info['baseline_id'] = 'baseline'
        self.info_io.write_info(info)
        self._info = None

    def extend(self, samples):
        """Consume and store samples incrementally."""
        iterator = iter(samples)
        try:
            for sample in iterator:
                self.append(sample)
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()

    def __getitem__(self, index):
        if isinstance(index, (bool, np.bool_)):
            raise TypeError('A scalar boolean is not a sample index.')
        if isinstance(index, numbers.Integral):
            sample = self._read_sample(int(index))
            if self.transform is not None:
                sample = self.transform(sample)
                if not isinstance(sample, dict):
                    raise TypeError(
                        'A sample transform must return a dictionary.')
            return sample
        if isinstance(index, slice):
            rows = self.info.iloc[index]
        else:
            positions = np.asarray(index)
            if positions.ndim != 1:
                raise ValueError('Sample indices must be one-dimensional.')
            if positions.dtype.kind == 'b':
                if len(positions) != len(self):
                    raise ValueError(
                        'Boolean mask length must equal dataset length.')
            elif positions.size and positions.dtype.kind not in 'iu':
                raise TypeError('Sample indices must be integers or booleans.')
            elif not positions.size:
                positions = positions.astype(np.intp)
            rows = self.info.iloc[positions]
        result = copy(self)
        result.info = rows
        return result

    def _read_sample(self, index):
        info = self.info.iloc[index].to_dict()
        record = info['_record_id']
        sample = {
            'eeg': deepcopy(self.eeg_io.read_eeg(info['clip_id'], record=record)),
            'info': deepcopy({key: value for key, value in info.items()
                              if key not in self._STORAGE_COLUMNS}),
        }
        baseline = info.get('baseline_id')
        if isinstance(baseline, str):
            sample['baseline'] = deepcopy(
                self.eeg_io.read_eeg(baseline, record=record))
        return sample

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __len__(self):
        return len(self.info)

    def __copy__(self):
        result = type(self).__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result._info = self.info.copy(deep=True)
        for name in result._info.select_dtypes(include='object'):
            result._info[name] = result._info[name].map(deepcopy)
        result._is_view = True
        return result

    def close(self):
        self.eeg_io.close()
        self.info_io.close()

    @property
    def repr_body(self):
        return {'eeg_io': type(self.eeg_io).__name__,
                'info_io': type(self.info_io).__name__}

    def __repr__(self):
        values = ', '.join(f'{key}={value!r}' for key,
                           value in self.repr_body.items())
        return f'{type(self).__name__}({values}, length={len(self)})'
