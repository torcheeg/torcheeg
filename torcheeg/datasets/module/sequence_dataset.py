import hashlib
import json
from copy import copy
from numbers import Integral, Real
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, default_collate


class SequenceDataset(Dataset):
    r'''Form complete sequences of consecutive epochs within recording groups.

    Pass an already partitioned dataset with an ``info`` DataFrame containing
    ``clip_id``, ``start_at`` and ``end_at``. Times must be seconds from the
    recording start. Each epoch must have ``epoch_duration`` seconds. The
    original dataset must return ``(signal, label)``. Numeric arrays and labels
    are stacked with PyTorch's ``default_collate``; dictionary labels retain
    their structure. SleepDG needs a scalar integer label for each epoch.

    Recording, subject, session, trial and dataset-source columns present in
    metadata are always group boundaries. ``groupby`` adds boundaries, such as
    a partition column; it cannot remove the built-in boundaries. At least
    ``record_id``, ``_record_id`` or an explicit grouping column is required.
    Within each group, epochs are sorted by start time. A gap starts a new run;
    duplicate or overlapping epochs raise an error. Incomplete tails are
    dropped, without padding or joining another run.

    .. code-block:: python

        from torcheeg.datasets import SequenceDataset

        # source_train already contains only the intended training partition.
        sequences = SequenceDataset(source_train, sequence_length=20,
                                    domain_id=0)
        eeg, labels, domain = sequences[0]  # [20, C, T], [20], scalar

    The wrapper snapshots the source metadata through a shallow dataset copy.
    Signal storage and transforms remain shared. Do not mutate that storage,
    transforms or nested source datasets while using the wrapper. Sequence IDs
    bind the source epoch identities, metadata order and window configuration;
    stale CSV split rows fail lookup instead of selecting another window.

    Split subjects/records before construction, or add an explicit partition
    column to ``groupby``. Randomly splitting overlapping sequences afterwards
    can leak epochs across partitions. This wrapper does not normalize signals,
    infer cross-night person identity or detect leakage between separate inputs.

    Args:
        dataset (Dataset): Source epoch dataset with positional metadata and
            two-item samples.
        sequence_length (int): Epochs in each complete sequence. (default: :obj:`20`)
        stride (int, optional): Window stride within each continuous run. None
            uses sequence_length, with no overlap. (default: :obj:`None`)
        epoch_duration (float): Epoch duration in seconds. (default: :obj:`30.0`)
        groupby (list of str, optional): Additional metadata group columns.
            (default: :obj:`None`)
        domain_id (int, optional): Constant nonnegative source-dataset code.
            Adds the third sample item. (default: :obj:`None`)
        domain_column (str, optional): Nonnegative integer source-dataset codes
            from metadata. Mutually exclusive with domain_id. This column is
            also a group boundary. (default: :obj:`None`)
    '''

    def __init__(self, dataset: Dataset, sequence_length: int = 20,
                 stride: Optional[int] = None, epoch_duration: float = 30.,
                 groupby: Optional[List[str]] = None, domain_id: Optional[int] = None,
                 domain_column: Optional[str] = None):
        for name, value in [('sequence_length', sequence_length),
                            ('stride', sequence_length if stride is None else stride)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if (isinstance(epoch_duration, (bool, np.bool_)) or not isinstance(epoch_duration, Real)
                or not np.isfinite(epoch_duration) or epoch_duration <= 0):
            raise ValueError('epoch_duration must be a finite positive number of seconds.')
        if domain_id is not None and domain_column is not None:
            raise ValueError('Choose domain_id or domain_column, not both.')
        if domain_id is not None:
            self._validate_domain(domain_id)
        if not isinstance(dataset, Dataset) or not isinstance(getattr(dataset, 'info', None), pd.DataFrame):
            raise TypeError('dataset must be a Dataset with an info DataFrame.')
        if len(dataset) != len(dataset.info) or not dataset.info.columns.is_unique:
            raise ValueError('Source metadata must have one row per epoch and unique columns.')
        extra = [] if groupby is None else list(groupby) if isinstance(groupby, (list, tuple)) else None
        if extra is None or any(not isinstance(name, str) or not name for name in extra) or len(set(extra)) != len(extra):
            raise ValueError('groupby must contain unique column names.')
        if domain_column is not None and (not isinstance(domain_column, str) or not domain_column):
            raise ValueError('domain_column must be a column name.')
        reserved = {'clip_id', 'start_at', 'end_at', 'sequence_id'}
        if set(extra) & reserved or domain_column in reserved:
            raise ValueError('Group/domain columns must identify boundaries, not epoch times or sample IDs.')
        info = dataset.info.copy(deep=True).reset_index(drop=True)
        boundaries = [name for name in ('dataset_source', 'subject_id', 'session_id',
                                        'record_id', 'trial_id', '_record_id') if name in info]
        if not extra and not any(name in info for name in ('record_id', '_record_id')):
            raise ValueError('Recording identity or explicit groupby columns are required.')
        columns = list(dict.fromkeys(boundaries + extra + ([domain_column] if domain_column else [])))
        required = list(dict.fromkeys(columns + ['clip_id', 'start_at', 'end_at']))
        for name in required:
            if name not in info or info[name].isna().any():
                raise ValueError(f'Source metadata needs nonmissing {name} values.')
        identity_columns = boundaries if boundaries else columns
        if info.duplicated(subset=identity_columns + ['clip_id']).any():
            raise ValueError('The same source epoch appears more than once in a recording group.')
        for name in ('start_at', 'end_at'):
            values = info[name].tolist()
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values):
                raise ValueError('Epoch times must be numeric seconds.')
            info[name] = np.asarray(values, dtype=float)
        times = info[['start_at', 'end_at']].to_numpy(dtype=float)
        tolerance = min(1e-6, float(epoch_duration) * 1e-6)
        if (not np.isfinite(times).all() or np.any(times < 0) or np.any(times[:, 1] <= times[:, 0])
                or not np.allclose(times[:, 1] - times[:, 0], epoch_duration, rtol=0, atol=tolerance)):
            raise ValueError('Epoch times must be finite, nonnegative and match epoch_duration.')
        if domain_column is not None:
            for value in info[domain_column]:
                self._validate_domain(value)
        self.dataset = copy(dataset)
        self.dataset.info = dataset.info.copy(deep=True)
        self.sequence_length, self.stride = int(sequence_length), int(sequence_length if stride is None else stride)
        self.epoch_duration = float(epoch_duration)
        self.groupby = tuple(columns)
        self.domain_id, self.domain_column = domain_id, domain_column
        # Include value types to distinguish string/integer IDs after CSV reads.
        typed = pd.DataFrame({name: [(type(v).__name__, v) for v in info[name].tolist()]
                              for name in required})
        try:
            hashes = pd.util.hash_pandas_object(typed, index=False).to_numpy()
        except TypeError as error:
            raise ValueError('Sequence metadata must contain hashable scalar values.') from error
        configuration = {'columns': [(name, str(info[name].dtype)) for name in required],
                         'sequence_length': self.sequence_length, 'stride': self.stride,
                         'epoch_duration': self.epoch_duration,
                         'domain_id': None if domain_id is None else int(domain_id),
                         'domain_column': domain_column}
        signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode() + hashes.tobytes()).hexdigest()
        self._sequences = {}
        self._domains = {}
        rows = []
        for _, group in info.groupby(columns, sort=False):
            ordered = group.sort_values('start_at', kind='mergesort')
            begins, ends = ordered['start_at'].to_numpy(), ordered['end_at'].to_numpy()
            if np.any(begins[1:] < ends[:-1] - tolerance):
                raise ValueError('Duplicate or overlapping epochs in one recording group.')
            cuts = np.flatnonzero(~np.isclose(begins[1:], ends[:-1], rtol=0, atol=tolerance)) + 1
            for run in np.split(ordered.index.to_numpy(), cuts):
                for start in range(0, len(run) - self.sequence_length + 1, self.stride):
                    positions = tuple(int(i) for i in run[start:start + self.sequence_length])
                    identity = 'seq_' + hashlib.sha256((signature + repr(positions)).encode()).hexdigest()
                    self._sequences[identity] = positions
                    first, last = info.iloc[positions[0]], info.iloc[positions[-1]]
                    if domain_column is not None or domain_id is not None:
                        self._domains[identity] = int(first[domain_column] if domain_column is not None else domain_id)
                    row = {name: first[name] for name in columns}
                    row.update(sequence_id=identity, clip_id=identity,
                               start_at=float(first['start_at']), end_at=float(last['end_at']))
                    rows.append(row)
        self.info = pd.DataFrame(rows, columns=columns + ['sequence_id', 'clip_id', 'start_at', 'end_at'])

    @staticmethod
    def _validate_domain(value):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                or value < 0 or value > np.iinfo(np.int64).max):
            raise ValueError('Source-dataset codes must be nonnegative int64 values.')

    def get_epoch_indices(self, index: int) -> Tuple[int, ...]:
        r'''Return source positional indices for one sequence in time order.'''
        identity = self.info.iloc[index]['sequence_id']
        if identity not in self._sequences:
            raise ValueError('Sequence metadata does not match this source/configuration; use a new split_path.')
        return self._sequences[identity]

    def read_info(self, index: int) -> Dict:
        r'''Return group identifiers, sequence ID and original epoch bounds.'''
        return self.info.iloc[index].to_dict()

    def __getitem__(self, index: int):
        samples = [self.dataset[position] for position in self.get_epoch_indices(index)]
        if any(not isinstance(sample, (tuple, list)) or len(sample) != 2 for sample in samples):
            raise ValueError('Source samples must contain exactly (signal, label).')
        signals, labels = default_collate(samples)
        if self.domain_column is not None or self.domain_id is not None:
            return signals, labels, self._domains[self.info.iloc[index]['sequence_id']]
        return signals, labels

    def __len__(self):
        return len(self.info)

    def get_labels(self) -> list:
        r'''Return the source label structures stacked for each sequence.'''
        return [self[index][1] for index in range(len(self))]
