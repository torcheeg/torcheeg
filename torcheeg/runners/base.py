import numbers
from copy import deepcopy


class BaseRunner:
    """Yield independent sample dictionaries without owning storage."""

    def __init__(self, num_worker=0):
        if isinstance(num_worker, bool) or not isinstance(num_worker, numbers.Integral):
            raise TypeError('num_worker must be an integer.')
        if num_worker < 0:
            raise ValueError('num_worker must be nonnegative.')
        self.num_worker = int(num_worker)

    def __call__(self, data):
        from ..datasets.base_data import BaseData
        if not isinstance(data, BaseData):
            raise TypeError('data must be a BaseData.')
        return self.iterate(data, data.get_records())

    def iterate(self, data, records):
        raise NotImplementedError(
            'Method iterate is not implemented in class ' + self.__class__.__name__)


def _iter_samples(data, record):
    for sample in data.iter_record(record):
        if not isinstance(sample, dict) or not {'eeg', 'info'} <= sample.keys():
            raise ValueError(
                'Data must yield dictionaries containing eeg and info.')
        yield deepcopy(sample)
