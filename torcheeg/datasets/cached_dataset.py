import os

from ..io import (LMDBEEGIO, CSVInfoIO, MemoryEEGIO, MemoryInfoIO, NumpyEEGIO,
                  PickleEEGIO, ZarrEEGIO)
from ..io.utils import ensure_cache
from ..runners import MultiprocessRunner
from ..utils import get_random_dir_path
from .io_dataset import IODataset
from .utils import compute_fingerprint


class CachedDataset(IODataset):
    """Build a dataset by composing Data, a runner and transactional storage."""

    data_class = None

    def __init__(self, io_path=None, io_size=1048576, io_mode='lmdb',
                 num_worker=0, verbose=True, offline_transform=None,
                 online_transform=None, label_transform=None, **configuration):
        if self.data_class is None:
            raise NotImplementedError(
                'Attribute data_class is not implemented in class ' + self.__class__.__name__)
        backends = {'lmdb': LMDBEEGIO, 'pickle': PickleEEGIO,
                    'numpy': NumpyEEGIO, 'zarr': ZarrEEGIO, 'memory': MemoryEEGIO}
        if io_mode not in backends:
            raise ValueError('Unsupported io_mode ' + str(io_mode))
        self.io_path = get_random_dir_path(
            dir_prefix='datasets') if io_path is None else os.fspath(io_path)
        self.io_size = io_size
        self.io_mode = io_mode
        self.num_worker = num_worker
        self.verbose = verbose
        for name, transform in (('offline_transform', offline_transform),
                                ('online_transform', online_transform),
                                ('label_transform', label_transform)):
            if transform is not None and not callable(transform):
                raise TypeError(name + ' must be callable.')
        if 'transform' in configuration:
            raise TypeError('Use offline_transform or online_transform.')
        self.offline_transform = offline_transform
        self.online_transform = online_transform
        self.data = self.data_class(
            transform=offline_transform, **configuration)
        eeg_path = None if io_mode == 'memory' else os.path.join(
            self.io_path, 'eeg')
        info_path = None if io_mode == 'memory' else os.path.join(
            self.io_path, 'info.csv')
        eeg = LMDBEEGIO(
            eeg_path, io_size) if io_mode == 'lmdb' else backends[io_mode](eeg_path)
        info = MemoryInfoIO() if io_mode == 'memory' else CSVInfoIO(info_path)
        records = self.data.get_records()
        if not records:
            raise ValueError('No records were found for the dataset.')
        options = dict(
            configuration, offline_transform=offline_transform, io_mode=io_mode)
        fingerprint = compute_fingerprint(
            self.data_class, options, records, self.io_path)

        def build(eeg_io, info_io):
            output = IODataset(eeg_io, info_io)
            output.extend(MultiprocessRunner(
                num_worker=num_worker).iterate(self.data, records))

        ensure_cache(eeg, info, fingerprint, build)
        super().__init__(eeg, info, transform=online_transform, label_transform=label_transform)
