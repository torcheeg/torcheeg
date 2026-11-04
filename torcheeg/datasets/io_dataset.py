from .base_dataset import BaseDataset


class IODataset(BaseDataset):
    """Read samples through independently selected EEG and metadata IO.

    Args:
        eeg_io (BaseEEGIO): Signal storage.
        info_io (BaseInfoIO): Sample metadata storage.
        transform (Callable, optional): Sample transform applied on each read.
    """
