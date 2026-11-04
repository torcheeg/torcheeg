from .base_dataset import BaseDataset


class IODataset(BaseDataset):
    """Read samples through independently selected EEG and metadata IO.

    Args:
        eeg_io (BaseEEGIO): Signal storage.
        info_io (BaseInfoIO): Sample metadata storage.
        transform (Callable, optional): Transform eeg and optional baseline on each read.
        label_transform (Callable, optional): Transform info into a dictionary containing label.
    """
