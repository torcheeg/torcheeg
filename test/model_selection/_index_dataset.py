import numpy as np
import pandas as pd

from torcheeg.datasets import BaseDataset
from torcheeg.io import EEGSignalIO


def make_index_dataset(subjects=None):
    if subjects is None:
        subjects = list(range(5))
    dataset = BaseDataset.__new__(BaseDataset)
    rows = []
    eeg_io = EEGSignalIO(io_path='unused', io_mode='memory')
    for subject in subjects:
        for trial in range(5):
            for clip in range(5):
                index = len(rows)
                clip_id = str(index)
                eeg_io.write_eeg(np.array([index, trial, clip]), key=clip_id)
                rows.append({'subject_id': subject, 'trial_id': trial,
                             'clip_id': clip_id, '_record_id': '_record_0'})
    dataset.info = pd.DataFrame(rows)
    dataset.info['category'] = pd.Categorical(
        ['rest', 'task', 'task', 'rest', 'rest'] * (len(rows) // 5))
    dataset.info['timestamp'] = pd.date_range('2020-01-01', periods=len(rows))
    dataset.eeg_io_router = {'_record_0': eeg_io}
    dataset.eeg_signal_paths = {}
    return dataset
