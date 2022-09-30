import mne

from braindecode.datasets import (create_from_mne_raw, create_from_mne_epochs)

# 5, 6, 7, 10, 13, 14 are codes for executed and imagined hands/feet
subject_id = 22
event_codes = [5, 6, 9, 10, 13, 14]
# event_codes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# This will download the files if you don't have them yet,
# and then return the paths to the files.
physionet_paths = mne.datasets.eegbci.load_data(subject_id,
                                                event_codes,
                                                update_path=False)

# Load each of the files
parts = [
    mne.io.read_raw_edf(path, preload=True, stim_channel='auto')
    for path in physionet_paths
]

descriptions = [{
    "event_code": code,
    "subject": subject_id
} for code in event_codes]
# [{'event_code': 5, 'subject': 22}, {'event_code': 6, 'subject': 22}, {'event_code': 9, 'subject': 22}, {'event_code': 10, 'subject': 22}, {'event_code': 13, 'subject': 22}, {'event_code': 14, 'subject': 22}]

windows_dataset = create_from_mne_raw(
    parts,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=500,
    window_stride_samples=500,
    drop_last_window=False,
    descriptions=descriptions,
)