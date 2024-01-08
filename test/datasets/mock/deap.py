import logging
import os
import pickle as pkl

import numpy as np
from tqdm import tqdm

log = logging.getLogger('torcheeg')


def mock_deap_dataset(
    root_path: str = './DEAP',
    num_subjects: int = 32,
    num_trials: int = 40,
    num_channels: int = 40,
    num_points: int = 8064,
    min_value: float = -38476.736033612986,
    max_value: float = 19385.584785802628,
):
    '''
    Fake DEAP dataset

    Args:
        root_path: The path to save the fake dataset
        num_subjects: The number of subjects
        num_trials: The number of trials
        num_channels: The number of channels
        num_points: The number of points
        min_value: The minimum value of the fake data
        max_value: The maximum value of the fake data

    Returns:
        None
    '''
    # Create the data_preprocessed_python folder
    os.makedirs(root_path, exist_ok=True)

    log.info(f'🌟 | Generating fake data to \033[92m{root_path}\033[0m.')

    # Generate fake data and labels
    # defalut num_points = 8064 = 63 * 128
    data_shape = (num_trials, num_channels, num_points)
    labels_shape = (num_trials, 4)

    # Generate random data and labels
    data = np.random.uniform(min_value, max_value,
                             data_shape).astype(np.float64)
    # labels are float range from 1 to 9
    labels = np.random.uniform(0, 1, labels_shape) * 8 + 1

    # fake data
    for i in tqdm(range(1, num_subjects + 1),
                  desc="[MOCK]",
                  total=num_subjects,
                  position=0,
                  leave=None):
        file_name = f"s{i:02d}.dat"
        file_path = os.path.join(root_path, file_name)

        # Generate random data and labels
        data = np.random.uniform(min_value, max_value,
                                 data_shape).astype(np.float64)
        labels = np.random.uniform(0, 1, labels_shape).astype(np.float64)

        # Create the dictionary
        data_dict = {"data": data, "labels": labels}

        # Save the dictionary as a pickle file
        with open(file_path, "wb") as f:
            pkl.dump(data_dict, f)