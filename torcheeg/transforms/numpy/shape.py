import numpy as np

from typing import List, Dict, Tuple
from scipy.interpolate import griddata


class PickElectrode:
    r'''
    Select parts of electrode signals based on a given electrode index list.

    .. code-block:: python

        transform = PickElectrode(PickElectrode.to_index_list(
            ['FP1', 'AF3', 'F3', 'F7',
             'FC5', 'FC1', 'C3', 'T7',
             'CP5', 'CP1', 'P3', 'P7',
             'PO3','O1', 'FP2', 'AF4',
             'F4', 'F8', 'FC6', 'FC2',
             'C4', 'T8', 'CP6', 'CP2',
             'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST))
        transform(torch.randn(32, 128)).shape
        >>> (28, 128)

    Args:
        pick_list (np.ndarray): Selected electrode list. Should consist of integers representing the corresponding electrode indices. :obj:`to_index_list` can be used to obtain an index list when we only know the names of the electrode and not their indices.

    .. automethod:: __call__
    '''
    def __init__(self, pick_list: List[int]):
        self.pick_list = pick_list

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].

        Returns:
            np.ndarray: The output signals with the shape of [number of picked electrodes, number of data points].
        '''
        assert max(
            self.pick_list
        ) < x.shape[0], f'The index {max(self.pick_list)} of the specified electrode is out of bounds {x.shape[0]}.'
        return x[self.pick_list]

    @staticmethod
    def to_index_list(electrode_list: List[str], dataset_electrode_list: List[str], strict_mode=False) -> List[int]:
        r'''
        Args:
            electrode_list (list): picked electrode name, consisting of strings.
            dataset_electrode_list (list): The description of the electrode information contained in the EEG signal in the dataset, consisting of strings. For the electrode position information, please refer to constants grouped by dataset :obj:`datasets.constants`.
            strict_mode: (bool): Whether to use strict mode. In strict mode, unmatched picked electrode names are thrown as errors. Otherwise, unmatched picked electrode names are automatically ignored. (defualt: :obj:`False`)
        Returns:
            list: Selected electrode list, consisting of integers representing the corresponding electrode indices.
        '''
        dataset_electrode_dict = dict(zip(dataset_electrode_list, list(range(len(dataset_electrode_list)))))
        if strict_mode:
            return [
                dataset_electrode_dict[electrode] for electrode in electrode_list
            ]
        return [
            dataset_electrode_dict[electrode] for electrode in electrode_list if electrode in dataset_electrode_dict
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class To2d:
    r'''
    Taking the electrode index as the row index and the temporal index as the column index, a two-dimensional EEG signal representation with the size of [number of electrodes, number of data points] is formed. While PyTorch performs convolution on the 2d tensor, an additional channel dimension is required, thus we append an additional dimension.

    .. code-block:: python

        transform = To2d()
        transform(torch.randn(32, 128)).shape
        >>> (1, 32, 128)

    .. automethod:: __call__
    '''
    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].

        Returns:
            np.ndarray: The transformed results with the shape of [1, number of electrodes, number of data points].
        '''
        return x[np.newaxis, ...]


class ToGrid:
    r'''
    A transform method to project the EEG signals of different channels onto the grid according to the electrode positions to form a 3D EEG signal representation with the size of [number of electrodes, width of grid, height of grid]. For the electrode position information, please refer to constants grouped by dataset:

    - datasets.constants.emotion_recognition.deap.DEAP_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.dreamer.DREAMER_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.seed.SEED_CHANNEL_LOCATION_DICT
    - ...

    .. code-block:: python

        transform = ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        transform(torch.randn(32, 128)).shape
        >>> (128, 9, 9)

    Args:
        channel_location (dict): Electrode location information. Represented in dictionary form, where :obj:`key` corresponds to the electrode name and :obj:`value` corresponds to the row index and column index of the electrode on the grid.
    
    .. automethod:: __call__
    '''
    def __init__(self, channel_location: Dict[str, Tuple[int, int]]):
        self.channel_location = channel_location

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].

        Returns:
            np.ndarray: The projected results with the shape of [number of electrodes, width of grid, height of grid].
        '''
        # electronode x timestep
        outputs = np.zeros([9, 9, x.shape[-1]])
        # 9 x 9 x timestep
        for i, (loc_x, loc_y) in enumerate(self.channel_location.values()):
            outputs[loc_x][loc_y] = x[i]

        outputs = outputs.transpose(2, 0, 1)
        # timestep x 9 x 9
        return outputs

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ToInterpolatedGrid:
    r'''
    A transform method to project the EEG signals of different channels onto the grid according to the electrode positions to form a 3D EEG signal representation with the size of [number of electrodes, width of grid, height of grid]. For the electrode position information, please refer to constants grouped by dataset:

    - datasets.constants.emotion_recognition.deap.DEAP_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.dreamer.DREAMER_CHANNEL_LOCATION_DICT
    - datasets.constants.emotion_recognition.seed.SEED_CHANNEL_LOCATION_DICT
    - ...

    .. code-block:: python
    
        transform = ToInterpolatedGrid(DEAP_CHANNEL_LOCATION_DICT)
        transform(torch.randn(32, 128)).shape
        >>> (128, 9, 9)

    Especially, missing values on the grid are supplemented using cubic interpolation

    Args:
        channel_location (dict): Electrode location information. Represented in dictionary form, where :obj:`key` corresponds to the electrode name and :obj:`value` corresponds to the row index and column index of the electrode on the grid.

    .. automethod:: __call__
    '''
    def __init__(self, channel_location: Dict[str, Tuple[int, int]]):
        self.channel_location = channel_location
        self.location_array = np.array(list(channel_location.values()))
        grid_x, grid_y = np.mgrid[min(self.location_array[:, 0]):max(self.location_array[:, 0]):9 * 1j,
                                  min(self.location_array[:, 1]):max(self.location_array[:, 1]):9 * 1j, ]
        self.grid_x = grid_x
        self.grid_y = grid_y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r'''
        Args:
            x (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            
        Returns:
            np.ndarray: The projected results with the shape of [number of electrodes, width of grid, height of grid].
        '''
        # channel x timestep
        x = x.transpose(1, 0)
        # timestep x channel
        outputs = []
        for timestep_split_x in x:
            outputs.append(
                griddata(self.location_array,
                         timestep_split_x, (self.grid_x, self.grid_y),
                         method='cubic',
                         fill_value=0))
        outputs = np.array(outputs)
        return outputs

    def __repr__(self):
        return f"{self.__class__.__name__}()"
