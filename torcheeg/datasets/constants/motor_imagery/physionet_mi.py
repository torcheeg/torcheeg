from ..standard_1005 import (
    STANDARD_1005_CHANNEL_LOCATION_DICT,
)
from ..utils import (
    format_adj_matrix_from_standard,
    format_channel_location_dict,
)

PHYSIONET_MI_CHANNEL_LIST = [
    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'FP1', 'FPZ', 'FP2',
    'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FT8',
    'T7', 'T8', 'T9', 'T10',
    'TP7', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2',
    'IZ',
]

PHYSIONET_MI_EDF_CHANNEL_LIST = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
    'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
    'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.',
    'Fp1.', 'Fpz.', 'Fp2.',
    'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.',
    'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..',
    'Ft7.', 'Ft8.',
    'T7..', 'T8..', 'T9..', 'T10.',
    'Tp7.', 'Tp8.',
    'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
    'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
    'O1..', 'Oz..', 'O2..',
    'Iz..',
]

PHYSIONET_MI_LOCATION_LIST = [
    ['-', '-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-', '-'],
    ['-', '-', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', '-', '-', '-', '-'],
    ['-', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', '-'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', '-', '-'],
    ['T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10'],
    ['-', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', '-'],
    ['-', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', '-'],
    ['-', '-', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', '-', '-', '-', '-'],
    ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', 'IZ', '-', '-', '-', '-', '-', '-'],
]

PHYSIONET_MI_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    PHYSIONET_MI_CHANNEL_LIST, PHYSIONET_MI_LOCATION_LIST)

PHYSIONET_MI_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    PHYSIONET_MI_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT)
