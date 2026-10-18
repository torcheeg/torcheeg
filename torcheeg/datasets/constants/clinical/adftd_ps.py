from ..region_1020 import (FRONTAL_REGION_LIST, GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST)
from ..standard_1005 import STANDARD_1005_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict,
                     format_region_channel_list)

# 19-channel scalp EEG montage (10-20 system) of the ADFTD-PS dataset. The
# order follows the channels.tsv shipped with the recordings. Note that the
# dataset uses the legacy 10-20 naming convention (T3/T4/T5/T6), which
# correspond to T7/T8/P7/P8 in the modern 10-10 convention.
ADFTD_PS_CHANNEL_LIST = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
]

ADFTD_PS_LOCATION_LIST = [
    ['-',  '-', '-',  'FP1', '-',  'FP2', '-',  '-', '-'],
    ['F7', '-', 'F3', '-',   'FZ', '-',   'F4', '-', 'F8'],
    ['T3', '-', 'C3', '-',   'CZ', '-',   'C4', '-', 'T4'],
    ['T5', '-', 'P3', '-',   'PZ', '-',   'P4', '-', 'T6'],
    ['-',  '-', '-',  'O1',  '-',  'O2',  '-',  '-', '-']
]

ADFTD_PS_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    ADFTD_PS_CHANNEL_LIST, ADFTD_PS_LOCATION_LIST)

ADFTD_PS_ADJACENCY_LIST = {
    'FP1': ['FP2', 'F3', 'FZ', 'F7'],
    'FP2': ['FP1', 'FZ', 'F4', 'F8'],
    'F7': ['FP1', 'F3', 'T3'],
    'F3': ['FP1', 'F7', 'FZ', 'C3'],
    'FZ': ['FP1', 'FP2', 'F3', 'F4', 'CZ'],
    'F4': ['FP2', 'FZ', 'F8', 'C4'],
    'F8': ['FP2', 'F4', 'T4'],
    'T3': ['F7', 'C3', 'T5'],
    'C3': ['F3', 'T3', 'CZ', 'P3'],
    'CZ': ['FZ', 'C3', 'C4', 'PZ'],
    'C4': ['F4', 'CZ', 'T4', 'P4'],
    'T4': ['F8', 'C4', 'T6'],
    'T5': ['T3', 'P3', 'O1'],
    'P3': ['C3', 'T5', 'PZ', 'O1'],
    'PZ': ['CZ', 'P3', 'P4'],
    'P4': ['C4', 'PZ', 'T6', 'O2'],
    'T6': ['T4', 'P4', 'O2'],
    'O1': ['T5', 'P3', 'O2'],
    'O2': ['P4', 'T6', 'O1']
}

ADFTD_PS_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    ADFTD_PS_CHANNEL_LIST, ADFTD_PS_ADJACENCY_LIST)

ADFTD_PS_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    ADFTD_PS_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT)

ADFTD_PS_GENERAL_REGION_LIST = format_region_channel_list(
    ADFTD_PS_CHANNEL_LIST, GENERAL_REGION_LIST)
ADFTD_PS_FRONTAL_REGION_LIST = format_region_channel_list(
    ADFTD_PS_CHANNEL_LIST, FRONTAL_REGION_LIST)
ADFTD_PS_HEMISPHERE_REGION_LIST = format_region_channel_list(
    ADFTD_PS_CHANNEL_LIST, HEMISPHERE_REGION_LIST)
