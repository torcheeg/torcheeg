from ..region_1020 import (FRONTAL_REGION_LIST, GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST)
from ..standard_1005 import STANDARD_1005_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict, format_region_channel_list)

TUHTUEG_CHANNEL_LIST = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6', 'T1', 'T2', 'FZ', 'CZ', 'PZ'
]

TUHTUEG_LOCATION_LIST = [
    ['-',  '-', '-',  'FP1', '-',  'FP2', '-',  '-', '-'],
    ['-',  '-', '-',  '-',  '-',  '-',   '-',  '-', '-'],
    ['F7', '-', 'F3', '-',  'FZ', '-',   'F4', '-', 'F8'],
    ['T1', '-', '-',  '-',  '-',  '-',   '-',  '-', 'T2'],
    ['T3', '-', 'C3', '-',  'CZ', '-',   'C4', '-', 'T4'],
    ['-',  '-', '-',  '-',  '-',  '-',   '-',  '-', '-'],
    ['T5', '-', 'P3', '-',  'PZ', '-',   'P4', '-', 'T6'],
    ['-',  '-', '-',  '-',  '-',  '-',   '-',  '-', '-'],
    ['-',  '-', '-',  'O1', '-',  'O2',  '-',  '-', '-']
]

TUHTUEG_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    TUHTUEG_CHANNEL_LIST, TUHTUEG_LOCATION_LIST)

TUHTUEG_ADJACENCY_LIST = {
    'FP1': ['FP2', 'F3', 'FZ'],
    'FP2': ['FP1', 'FZ', 'F4'],
    'F7': ['T1', 'T3'],
    'F3': ['FP1', 'FZ', 'C3', 'CZ'],
    'FZ': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'CZ', 'C4'],
    'F4': ['FP2', 'FZ', 'CZ', 'C4'],
    'F8': ['T2', 'T4'],
    'T1': ['F7', 'T3'],
    'T2': ['F8', 'T4'],
    'T3': ['F7', 'T1', 'T5'],
    'C3': ['F3', 'FZ', 'CZ', 'P3', 'PZ'],
    'CZ': ['F3', 'FZ', 'F4', 'C3', 'C4', 'P3', 'PZ', 'P4'],
    'C4': ['FZ', 'F4', 'CZ', 'PZ', 'P4'],
    'T4': ['F8', 'T2', 'T6'],
    'T5': ['T3'],
    'P3': ['C3', 'CZ', 'PZ', 'O1'],
    'PZ': ['C3', 'CZ', 'C4', 'P3', 'P4', 'O1', 'O2'],
    'P4': ['CZ', 'C4', 'PZ', 'O2'],
    'T6': ['T4'],
    'O1': ['P3', 'PZ', 'O2'],
    'O2': ['PZ', 'P4', 'O1']
}

TUHTUEG_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    TUHTUEG_CHANNEL_LIST, TUHTUEG_ADJACENCY_LIST)

TUHTUEG_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    TUHTUEG_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT)

TUHTUEG_GENERAL_REGION_LIST = format_region_channel_list(
    TUHTUEG_CHANNEL_LIST, GENERAL_REGION_LIST)
TUHTUEG_FRONTAL_REGION_LIST = format_region_channel_list(
    TUHTUEG_CHANNEL_LIST, FRONTAL_REGION_LIST)
TUHTUEG_HEMISPHERE_REGION_LIST = format_region_channel_list(
    TUHTUEG_CHANNEL_LIST, HEMISPHERE_REGION_LIST)