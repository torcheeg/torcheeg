from ..standard_1020 import (
    STANDARD_1020_CHANNEL_LOCATION_DICT,
)
from ..utils import (
    format_adj_matrix_from_adj_list,
    format_adj_matrix_from_standard,
    format_channel_location_dict,
)

SHU_MI_CHANNEL_LIST = [
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'CZ', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6',
    'PZ', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ', 'O1', 'O2'
]

SHU_MI_LOCATION_LIST = [
    ['-', '-', '-', 'FP1', '-', 'FP2', '-', '-', '-'],
    ['F7', '-', 'F3', '-', 'FZ', '-', 'F4', '-', 'F8'],
    ['-', 'FC5', '-', 'FC1', '-', 'FC2', '-', 'FC6', '-'],
    ['T3', '-', 'C3', '-', 'CZ', '-', 'C4', '-', 'T4'],
    ['-', 'CP5', '-', 'CP1', '-', 'CP2', '-', 'CP6', '-'],
    ['T5', '-', 'P3', '-', 'PZ', '-', 'P4', '-', 'T6'],
    ['-', '-', '-', 'PO3', '-', 'PO4', '-', '-', '-'],
    ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-'],
    ['A1', '-', '-', '-', '-', '-', '-', '-', 'A2'],
]

SHU_MI_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    SHU_MI_CHANNEL_LIST, SHU_MI_LOCATION_LIST)

SHU_MI_ADJACENCY_LIST = {
    'FP1': ['F3', 'FZ'],
    'FP2': ['FZ', 'F4'],
    'F7': ['FC5'],
    'F3': ['FC1', 'FC5'],
    'FZ': ['FC2', 'FC1'],
    'F4': ['FC6', 'FC2'],
    'F8': ['FC6'],
    'FC5': ['F7', 'F3', 'C3', 'T3'],
    'FC1': ['F3', 'FZ', 'CZ', 'C3'],
    'FC2': ['FZ', 'F4', 'C4', 'CZ'],
    'FC6': ['F4', 'F8', 'T4', 'C4'],
    'T3': ['FC5', 'CP5'],
    'C3': ['FC5', 'FC1', 'CP1', 'CP5'],
    'CZ': ['FC1', 'FC2', 'CP2', 'CP1'],
    'C4': ['FC2', 'FC6', 'CP6', 'CP2'],
    'T4': ['FC6', 'CP6'],
    'CP5': ['T3', 'C3', 'P3', 'T5'],
    'CP1': ['C3', 'CZ', 'PZ', 'P3'],
    'CP2': ['CZ', 'C4', 'P4', 'PZ'],
    'CP6': ['C4', 'T4', 'T6', 'P4'],
    'T5': ['CP5'],
    'P3': ['CP5', 'CP1', 'PO3'],
    'PZ': ['CP1', 'CP2', 'PO4', 'PO3'],
    'P4': ['CP2', 'CP6', 'PO4'],
    'T6': ['CP6'],
    'PO3': ['P3', 'PZ', 'OZ', 'O1'],
    'PO4': ['PZ', 'P4', 'O2', 'OZ'],
    'O1': ['PO3', 'OZ'],
    'OZ': ['PO3', 'PO4', 'O2', 'O1'],
    'O2': ['PO4', 'OZ'],
    'A1': [],
    'A2': [],
}

SHU_MI_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    SHU_MI_CHANNEL_LIST, SHU_MI_ADJACENCY_LIST)

SHU_MI_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    SHU_MI_CHANNEL_LIST, STANDARD_1020_CHANNEL_LOCATION_DICT)
