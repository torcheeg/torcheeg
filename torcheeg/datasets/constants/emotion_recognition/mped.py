from ..region_1020 import (FRONTAL_REGION_LIST, GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST)
from ..standard_1005 import STANDARD_1005_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict, format_region_channel_list)

MPED_CHANNEL_LIST = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

MPED_LOCATION_LIST = [
    ['-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-'],
    ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-'],
    ['-', '-', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '-', '-']
]

MPED_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    MPED_CHANNEL_LIST, MPED_LOCATION_LIST)

MPED_ADJACENCY_LIST = {
    'FP1': ['FPZ', 'AF3'],
    'FPZ': ['FP1', 'FP2'],
    'FP2': ['FPZ', 'AF4'],
    'AF3': ['FP1', 'F5', 'F3', 'F1'],
    'AF4': ['F2', 'F4', 'F6', 'FP2'],
    'F7': ['F5', 'FT7'],
    'F5': ['F7', 'AF3', 'F3', 'FC5'],
    'F3': ['AF3', 'F5', 'FC3', 'F1'],
    'F1': ['AF3', 'F3', 'FC1', 'FZ'],
    'FZ': ['F1', 'FCZ', 'F2'],
    'F2': ['FZ', 'FC2', 'F4', 'AF4'],
    'F4': ['F2', 'FC4', 'F6', 'AF4'],
    'F6': ['AF4', 'F4', 'FC6', 'F8'],
    'F8': ['F6', 'FT8'],
    'FT7': ['F7', 'FC5', 'T7'],
    'FC5': ['F5', 'FT7', 'C5', 'FC3'],
    'FC3': ['F3', 'FC5', 'C3', 'FC1'],
    'FC1': ['F1', 'FC3', 'C1', 'FCZ'],
    'FCZ': ['FZ', 'FC1', 'CZ', 'FC2'],
    'FC2': ['F2', 'FCZ', 'C2', 'FC4'],
    'FC4': ['F4', 'FC2', 'C4', 'FC6'],
    'FC6': ['F6', 'FC4', 'C6', 'FT8'],
    'FT8': ['F8', 'FC6', 'T8'],
    'T7': ['FT7', 'C5', 'TP7'],
    'C5': ['FC5', 'T7', 'C3', 'CP5'],
    'C3': ['FC3', 'C5', 'C1', 'CP3'],
    'C1': ['FC1', 'C3', 'CP1', 'CZ'],
    'CZ': ['FCZ', 'C1', 'CPZ', 'C2'],
    'C2': ['FC2', 'CZ', 'CP2', 'C4'],
    'C4': ['FC4', 'C2', 'CP4', 'C6'],
    'C6': ['FC6', 'C4', 'CP6', 'T8'],
    'T8': ['FT8', 'C6', 'TP8'],
    'TP7': ['T7', 'CP5', 'P7'],
    'CP5': ['C5', 'TP7', 'P5', 'CP3'],
    'CP3': ['C3', 'CP5', 'P3', 'CP1'],
    'CP1': ['C1', 'CP3', 'P1', 'CPZ'],
    'CPZ': ['CZ', 'CP1', 'PZ', 'CP2'],
    'CP2': ['C2', 'CPZ', 'P2', 'CP4'],
    'CP4': ['C4', 'CP2', 'P4', 'CP6'],
    'CP6': ['C6', 'CP4', 'P6', 'TP8'],
    'TP8': ['T8', 'CP6', 'P8'],
    'P7': ['TP7', 'P5', 'PO7'],
    'P5': ['CP5', 'P7', 'PO5', 'P3'],
    'P3': ['CP3', 'P5', 'P1'],
    'P1': ['CP1', 'P3', 'PO3', 'PZ'],
    'PZ': ['CPZ', 'P1', 'POZ', 'P2'],
    'P2': ['CP2', 'PZ', 'PO4', 'P4'],
    'P4': ['CP4', 'P2', 'P6'],
    'P6': ['CP6', 'P4', 'P8'],
    'P8': ['TP8', 'P6', 'PO8'],
    'PO7': ['P7', 'PO5', 'CB1'],
    'PO5': ['P5', 'PO7', 'CB1', 'PO3'],
    'PO3': ['P1', 'PO5', 'O1', 'POZ'],
    'POZ': ['PZ', 'PO3', 'OZ', 'PO4'],
    'PO4': ['P2', 'POZ', 'O2', 'PO6'],
    'PO6': ['P6', 'PO4', 'CB2', 'PO8'],
    'PO8': ['P8', 'PO6', 'CB2'],
    'CB1': ['PO7', 'PO5', 'O1'],
    'O1': ['CB1', 'PO3', 'OZ'],
    'OZ': ['POZ', 'O1', 'O2'],
    'O2': ['PO4', 'OZ', 'CB2'],
    'CB2': ['PO6', 'O2', 'PO8']
}

MPED_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(MPED_CHANNEL_LIST,
                                                        MPED_ADJACENCY_LIST)

MPED_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    MPED_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT)

MPED_GENERAL_REGION_LIST = format_region_channel_list(MPED_CHANNEL_LIST,
                                                      GENERAL_REGION_LIST)
MPED_FRONTAL_REGION_LIST = format_region_channel_list(MPED_CHANNEL_LIST,
                                                      FRONTAL_REGION_LIST)
MPED_HEMISPHERE_REGION_LIST = format_region_channel_list(
    MPED_CHANNEL_LIST, HEMISPHERE_REGION_LIST)