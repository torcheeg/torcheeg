from ..region_1020 import (FRONTAL_REGION_LIST, GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST)
from ..standard_1005 import STANDARD_1005_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict, format_region_channel_list)

M3CV_CHANNEL_LIST = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
    'T7', 'T8', 'P7', 'P8', 'FZ', 'CZ', 'PZ', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5',
    'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2',
    'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5',
    'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8',
    'PO7', 'PO8', 'FPZ', 'CPZ', 'POZ', 'OZ', 'FCZ'
]

M3CV_LOCATION_LIST = [
    ['-', '-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-', '-'],
    ['-', '-', 'AF7', '-', 'AF3', '-', 'AF4', '-', 'AF8', '-', '-'],
    ['-', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', '-'],
    [
        'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'FT10'
    ], ['-', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', '-'],
    [
        'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        'TP10'
    ], ['-', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', '-'],
    ['-', '-', 'PO7', '-', 'PO3', 'POZ', 'PO4', '-', 'PO8', '-', '-'],
    ['-', '-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-', '-']
]

M3CV_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    M3CV_CHANNEL_LIST, M3CV_LOCATION_LIST)

M3CV_ADJACENCY_LIST = {
    'FP1': ['AF3', 'FPZ'],
    'FPZ': ['FP1', 'AF3', 'FP2', 'AF4'],
    'FP2': ['FPZ', 'AF4'],
    'AF7': ['F7', 'F5', 'F3'],
    'AF3': ['F3', 'FP1', 'F1', 'FPZ', 'FZ'],
    'AF4': ['FPZ', 'FZ', 'FP2', 'F2', 'F4'],
    'AF8': ['F4', 'F6', 'F8'],
    'F7': ['FT9', 'FT7', 'AF7', 'F5', 'FC5'],
    'F5': ['F7', 'FT7', 'AF7', 'FC5', 'F3', 'FC3'],
    'F3': ['AF7', 'F5', 'FC5', 'FC3', 'AF3', 'F1', 'FC1'],
    'F1': ['F3', 'FC3', 'AF3', 'FC1', 'FZ', 'FCZ'],
    'FZ': ['AF3', 'F1', 'FC1', 'FCZ', 'AF4', 'F2', 'FC2'],
    'F2': ['FZ', 'FCZ', 'AF4', 'FC2', 'F4', 'FC4'],
    'F4': ['AF4', 'F2', 'FC2', 'FC4', 'AF8', 'F6', 'FC6'],
    'F6': ['F4', 'FC4', 'AF8', 'FC6', 'F8', 'FT8'],
    'F8': ['AF8', 'F6', 'FC6', 'FT8', 'FT10'],
    'FT9': ['F7', 'FT7', 'T7'],
    'FT7': ['FT9', 'F7', 'T7', 'F5', 'FC5', 'C5'],
    'FC5': ['F7', 'FT7', 'T7', 'F5', 'C5', 'F3', 'FC3', 'C3'],
    'FC3': ['F5', 'FC5', 'C5', 'F3', 'C3', 'F1', 'FC1', 'C1'],
    'FC1': ['F3', 'FC3', 'C3', 'F1', 'C1', 'FZ', 'FCZ', 'CZ'],
    'FCZ': ['F1', 'FC1', 'C1', 'FZ', 'CZ', 'F2', 'FC2', 'C2'],
    'FC2': ['FZ', 'FCZ', 'CZ', 'F2', 'C2', 'F4', 'FC4', 'C4'],
    'FC4': ['F2', 'FC2', 'C2', 'F4', 'C4', 'F6', 'FC6', 'C6'],
    'FC6': ['F4', 'FC4', 'C4', 'F6', 'C6', 'F8', 'FT8', 'T8'],
    'FT8': ['F6', 'FC6', 'C6', 'F8', 'T8', 'FT10'],
    'FT10': ['F8', 'FT8', 'T8'],
    'T7': ['FT9', 'TP9', 'FT7', 'TP7', 'FC5', 'C5', 'CP5'],
    'C5': ['FT7', 'T7', 'TP7', 'FC5', 'CP5', 'FC3', 'C3', 'CP3'],
    'C3': ['FC5', 'C5', 'CP5', 'FC3', 'CP3', 'FC1', 'C1', 'CP1'],
    'C1': ['FC3', 'C3', 'CP3', 'FC1', 'CP1', 'FCZ', 'CZ', 'CPZ'],
    'CZ': ['FC1', 'C1', 'CP1', 'FCZ', 'CPZ', 'FC2', 'C2', 'CP2'],
    'C2': ['FCZ', 'CZ', 'CPZ', 'FC2', 'CP2', 'FC4', 'C4', 'CP4'],
    'C4': ['FC2', 'C2', 'CP2', 'FC4', 'CP4', 'FC6', 'C6', 'CP6'],
    'C6': ['FC4', 'C4', 'CP4', 'FC6', 'CP6', 'FT8', 'T8', 'TP8'],
    'T8': ['FC6', 'C6', 'CP6', 'FT8', 'TP8', 'FT10', 'TP10'],
    'TP9': ['T7', 'TP7', 'P7'],
    'TP7': ['TP9', 'T7', 'P7', 'C5', 'CP5', 'P5'],
    'CP5': ['T7', 'TP7', 'P7', 'C5', 'P5', 'C3', 'CP3', 'P3'],
    'CP3': ['C5', 'CP5', 'P5', 'C3', 'P3', 'C1', 'CP1', 'P1'],
    'CP1': ['C3', 'CP3', 'P3', 'C1', 'P1', 'CZ', 'CPZ', 'PZ'],
    'CPZ': ['C1', 'CP1', 'P1', 'CZ', 'PZ', 'C2', 'CP2', 'P2'],
    'CP2': ['CZ', 'CPZ', 'PZ', 'C2', 'P2', 'C4', 'CP4', 'P4'],
    'CP4': ['C2', 'CP2', 'P2', 'C4', 'P4', 'C6', 'CP6', 'P6'],
    'CP6': ['C4', 'CP4', 'P4', 'C6', 'P6', 'T8', 'TP8', 'P8'],
    'TP8': ['C6', 'CP6', 'P6', 'T8', 'P8', 'TP10'],
    'TP10': ['T8', 'TP8', 'P8'],
    'P7': ['TP9', 'TP7', 'CP5', 'P5', 'PO7'],
    'P5': ['TP7', 'P7', 'CP5', 'PO7', 'CP3', 'P3'],
    'P3': ['CP5', 'P5', 'PO7', 'CP3', 'CP1', 'P1', 'PO3'],
    'P1': ['CP3', 'P3', 'CP1', 'PO3', 'CPZ', 'PZ', 'POZ'],
    'PZ': ['CP1', 'P1', 'PO3', 'CPZ', 'POZ', 'CP2', 'P2', 'PO4'],
    'P2': ['CPZ', 'PZ', 'POZ', 'CP2', 'PO4', 'CP4', 'P4'],
    'P4': ['CP2', 'P2', 'PO4', 'CP4', 'CP6', 'P6', 'PO8'],
    'P6': ['CP4', 'P4', 'CP6', 'PO8', 'TP8', 'P8'],
    'P8': ['CP6', 'P6', 'PO8', 'TP8', 'TP10'],
    'PO7': ['P7', 'P5', 'P3'],
    'PO3': ['P3', 'P1', 'O1', 'PZ', 'POZ', 'OZ'],
    'POZ': ['P1', 'PO3', 'O1', 'PZ', 'OZ', 'P2', 'PO4', 'O2'],
    'PO4': ['PZ', 'POZ', 'OZ', 'P2', 'O2', 'P4'],
    'PO8': ['P4', 'P6', 'P8'],
    'O1': ['PO3', 'POZ', 'OZ'],
    'OZ': ['PO3', 'O1', 'POZ', 'PO4', 'O2'],
    'O2': ['POZ', 'OZ', 'PO4']
}

M3CV_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(M3CV_CHANNEL_LIST,
                                                        M3CV_ADJACENCY_LIST)

M3CV_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    M3CV_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT)

M3CV_GENERAL_REGION_LIST = format_region_channel_list(M3CV_CHANNEL_LIST,
                                                      GENERAL_REGION_LIST)
M3CV_FRONTAL_REGION_LIST = format_region_channel_list(M3CV_CHANNEL_LIST,
                                                      FRONTAL_REGION_LIST)
M3CV_HEMISPHERE_REGION_LIST = format_region_channel_list(
    M3CV_CHANNEL_LIST, HEMISPHERE_REGION_LIST)
