from ..utils import format_adj_matrix_from_standard, format_channel_location_dict, format_adj_matrix_from_adj_list
from ..standard_1005 import STANDARD_1005_CHANNEL_LOCATION_DICT

TSUBENCHMARK_CHANNEL_LIST = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5',
    'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3',
    'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4',
    'PO6', 'PO8', 'POO7', 'O1', 'OZ', 'O2', 'POO8'
]

TSUBENCHMARK_LOCATION_LIST = [
    ['-', '-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-', '-'],
    ['-', '-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-', '-'],
    ['-', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', '-'],
    ['-', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', '-'],
    ['-', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', '-'],
    [
        'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        'TP10'
    ], ['-', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', '-'],
    ['-', '-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-', '-'],
    ['-', '-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-', '-'],
    ['-', '-', '-', '-', 'POO7', '-', 'POO8', '-', '-', '-', '-']
]

TSUBENCHMARK_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    TSUBENCHMARK_CHANNEL_LIST, TSUBENCHMARK_LOCATION_LIST)

TSUBENCHMARK_ADJACENCY_LIST = {
    'FP1': ['FPZ', 'AF3'],
    'FPZ': ['FP1', 'FP2', 'AF3', 'AF4'],
    'FP2': ['FPZ', 'AF4'],
    'AF3': ['F1', 'FP1', 'F3', 'FZ', 'FPZ'],
    'AF4': ['F2', 'FP2', 'FZ', 'F4', 'FPZ'],
    'F7': ['F5', 'FT7', 'FC5'],
    'F5': ['F7', 'F3', 'FC5', 'FT7', 'FC3'],
    'F3': ['F5', 'F1', 'FC3', 'FC5', 'FC1', 'AF3'],
    'F1': ['F3', 'FZ', 'FC1', 'AF3', 'FC3', 'FCZ'],
    'FZ': ['F1', 'F2', 'FCZ', 'FC1', 'FC2', 'AF3', 'AF4'],
    'F2': ['FZ', 'F4', 'FC2', 'AF4', 'FCZ', 'FC4'],
    'F4': ['F2', 'F6', 'FC4', 'FC2', 'FC6', 'AF4'],
    'F6': ['F4', 'F8', 'FC6', 'FC4', 'FT8'],
    'F8': ['F6', 'FT8', 'FC6'],
    'FT7': ['FC5', 'T7', 'F7', 'C5', 'F5'],
    'FC5': ['FT7', 'FC3', 'C5', 'F5', 'T7', 'C3', 'F7', 'F3'],
    'FC3': ['FC5', 'FC1', 'C3', 'F3', 'C5', 'C1', 'F5', 'F1'],
    'FC1': ['FC3', 'FCZ', 'C1', 'F1', 'C3', 'CZ', 'F3', 'FZ'],
    'FCZ': ['FC1', 'FC2', 'CZ', 'FZ', 'C1', 'C2', 'F1', 'F2'],
    'FC2': ['FCZ', 'FC4', 'C2', 'F2', 'CZ', 'C4', 'FZ', 'F4'],
    'FC4': ['FC2', 'FC6', 'C4', 'F4', 'C2', 'C6', 'F2', 'F6'],
    'FC6': ['FC4', 'FT8', 'C6', 'F6', 'C4', 'T8', 'F4', 'F8'],
    'FT8': ['FC6', 'T8', 'F8', 'C6', 'F6'],
    'T7': ['C5', 'TP7', 'FT7', 'TP9', 'CP5', 'FC5'],
    'C5': ['T7', 'C3', 'CP5', 'FC5', 'TP7', 'CP3', 'FT7', 'FC3'],
    'C3': ['C5', 'C1', 'CP3', 'FC3', 'CP5', 'CP1', 'FC5', 'FC1'],
    'C1': ['C3', 'CZ', 'CP1', 'FC1', 'CP3', 'CPZ', 'FC3', 'FCZ'],
    'CZ': ['C1', 'C2', 'CPZ', 'FCZ', 'CP1', 'CP2', 'FC1', 'FC2'],
    'C2': ['CZ', 'C4', 'CP2', 'FC2', 'CPZ', 'CP4', 'FCZ', 'FC4'],
    'C4': ['C2', 'C6', 'CP4', 'FC4', 'CP2', 'CP6', 'FC2', 'FC6'],
    'C6': ['C4', 'T8', 'CP6', 'FC6', 'CP4', 'TP8', 'FC4', 'FT8'],
    'T8': ['C6', 'TP8', 'FT8', 'CP6', 'TP10', 'FC6'],
    'TP9': ['TP7', 'P7', 'T7'],
    'TP7': ['TP9', 'CP5', 'P7', 'T7', 'P5', 'C5'],
    'CP5': ['TP7', 'CP3', 'P5', 'C5', 'P7', 'P3', 'T7', 'C3'],
    'CP3': ['CP5', 'CP1', 'P3', 'C3', 'P5', 'P1', 'C5', 'C1'],
    'CP1': ['CP3', 'CPZ', 'P1', 'C1', 'P3', 'PZ', 'C3', 'CZ'],
    'CPZ': ['CP1', 'CP2', 'PZ', 'CZ', 'P1', 'P2', 'C1', 'C2'],
    'CP2': ['CPZ', 'CP4', 'P2', 'C2', 'PZ', 'P4', 'CZ', 'C4'],
    'CP4': ['CP2', 'CP6', 'P4', 'C4', 'P2', 'P6', 'C2', 'C6'],
    'CP6': ['CP4', 'TP8', 'P6', 'C6', 'P4', 'P8', 'C4', 'T8'],
    'TP8': ['CP6', 'TP10', 'P8', 'T8', 'P6', 'C6'],
    'TP10': ['TP8', 'P8', 'T8'],
    'P7': ['P5', 'TP7', 'PO7', 'TP9', 'CP5'],
    'P5': ['P7', 'P3', 'PO7', 'CP5', 'PO5', 'TP7', 'CP3'],
    'P3': ['P5', 'P1', 'PO5', 'CP3', 'PO7', 'PO3', 'CP5', 'CP1'],
    'P1': ['P3', 'PZ', 'PO3', 'CP1', 'PO5', 'POZ', 'CP3', 'CPZ'],
    'PZ': ['P1', 'P2', 'POZ', 'CPZ', 'PO3', 'PO4', 'CP1', 'CP2'],
    'P2': ['PZ', 'P4', 'PO4', 'CP2', 'POZ', 'PO6', 'CPZ', 'CP4'],
    'P4': ['P2', 'P6', 'PO6', 'CP4', 'PO4', 'PO8', 'CP2', 'CP6'],
    'P6': ['P4', 'P8', 'PO8', 'CP6', 'PO6', 'CP4', 'TP8'],
    'P8': ['P6', 'TP8', 'PO8', 'CP6', 'TP10'],
    'PO7': ['PO5', 'P5', 'P7', 'P3'],
    'PO5': ['PO7', 'PO3', 'P3', 'O1', 'P5', 'P1'],
    'PO3': ['PO5', 'POZ', 'O1', 'P1', 'OZ', 'P3', 'PZ'],
    'POZ': ['PO3', 'PO4', 'OZ', 'PZ', 'O1', 'O2', 'P1', 'P2'],
    'PO4': ['POZ', 'PO6', 'O2', 'P2', 'OZ', 'PZ', 'P4'],
    'PO6': ['PO4', 'PO8', 'P4', 'O2', 'P2', 'P6'],
    'PO8': ['PO6', 'P6', 'P4', 'P8'],
    'O1': ['OZ', 'POO7', 'PO3', 'PO5', 'POZ'],
    'OZ': ['O1', 'O2', 'POZ', 'POO7', 'POO8', 'PO3', 'PO4'],
    'O2': ['OZ', 'POO8', 'PO4', 'POZ', 'PO6'],
    'POO7': ['O1', 'OZ'],
    'POO8': ['O2', 'OZ']
}

TSUBENCHMARK_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    TSUBENCHMARK_CHANNEL_LIST, TSUBENCHMARK_ADJACENCY_LIST)

TSUBENCHMARK_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    TSUBENCHMARK_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT)
