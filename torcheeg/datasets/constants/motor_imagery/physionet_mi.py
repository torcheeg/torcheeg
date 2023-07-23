from ..region_1020 import (FRONTAL_REGION_LIST, GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST)
from ..standard_1020 import STANDARD_1020_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict, format_region_channel_list)

PHYSIONETMI_CHANNEL_LIST = [
    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ',
    'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1',
    'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1',
    'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3',
    'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ'
]

PHYSIONETMI_LOCATION_LIST = [
    ['-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-'],
    ['-', 'AF7', '-', 'AF3', 'AFZ', 'AF4', '-', 'AF8', '-'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['-', 'PO7', '-', 'PO3', 'POZ', 'PO4', '-', 'PO8', '-'],
    ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-']
]
# without {'T10', 'T9', 'IZ'}

PHYSIONETMI_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    PHYSIONETMI_CHANNEL_LIST, PHYSIONETMI_LOCATION_LIST)

PHYSIONETMI_ADJACENCY_LIST = {
    'FP1': ['AF3', 'FPZ', 'AFZ'],
    'FPZ': ['AFZ', 'FP1', 'FP2', 'AF4', 'AF3'],
    'FP2': ['AF4', 'FPZ', 'AFZ'],
    'AF7': ['F5', 'F3', 'F7'],
    'AF3': ['FP1', 'F1', 'AFZ', 'FPZ', 'FZ', 'F3'],
    'AFZ': ['FPZ', 'FZ', 'AF3', 'AF4', 'FP2', 'F2', 'FP1', 'F1'],
    'AF4': ['FP2', 'F2', 'AFZ', 'F4', 'FPZ', 'FZ'],
    'AF8': ['F6', 'F8', 'F4'],
    'F7': ['FT7', 'F5', 'AF7', 'FC5'],
    'F5': ['AF7', 'FC5', 'F7', 'F3', 'FC3', 'FT7'],
    'F3': ['FC3', 'F5', 'F1', 'AF3', 'FC1', 'AF7', 'FC5'],
    'F1': ['AF3', 'FC1', 'F3', 'FZ', 'AFZ', 'FCZ', 'FC3'],
    'FZ': ['AFZ', 'FCZ', 'F1', 'F2', 'AF4', 'FC2', 'AF3', 'FC1'],
    'F2': ['AF4', 'FC2', 'FZ', 'F4', 'FC4', 'AFZ', 'FCZ'],
    'F4': ['FC4', 'F2', 'F6', 'AF8', 'FC6', 'AF4', 'FC2'],
    'F6': ['AF8', 'FC6', 'F4', 'F8', 'FT8', 'FC4'],
    'F8': ['FT8', 'F6', 'AF8', 'FC6'],
    'FT7': ['F7', 'T7', 'FC5', 'F5', 'C5'],
    'FC5': ['F5', 'C5', 'FT7', 'FC3', 'F3', 'C3', 'F7', 'T7'],
    'FC3': ['F3', 'C3', 'FC5', 'FC1', 'F1', 'C1', 'F5', 'C5'],
    'FC1': ['F1', 'C1', 'FC3', 'FCZ', 'FZ', 'CZ', 'F3', 'C3'],
    'FCZ': ['FZ', 'CZ', 'FC1', 'FC2', 'F2', 'C2', 'F1', 'C1'],
    'FC2': ['F2', 'C2', 'FCZ', 'FC4', 'F4', 'C4', 'FZ', 'CZ'],
    'FC4': ['F4', 'C4', 'FC2', 'FC6', 'F6', 'C6', 'F2', 'C2'],
    'FC6': ['F6', 'C6', 'FC4', 'FT8', 'F8', 'T8', 'F4', 'C4'],
    'FT8': ['F8', 'T8', 'FC6', 'F6', 'C6'],
    'T7': ['FT7', 'TP7', 'C5', 'FC5', 'CP5'],
    'C5': ['FC5', 'CP5', 'T7', 'C3', 'FC3', 'CP3', 'FT7', 'TP7'],
    'C3': ['FC3', 'CP3', 'C5', 'C1', 'FC1', 'CP1', 'FC5', 'CP5'],
    'C1': ['FC1', 'CP1', 'C3', 'CZ', 'FCZ', 'CPZ', 'FC3', 'CP3'],
    'CZ': ['FCZ', 'CPZ', 'C1', 'C2', 'FC2', 'CP2', 'FC1', 'CP1'],
    'C2': ['FC2', 'CP2', 'CZ', 'C4', 'FC4', 'CP4', 'FCZ', 'CPZ'],
    'C4': ['FC4', 'CP4', 'C2', 'C6', 'FC6', 'CP6', 'FC2', 'CP2'],
    'C6': ['FC6', 'CP6', 'C4', 'T8', 'FT8', 'TP8', 'FC4', 'CP4'],
    'T8': ['FT8', 'TP8', 'C6', 'FC6', 'CP6'],
    'TP7': ['T7', 'P7', 'CP5', 'C5', 'P5'],
    'CP5': ['C5', 'P5', 'TP7', 'CP3', 'C3', 'P3', 'T7', 'P7'],
    'CP3': ['C3', 'P3', 'CP5', 'CP1', 'C1', 'P1', 'C5', 'P5'],
    'CP1': ['C1', 'P1', 'CP3', 'CPZ', 'CZ', 'PZ', 'C3', 'P3'],
    'CPZ': ['CZ', 'PZ', 'CP1', 'CP2', 'C2', 'P2', 'C1', 'P1'],
    'CP2': ['C2', 'P2', 'CPZ', 'CP4', 'C4', 'P4', 'CZ', 'PZ'],
    'CP4': ['C4', 'P4', 'CP2', 'CP6', 'C6', 'P6', 'C2', 'P2'],
    'CP6': ['C6', 'P6', 'CP4', 'TP8', 'T8', 'P8', 'C4', 'P4'],
    'TP8': ['T8', 'P8', 'CP6', 'C6', 'P6'],
    'P7': ['TP7', 'P5', 'CP5', 'PO7'],
    'P5': ['CP5', 'PO7', 'P7', 'P3', 'CP3', 'TP7'],
    'P3': ['CP3', 'P5', 'P1', 'CP1', 'PO3', 'CP5', 'PO7'],
    'P1': ['CP1', 'PO3', 'P3', 'PZ', 'CPZ', 'POZ', 'CP3'],
    'PZ': ['CPZ', 'POZ', 'P1', 'P2', 'CP2', 'PO4', 'CP1', 'PO3'],
    'P2': ['CP2', 'PO4', 'PZ', 'P4', 'CP4', 'CPZ', 'POZ'],
    'P4': ['CP4', 'P2', 'P6', 'CP6', 'PO8', 'CP2', 'PO4'],
    'P6': ['CP6', 'PO8', 'P4', 'P8', 'TP8', 'CP4'],
    'P8': ['TP8', 'P6', 'CP6', 'PO8'],
    'PO7': ['P5', 'P3', 'P7'],
    'PO3': ['P1', 'O1', 'POZ', 'PZ', 'OZ', 'P3'],
    'POZ': ['PZ', 'OZ', 'PO3', 'PO4', 'P2', 'O2', 'P1', 'O1'],
    'PO4': ['P2', 'O2', 'POZ', 'P4', 'PZ', 'OZ'],
    'PO8': ['P6', 'P8', 'P4'],
    'O1': ['PO3', 'OZ', 'POZ'],
    'OZ': ['POZ', 'O1', 'O2', 'PO4', 'PO3'],
    'O2': ['PO4', 'OZ', 'POZ']
}

PHYSIONETMI_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    PHYSIONETMI_CHANNEL_LIST, PHYSIONETMI_ADJACENCY_LIST)

PHYSIONETMI_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    PHYSIONETMI_CHANNEL_LIST,
    STANDARD_1020_CHANNEL_LOCATION_DICT,
    delta=0.00035)

PHYSIONETMI_GENERAL_REGION_LIST = format_region_channel_list(
    PHYSIONETMI_CHANNEL_LIST, GENERAL_REGION_LIST)
PHYSIONETMI_FRONTAL_REGION_LIST = format_region_channel_list(
    PHYSIONETMI_CHANNEL_LIST, FRONTAL_REGION_LIST)
PHYSIONETMI_HEMISPHERE_REGION_LIST = format_region_channel_list(
    PHYSIONETMI_CHANNEL_LIST, HEMISPHERE_REGION_LIST)
