from ..region_1020 import (FRONTAL_REGION_LIST, GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST)
from ..standard_1020 import STANDARD_1020_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_list_from_location_list,
                     format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict, format_region_channel_list)

DTU_CHANNEL_LIST = ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ',
                    'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

DTU_LOCATION_LIST = [
    ['-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-'],
    ['-', 'AF7', 'AF3', '-', 'AFZ', '-', 'AF4', 'AF8', '-'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['P9', 'PO7', 'PO3', '-', 'POZ', '-', 'PO4', 'PO8', 'P10'],
    ['-', '-', 'O1', '-', 'OZ', '-', 'O2', '-', '-'],
    ['-', '-', '-', '-', 'IZ', '-', '-', '-', '-']
]

DTU_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    DTU_CHANNEL_LIST, DTU_LOCATION_LIST)

DTU_ADJACENCY_LIST = format_adj_list_from_location_list(DTU_LOCATION_LIST)

DTU_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    DTU_CHANNEL_LIST, DTU_ADJACENCY_LIST)

DTU_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    DTU_CHANNEL_LIST, STANDARD_1020_CHANNEL_LOCATION_DICT, delta=0.00035)

DTU_GENERAL_REGION_LIST = format_region_channel_list(DTU_CHANNEL_LIST,
                                                     GENERAL_REGION_LIST)
DTU_FRONTAL_REGION_LIST = format_region_channel_list(DTU_CHANNEL_LIST,
                                                     FRONTAL_REGION_LIST)
DTU_HEMISPHERE_REGION_LIST = format_region_channel_list(
    DTU_CHANNEL_LIST, HEMISPHERE_REGION_LIST)

DTU_MASTOID_CHANNEL_LIST = ['EXG1', 'EXG2']
