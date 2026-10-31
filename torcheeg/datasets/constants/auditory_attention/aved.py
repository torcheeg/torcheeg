from ..region_1020 import (FRONTAL_REGION_LIST, GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST)
from ..standard_1020 import STANDARD_1020_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_list_from_location_list,
                     format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict, format_region_channel_list)

AVED_CHANNEL_LIST = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3',
                     'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'PZ', 'P4', 'T6', 'A2', 'O1', 'OZ', 'O2']

AVED_LOCATION_LIST = [
    ['-', '-', '-', 'FP1', '-', 'FP2', '-', '-', '-'],
    ['F7', '-', 'F3', '-', 'FZ', '-', 'F4', '-', 'F8'],
    ['FT7', '-', 'FC3', '-', 'FCZ', '-', 'FC4', '-', 'FT8'],
    ['T3', '-', 'C3', '-', 'CZ', '-', 'C4', '-', 'T4'],
    ['TP7', '-', 'CP3', '-', 'CPZ', '-', 'CP4', '-', 'TP8'],
    ['T5', '-', 'P3', '-', 'PZ', '-', 'P4', '-', 'T6'],
    ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-'],
    ['A1', '-', '-', '-', '-', '-', '-', '-', 'A2']
]

AVED_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    AVED_CHANNEL_LIST, AVED_LOCATION_LIST)

AVED_ADJACENCY_LIST = format_adj_list_from_location_list(AVED_LOCATION_LIST)

AVED_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    AVED_CHANNEL_LIST, AVED_ADJACENCY_LIST)

AVED_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    AVED_CHANNEL_LIST, STANDARD_1020_CHANNEL_LOCATION_DICT, delta=0.00035)

AVED_GENERAL_REGION_LIST = format_region_channel_list(AVED_CHANNEL_LIST,
                                                      GENERAL_REGION_LIST)
AVED_FRONTAL_REGION_LIST = format_region_channel_list(AVED_CHANNEL_LIST,
                                                      FRONTAL_REGION_LIST)
AVED_HEMISPHERE_REGION_LIST = format_region_channel_list(
    AVED_CHANNEL_LIST, HEMISPHERE_REGION_LIST)

AVED_RAW_CHANNEL_LIST = ['HEOL', 'HEOR', 'FP1', 'FP2', 'VEOU', 'VEOL', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4',
                         'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'PZ', 'P4', 'T6', 'A2', 'O1', 'OZ', 'O2']
