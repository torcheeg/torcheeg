from ..region_1020 import (FRONTAL_REGION_LIST, GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST)
from ..standard_1020 import STANDARD_1020_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict, format_region_channel_list)

DEAP_CHANNEL_LIST = [
    'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3',
    'P7', 'PO3', 'O1', 'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2',
    'CZ', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]

DEAP_LOCATION_LIST = [['-', '-', '-', 'FP1', '-', 'FP2', '-', '-', '-'],
                      ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
                      ['F7', '-', 'F3', '-', 'FZ', '-', 'F4', '-', 'F8'],
                      ['-', 'FC5', '-', 'FC1', '-', 'FC2', '-', 'FC6', '-'],
                      ['T7', '-', 'C3', '-', 'CZ', '-', 'C4', '-', 'T8'],
                      ['-', 'CP5', '-', 'CP1', '-', 'CP2', '-', 'CP6', '-'],
                      ['P7', '-', 'P3', '-', 'PZ', '-', 'P4', '-', 'P8'],
                      ['-', '-', '-', 'PO3', '-', 'PO4', '-', '-', '-'],
                      ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-']]

DEAP_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    DEAP_CHANNEL_LIST, DEAP_LOCATION_LIST)

DEAP_ADJACENCY_LIST = {
    'FP1': ['AF3'],
    'FP2': ['AF4'],
    'AF3': ['FP1', 'FZ', 'F3'],
    'AF4': ['FP2', 'F4', 'FZ'],
    'F7': ['FC5'],
    'F3': ['AF3', 'FC1', 'FC5'],
    'FZ': ['AF3', 'AF4', 'FC2', 'FC1'],
    'F4': ['AF4', 'FC6', 'FC2'],
    'F8': ['FC6'],
    'FC5': ['F7', 'F3', 'C3', 'T7'],
    'FC1': ['F3', 'FZ', 'CZ', 'C3'],
    'FC2': ['FZ', 'F4', 'C4', 'CZ'],
    'FC6': ['F4', 'F8', 'T8', 'C4'],
    'T7': ['FC5', 'CP5'],
    'C3': ['FC5', 'FC1', 'CP1', 'CP5'],
    'CZ': ['FC1', 'FC2', 'CP2', 'CP1'],
    'C4': ['FC2', 'FC6', 'CP6', 'CP2'],
    'T8': ['FC6', 'CP6'],
    'CP5': ['T7', 'C3', 'P3', 'P7'],
    'CP1': ['C3', 'CZ', 'PZ', 'P3'],
    'CP2': ['CZ', 'C4', 'P4', 'PZ'],
    'CP6': ['C4', 'T8', 'P8', 'P4'],
    'P7': ['CP5'],
    'P3': ['CP5', 'CP1', 'PO3'],
    'PZ': ['CP1', 'CP2', 'PO4', 'PO3'],
    'P4': ['CP2', 'CP6', 'PO4'],
    'P8': ['CP6'],
    'PO3': ['P3', 'PZ', 'OZ', 'O1'],
    'PO4': ['PZ', 'P4', 'O2', 'OZ'],
    'O1': ['PO3', 'OZ'],
    'OZ': ['PO3', 'PO4', 'O2', 'O1'],
    'O2': ['PO4', 'OZ']
}

DEAP_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(DEAP_CHANNEL_LIST,
                                                        DEAP_ADJACENCY_LIST)

DEAP_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    DEAP_CHANNEL_LIST, STANDARD_1020_CHANNEL_LOCATION_DICT)

DEAP_GENERAL_REGION_LIST = format_region_channel_list(DEAP_CHANNEL_LIST,
                                                      GENERAL_REGION_LIST)
DEAP_FRONTAL_REGION_LIST = format_region_channel_list(DEAP_CHANNEL_LIST,
                                                      FRONTAL_REGION_LIST)
DEAP_HEMISPHERE_REGION_LIST = format_region_channel_list(
    DEAP_CHANNEL_LIST, HEMISPHERE_REGION_LIST)
