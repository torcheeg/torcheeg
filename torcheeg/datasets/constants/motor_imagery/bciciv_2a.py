from ..region_1020 import (FRONTAL_REGION_LIST, GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST)
from ..standard_1020 import STANDARD_1020_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_list_from_location_list,
                     format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict, format_region_channel_list)

BCICIV2A_CHANNEL_LIST = [
    'FZ', '0', '1', '2', '3', '4', '5', 'C3', '6', 'CZ', '7', 'C4', '8', '9', '10', '11', '12', '13', '14', 'PZ', '15', '16'
]
BCICIV2A_LOCATION_LIST = [['-',  '-',  '-',  'FZ', '-',  '-',  '-'],
                          ['-',  '0',  '1',  '2',  '3',  '4',  '-'],
                          ['5',  'C3', '6',  'CZ', '7',  'C4', '8'],
                          ['-',  '9',  '10', '11', '12', '13', '-'],
                          ['-',  '-',  '14', 'PZ', '15', '-',  '-'],
                          ['-',  '-',  '-',  '16', '-',  '-',  '-']]
BCICIV2A_LOCATION_DICT = format_channel_location_dict(BCICIV2A_CHANNEL_LIST,BCICIV2A_LOCATION_LIST)


# Physical names follow the same 22 EEG columns as the legacy grid aliases.
BCICIV2A_STANDARD_CHANNEL_LIST = [
    'FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2',
    'C4', 'C6', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'PZ', 'P2', 'POZ'
]

BCICIV2A_ADJACENCY_LIST = format_adj_list_from_location_list(
    BCICIV2A_LOCATION_LIST)
BCICIV2A_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    BCICIV2A_CHANNEL_LIST, BCICIV2A_ADJACENCY_LIST)
BCICIV2A_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    BCICIV2A_STANDARD_CHANNEL_LIST, STANDARD_1020_CHANNEL_LOCATION_DICT,
    delta=0.00035)

BCICIV2A_GENERAL_REGION_LIST = format_region_channel_list(
    BCICIV2A_STANDARD_CHANNEL_LIST, GENERAL_REGION_LIST)
BCICIV2A_FRONTAL_REGION_LIST = format_region_channel_list(
    BCICIV2A_STANDARD_CHANNEL_LIST, FRONTAL_REGION_LIST)
BCICIV2A_HEMISPHERE_REGION_LIST = format_region_channel_list(
    BCICIV2A_STANDARD_CHANNEL_LIST, HEMISPHERE_REGION_LIST)
