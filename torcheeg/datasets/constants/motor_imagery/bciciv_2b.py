from ..region_1020 import (
    FRONTAL_REGION_LIST,
    GENERAL_REGION_LIST,
    HEMISPHERE_REGION_LIST,
)
from ..standard_1020 import (
    STANDARD_1020_CHANNEL_LOCATION_DICT,
)
from ..utils import (
    format_adj_list_from_location_list,
    format_adj_matrix_from_adj_list,
    format_adj_matrix_from_standard,
    format_channel_location_dict,
    format_region_channel_list,
)

BCICIV2B_CHANNEL_LIST = ['C3', 'CZ', 'C4']

BCICIV2B_LOCATION_LIST = [
    ['-', '-', '-',  '-',  '-', '-', '-'],
    ['-', '-', '-',  '-',  '-', '-', '-'],
    ['-', '-', '-',  '-',  '-', '-', '-'],
    ['-', '-', 'C3', 'CZ', 'C4', '-', '-'],
    ['-', '-', '-',  '-',  '-', '-', '-'],
    ['-', '-', '-',  '-',  '-', '-', '-'],
    ['-', '-', '-',  '-',  '-', '-', '-'],
]

BCICIV2B_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    BCICIV2B_CHANNEL_LIST, BCICIV2B_LOCATION_LIST)

BCICIV2B_ADJACENCY_LIST = format_adj_list_from_location_list(
    BCICIV2B_LOCATION_LIST)

BCICIV2B_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    BCICIV2B_CHANNEL_LIST, BCICIV2B_ADJACENCY_LIST)

BCICIV2B_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    BCICIV2B_CHANNEL_LIST, STANDARD_1020_CHANNEL_LOCATION_DICT, delta=0.00035)

BCICIV2B_GENERAL_REGION_LIST = format_region_channel_list(BCICIV2B_CHANNEL_LIST,
                                                          GENERAL_REGION_LIST)
BCICIV2B_FRONTAL_REGION_LIST = format_region_channel_list(BCICIV2B_CHANNEL_LIST,
                                                          FRONTAL_REGION_LIST)
BCICIV2B_HEMISPHERE_REGION_LIST = format_region_channel_list(
    BCICIV2B_CHANNEL_LIST, HEMISPHERE_REGION_LIST)
