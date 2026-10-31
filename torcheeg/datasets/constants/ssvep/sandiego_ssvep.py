from ..standard_1005 import STANDARD_1005_CHANNEL_LOCATION_DICT
from ..utils import (format_adj_list_from_location_list,
                     format_adj_matrix_from_adj_list,
                     format_adj_matrix_from_standard,
                     format_channel_location_dict)

SANDIEGOSSVEP_CHANNEL_LIST = ['PO7', 'PO3',
                              'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']
SANDIEGOSSVEP_LOCATION_LIST = [
    ['PO7', 'PO3', 'POZ', 'PO4', 'PO8'], ['-', 'O1', 'OZ', 'O2', '-']]
SANDIEGOSSVEP_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    SANDIEGOSSVEP_CHANNEL_LIST, SANDIEGOSSVEP_LOCATION_LIST)
SANDIEGOSSVEP_ADJACENCY_LIST = format_adj_list_from_location_list(
    SANDIEGOSSVEP_LOCATION_LIST)
SANDIEGOSSVEP_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    SANDIEGOSSVEP_CHANNEL_LIST, SANDIEGOSSVEP_ADJACENCY_LIST)
SANDIEGOSSVEP_STANDARD_ADJACENCY_MATRIX = format_adj_matrix_from_standard(
    SANDIEGOSSVEP_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT)
SANDIEGOSSVEP_FREQUENCIES = [9.25, 11.25, 13.25, 9.75,
                             11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
