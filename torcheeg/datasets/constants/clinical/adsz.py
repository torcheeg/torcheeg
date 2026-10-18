from ..utils import (format_adj_matrix_from_adj_list,
                     format_channel_location_dict)

ADSZ_CHANNEL_LIST = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2',
]

ADSZ_LOCATION_LIST = [
    ['-', 'Fp1', '-', 'Fp2', '-'],
    ['F7', 'F3', 'Fz', 'F4', 'F8'],
    ['T3', 'C3', 'Cz', 'C4', 'T4'],
    ['T5', 'P3', 'Pz', 'P4', 'T6'],
    ['-', 'O1', '-', 'O2', '-'],
]

ADSZ_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    ADSZ_CHANNEL_LIST, ADSZ_LOCATION_LIST)

ADSZ_ADJACENCY_LIST = {
    'Fp1': ['Fp2', 'F7', 'F3', 'Fz'],
    'Fp2': ['Fp1', 'Fz', 'F4', 'F8'],
    'F7': ['Fp1', 'F3', 'T3'],
    'F3': ['Fp1', 'F7', 'Fz', 'C3'],
    'Fz': ['Fp1', 'Fp2', 'F3', 'F4', 'Cz'],
    'F4': ['Fp2', 'Fz', 'F8', 'C4'],
    'F8': ['Fp2', 'F4', 'T4'],
    'T3': ['F7', 'C3', 'T5'],
    'C3': ['F3', 'T3', 'Cz', 'P3'],
    'Cz': ['Fz', 'C3', 'C4', 'Pz'],
    'C4': ['F4', 'Cz', 'T4', 'P4'],
    'T4': ['F8', 'C4', 'T6'],
    'T5': ['T3', 'P3', 'O1'],
    'P3': ['C3', 'T5', 'Pz', 'O1'],
    'Pz': ['Cz', 'P3', 'P4', 'O1', 'O2'],
    'P4': ['C4', 'Pz', 'T6', 'O2'],
    'T6': ['T4', 'P4', 'O2'],
    'O1': ['T5', 'P3', 'Pz'],
    'O2': ['Pz', 'P4', 'T6'],
}

ADSZ_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    ADSZ_CHANNEL_LIST, ADSZ_ADJACENCY_LIST)

ADSZ_STANDARD_1020_RENAME = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8',
}

# MSU's documented consecutive channel-block order for the separate SZ cohort.
ADSZ_SCHIZOPHRENIA_CHANNEL_LIST = [
    'F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz',
    'P4', 'T6', 'O1', 'O2',
]
ADSZ_SCHIZOPHRENIA_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    ADSZ_SCHIZOPHRENIA_CHANNEL_LIST, ADSZ_LOCATION_LIST)
ADSZ_SCHIZOPHRENIA_ADJACENCY_LIST = {
    channel: [neighbor for neighbor in ADSZ_ADJACENCY_LIST[channel]
              if neighbor in ADSZ_SCHIZOPHRENIA_CHANNEL_LIST]
    for channel in ADSZ_SCHIZOPHRENIA_CHANNEL_LIST
}
ADSZ_SCHIZOPHRENIA_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    ADSZ_SCHIZOPHRENIA_CHANNEL_LIST, ADSZ_SCHIZOPHRENIA_ADJACENCY_LIST)
