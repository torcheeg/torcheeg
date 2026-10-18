from ..utils import (format_adj_matrix_from_adj_list,
                     format_channel_location_dict)

# The 19 scalp electrodes (international 10-20 system) used by the ADFSU
# dataset. The dataset keeps the classic temporal/parietal nomenclature
# (T3, T4, T5, T6) instead of the modern T7, T8, P7, P8 names.
ADFSU_CHANNEL_LIST = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2',
]

# Mapping from the classic ADFSU channel names to the standard ``standard_1020``
# montage names, used when attaching electrode positions in MNE.
ADFSU_CHANNEL_NAME_TO_1020 = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8',
}

ADFSU_LOCATION_LIST = [
    ['-', 'Fp1', '-', 'Fp2', '-'],
    ['F7', 'F3', 'Fz', 'F4', 'F8'],
    ['T3', 'C3', 'Cz', 'C4', 'T4'],
    ['T5', 'P3', 'Pz', 'P4', 'T6'],
    ['-', 'O1', '-', 'O2', '-'],
]

ADFSU_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    ADFSU_CHANNEL_LIST, ADFSU_LOCATION_LIST)

ADFSU_ADJACENCY_LIST = {
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

ADFSU_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    ADFSU_CHANNEL_LIST, ADFSU_ADJACENCY_LIST)
