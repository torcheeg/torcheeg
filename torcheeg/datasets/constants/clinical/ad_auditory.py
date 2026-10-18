from ..utils import (format_adj_matrix_from_adj_list,
                     format_channel_location_dict)

AD_AUDITORY_CHANNEL_LIST = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]

AD_AUDITORY_LOCATION_LIST = [
    ['-', 'Fp1', '-', 'Fp2', '-'],
    ['F7', 'F3', 'Fz', 'F4', 'F8'],
    ['T7', 'C3', 'Cz', 'C4', 'T8'],
    ['P7', 'P3', 'Pz', 'P4', 'P8'],
    ['-', 'O1', '-', 'O2', '-'],
]

AD_AUDITORY_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    AD_AUDITORY_CHANNEL_LIST, AD_AUDITORY_LOCATION_LIST)

AD_AUDITORY_ADJACENCY_LIST = {
    'Fp1': ['Fp2', 'F7', 'F3', 'Fz'],
    'Fp2': ['Fp1', 'Fz', 'F4', 'F8'],
    'F7': ['Fp1', 'F3', 'T7'],
    'F3': ['Fp1', 'F7', 'Fz', 'C3'],
    'Fz': ['Fp1', 'Fp2', 'F3', 'F4', 'Cz'],
    'F4': ['Fp2', 'Fz', 'F8', 'C4'],
    'F8': ['Fp2', 'F4', 'T8'],
    'T7': ['F7', 'C3', 'P7'],
    'C3': ['F3', 'T7', 'Cz', 'P3'],
    'Cz': ['Fz', 'C3', 'C4', 'Pz'],
    'C4': ['F4', 'Cz', 'T8', 'P4'],
    'T8': ['F8', 'C4', 'P8'],
    'P7': ['T7', 'P3', 'O1'],
    'P3': ['C3', 'P7', 'Pz', 'O1'],
    'Pz': ['Cz', 'P3', 'P4', 'O1', 'O2'],
    'P4': ['C4', 'Pz', 'P8', 'O2'],
    'P8': ['T8', 'P4', 'O2'],
    'O1': ['P7', 'P3', 'Pz'],
    'O2': ['Pz', 'P4', 'P8'],
}

AD_AUDITORY_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    AD_AUDITORY_CHANNEL_LIST, AD_AUDITORY_ADJACENCY_LIST)
