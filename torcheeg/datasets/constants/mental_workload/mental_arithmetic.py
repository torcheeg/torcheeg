from ..utils import format_adj_matrix_from_adj_list, format_channel_location_dict

MENTAL_ARITHMETIC_CHANNEL_LIST = ['Fp1',
 'Fp2',
 'F3',
 'F4',
 'F7',
 'F8',
 'T3',
 'T4',
 'C3',
 'C4',
 'T5',
 'T6',
 'P3',
 'P4',
 'O1',
 'O2',
 'Fz',
 'Cz',
 'Pz']

MENTAL_ARITHMETIC_LOCATION_LIST = [['-', 'Fp1', '-', 'Fp2', '-'],
 ['F7', 'F3', 'Fz', 'F4', 'F8'],
 ['T3', 'C3', 'Cz', 'C4', 'T4'],
 ['T5', 'P3', 'Pz', 'P4', 'T6'],
 ['-', 'O1', '-', 'O2', '-']]

MENTAL_ARITHMETIC_ADJACENCY_LIST = {'Fp1': ['F7', 'F3', 'Fz', 'Fp2'],
 'Fp2': ['Fp1', 'Fz', 'F4', 'F8'],
 'F3': ['Fp1', 'F7', 'Fz', 'C3'],
 'F4': ['Fp2', 'Fz', 'F8', 'C4'],
 'F7': ['Fp1', 'F3', 'T3'],
 'F8': ['Fp2', 'F4', 'T4'],
 'T3': ['F7', 'C3', 'T5'],
 'T4': ['F8', 'C4', 'T6'],
 'C3': ['F3', 'T3', 'Cz', 'P3'],
 'C4': ['F4', 'Cz', 'T4', 'P4'],
 'T5': ['T3', 'P3', 'O1'],
 'T6': ['T4', 'P4', 'O2'],
 'P3': ['C3', 'T5', 'Pz', 'O1'],
 'P4': ['C4', 'Pz', 'T6', 'O2'],
 'O1': ['P3', 'T5', 'Pz'],
 'O2': ['P4', 'Pz', 'T6'],
 'Fz': ['Fp1', 'Fp2', 'F3', 'F4', 'Cz'],
 'Cz': ['Fz', 'C3', 'C4', 'Pz'],
 'Pz': ['Cz', 'P3', 'P4', 'O1', 'O2']}

MENTAL_ARITHMETIC_CHANNEL_LOCATION_DICT = format_channel_location_dict(
    MENTAL_ARITHMETIC_CHANNEL_LIST, MENTAL_ARITHMETIC_LOCATION_LIST)

MENTAL_ARITHMETIC_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(
    MENTAL_ARITHMETIC_CHANNEL_LIST, MENTAL_ARITHMETIC_ADJACENCY_LIST)
