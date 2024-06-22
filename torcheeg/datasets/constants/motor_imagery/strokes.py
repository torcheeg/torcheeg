from ..utils import format_channel_location_dict
STROKEPATIENTSMI_CHANNEL_LIST  = ['FP1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4', 'FT7', 
                                  'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CPz', 'CP3', 'CP4', 'TP7', 'TP8', 
                                  'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2']
STROKEPATIENTSMI_LOCATION_LIST = \
                         [['-',   'FP1',  '-',    'FP2',   '-'],
                          ['F7',  'F3',   'Fz',   'F4',   'F8'],
                          ['FT7',  'FC3', 'FCz',  'FC4', 'FT8'],
                          ['T3',   'C3',  'Cz',   'C4',   'T4'],
                          ['TP7',  'CP3',  'CPz',   'CP4', 'TP8'],
                          ['T5',   'P3',   'Pz',  'P4',   'T6'],
                          ['-',    'O1',  'Oz',   'O2',    '-']]

STROKEPATIENTSMI_LOCATION_DICT = format_channel_location_dict(STROKEPATIENTSMI_CHANNEL_LIST,STROKEPATIENTSMI_LOCATION_LIST)