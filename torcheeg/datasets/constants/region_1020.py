# https://arxiv.org/abs/2105.02786
GENERAL_REGION_LIST = [['FP1', 'FPZ', 'FP2'],
                       ['AF7', 'AF3', 'AFZ', 'AF4', 'AF8'],
                       ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
                       ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6'],
                       ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'],
                       ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
                       ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
                       ['PO7', 'PO3', 'POZ', 'PO4', 'PO8'], ['O1', 'OZ', 'O2'],
                       ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8']]

FRONTAL_REGION_LIST = [['F7', 'F5', 'F3', 'F1'], ['F2', 'F4', 'F6', 'F8'],
                       ['FC5', 'FC3', 'FC1'], ['FC2', 'FC4', 'FC6'],
                       ['FP1', 'AF7', 'AF3'], ['FP2', 'AF4', 'AF8'],
                       ['FPZ', 'AFZ', 'FZ', 'FCZ'],
                       ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'],
                       ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
                       ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
                       ['PO7', 'PO3', 'POZ', 'PO4', 'PO8'], ['O1', 'OZ', 'O2'],
                       ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8']]

HEMISPHERE_REGION_LIST = [
    ['F7', 'F5', 'F3', 'F1'], ['F2', 'F4', 'F6', 'F8'], ['FC5', 'FC3', 'FC1'],
    ['FC2', 'FC4', 'FC6'], ['FP1', 'AF7', 'AF3'], ['FP2', 'AF4', 'AF8'],
    ['FPZ', 'AFZ', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ'],
    ['C5', 'C3', 'C1'], ['C2', 'C4', 'C6'], ['CP5', 'CP3', 'CP1'],
    ['CP2', 'CP4', 'CP6'], ['P7', 'P5', 'P3', 'P1'], ['P2', 'P4', 'P6', 'P8'],
    ['PO7', 'PO3', 'O1'], ['PO4', 'PO8', 'O2'], ['FT7', 'T7', 'TP7'],
    ['FT8', 'T8', 'TP8']
]