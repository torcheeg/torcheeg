from ..utils import format_channel_location_dict
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
