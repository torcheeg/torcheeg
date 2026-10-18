import numpy as np

from typing import List, Tuple, Dict


def format_region_channel_list(channel_list, region_list):
    output = []
    for region in region_list:
        region_channel_index_list = []
        for region_channel in region:
            try:
                channel_index = channel_list.index(region_channel)
            except:
                continue
            region_channel_index_list.append(channel_index)
        if len(region_channel_index_list) > 0:
            output.append(region_channel_index_list)
    return output


def format_channel_location_dict(channel_list, location_list):
    location_list = np.array(location_list)
    output = {}
    for channel in channel_list:
        if len(np.argwhere(location_list == channel)):
            location = (np.argwhere(location_list == channel)[0]).tolist()
            output[channel] = location
        else:
            output[channel] = None
    return output


def format_adj_matrix_from_adj_list(channel_list: List,
                                    adj_list: List) -> List[List]:
    node_map = {k: i for i, k in enumerate(channel_list)}
    adj_matrix = np.zeros((len(channel_list), len(channel_list)))

    for start_node_name in adj_list:
        if not start_node_name in channel_list:
            continue
        start_node_index = node_map[start_node_name]
        end_node_list = adj_list[start_node_name]

        for end_node_name in end_node_list:
            if not end_node_name in node_map:
                continue
            end_node_index = node_map[end_node_name]
            adj_matrix[start_node_index][end_node_index] = 1

    return adj_matrix.tolist()


def format_adj_list_from_location_list(location_list: List[List[str]]) -> Dict[str, List[str]]:
    r'''
    Create an adjacency list from a rectangular electrode grid. Each electrode
    connects to occupied horizontal, vertical and diagonal neighboring cells.
    Empty cells contain ``'-'``. The result has no self-loops or wraparound edges.

    Args:
        location_list (list): Rows of unique, nonempty electrode names. A
            two-dimensional NumPy array is also accepted. Empty grids return
            an empty dictionary. Ragged grids and duplicate names raise
            :class:`ValueError`.

    Returns:
        dict: Electrode names mapped to neighbor-name lists, in grid order.
    '''
    try:
        locations = np.asarray(location_list, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError('location_list must be a rectangular electrode grid.') from error
    if locations.size == 0 and locations.ndim in (1, 2):
        return {}
    if locations.ndim != 2:
        raise ValueError('location_list must be a rectangular electrode grid.')
    positions = {}
    for (row, column), electrode in np.ndenumerate(locations):
        if not isinstance(electrode, str) or not electrode:
            raise ValueError('Grid cells must contain nonempty electrode names or a dash.')
        if electrode == '-':
            continue
        if electrode in positions:
            raise ValueError('Each electrode name must occupy one grid cell.')
        positions[electrode] = (row, column)
    adjacency_list = {}
    rows, columns = locations.shape
    for electrode, (row, column) in positions.items():
        neighbors = []
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                if row_offset == column_offset == 0:
                    continue
                neighbor_row, neighbor_column = row + row_offset, column + column_offset
                if 0 <= neighbor_row < rows and 0 <= neighbor_column < columns:
                    neighbor = locations[neighbor_row, neighbor_column]
                    if neighbor != '-':
                        neighbors.append(neighbor)
        adjacency_list[electrode] = neighbors
    return adjacency_list


DEFAULT_GLOBAL_CHANNEL_LIST = [('FP1', 'FP2'), ('AF3', 'AF4'), ('F5', 'F6'),
                               ('FC5', 'FC6'), ('C5', 'C6'), ('CP5', 'CP6'),
                               ('P5', 'P6'), ('PO5', 'PO6'), ('O1', 'O2')]


def format_adj_matrix_from_standard(
    channel_list: List,
    standard_channel_location_dict: Dict,
    delta: float = 0.00056,
    global_channel_list: List[Tuple[str]] = DEFAULT_GLOBAL_CHANNEL_LIST
) -> List[List]:
    r'''
    Creates an adjacency matrix based on the relative positions of electrodes in a standard system, allowing the addition of global electrode links to connect non-adjacent but symmetrical electrodes.

    - Paper: Zhong P, Wang D, Miao C. EEG-based emotion recognition using regularized graph neural networks[J]. IEEE Transactions on Affective Computing, 2020.
    - URL: https://ieeexplore.ieee.org/abstract/document/9091308
    - Related Project: https://github.com/zhongpeixiang/RGNN

    Args:
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.constants` module.
        standard_channel_location_dict (dict): Electrode location information. Represented in dictionary form, where :obj:`key` corresponds to the electrode name and :obj:`value` corresponds to (x, y, z) of the electrode on the grid. please refer to STANDARD_1005_CHANNEL_LOCATION_DICT and STANDARD_1020_CHANNEL_LOCATION_DICT.
        delta (float): The calibration constant. Due to differences in electrode coordinate scales, the values in the original paper are not recommended. 0.00056 means 20% more nodes are connected to each other. (default: :obj:`0.00056`)
        global_channel_list (float): To leverage the differential asymmetry information, the authors initialize the global inter-channel relations in the adjacency matrix. (default: :obj:`[('FP1', 'FP2'), ('AF3', 'AF4'), ('F5', 'F6'), ('FC5', 'FC6'), ('C5', 'C6'), ('CP5', 'CP6'), ('P5', 'P6'), ('PO5', 'PO6'), ('O1', 'O2')]`)
    '''
    node_map = {k: i for i, k in enumerate(channel_list)}
    adj_matrix = np.zeros((len(channel_list), len(channel_list)))

    for start_node_name in channel_list:
        if not start_node_name in standard_channel_location_dict:
            continue
        for end_node_name in channel_list:
            if not end_node_name in standard_channel_location_dict:
                continue
            start_node_pos = np.array(
                standard_channel_location_dict[start_node_name])
            end_node_pos = np.array(
                standard_channel_location_dict[end_node_name])
            edge_weight = np.linalg.norm(start_node_pos - end_node_pos)
            edge_weight = min(1.0, delta / (edge_weight**2 + 1e-6))

            adj_matrix[node_map[start_node_name]][
                node_map[end_node_name]] = edge_weight

    for start_node_name, end_node_name in global_channel_list:
        if (not start_node_name in node_map) or (not end_node_name in node_map):
            continue
        adj_matrix[node_map[start_node_name]][
            node_map[end_node_name]] = adj_matrix[node_map[start_node_name]][
                node_map[end_node_name]] - 1.0

    return adj_matrix.tolist()