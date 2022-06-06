import numpy as np

from typing import List


def format_channel_location_dict(channel_list, location_list):
    location_list = np.array(location_list)
    output = {}
    for channel in channel_list:
        location = (np.argwhere(location_list == channel)[0]).tolist()
        output[channel] = location
    return output


def format_adj_matrix_from_adj_list(channel_list: List, adj_list: List) -> List[List]:
    node_map = {k: i for i, k in enumerate(channel_list)}
    adj_matrix = np.zeros((len(channel_list), len(channel_list)))
    for start_node_name in adj_list:
        if start_node_name in channel_list:
            start_node_idx = node_map[start_node_name]
            end_node_list = adj_list[start_node_name]
            for end_node_name in end_node_list:
                if end_node_name in node_map:
                    end_node_idx = node_map[end_node_name]
                    adj_matrix[start_node_idx][end_node_idx] = 1
    return adj_matrix.tolist()