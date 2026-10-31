import functools
import inspect
import marshal
import os
import types
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from ..utils import compute_checksum
from ..utils import compute_fingerprint as _compute_fingerprint
from ..utils import ensure_cache, validate_cache


def _describe_value(value, seen=None):
    if value is None or isinstance(value, (str, bytes, int, float, bool, np.generic)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return str(tensor.dtype), tuple(tensor.shape), joblib.hash(
            tensor.reshape(-1).view(torch.uint8).numpy())
    if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
        return joblib.hash(value)
    if isinstance(value, types.ModuleType):
        return value.__name__
    if isinstance(value, type):
        return value.__module__, value.__qualname__
    seen = set() if seen is None else seen
    if id(value) in seen:
        return type(value).__module__, type(value).__qualname__
    seen = seen | {id(value)}
    if isinstance(value, dict):
        return {key: _describe_value(item, seen) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value).__name__, [_describe_value(item, seen) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_describe_value(item, seen) for item in value), key=repr)
    if isinstance(value, functools.partial):
        return (_describe_value(value.func, seen), _describe_value(value.args, seen),
                _describe_value(value.keywords, seen))
    if inspect.ismethod(value):
        return (_describe_value(value.__func__,
                                seen), _describe_value(value.__self__, seen))
    if inspect.isfunction(value):
        closure = [cell.cell_contents for cell in (value.__closure__ or ())]
        constants = {
            name: value.__globals__[name]
            for name in value.__code__.co_names
            if name in value.__globals__ and isinstance(value.__globals__[name], (
                str, bytes, int, float, bool, tuple, list, dict, np.ndarray))
        }
        return (value.__module__, value.__qualname__, marshal.dumps(value.__code__),
                _describe_value(value.__defaults__,
                                seen), _describe_value(value.__kwdefaults__, seen),
                _describe_value(closure, seen), _describe_value(constants, seen))
    if hasattr(value, '__dict__'):
        cls = type(value)
        try:
            implementation = inspect.getsource(cls)
        except (OSError, TypeError):
            implementation = cls.__module__, cls.__qualname__
        return implementation, _describe_value(vars(value), seen)
    return joblib.hash(value)


def _collect_sources(value, excluded):
    if isinstance(value, dict):
        for item in value.values():
            yield from _collect_sources(item, excluded)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _collect_sources(item, excluded)
    elif isinstance(value, (pd.DataFrame, pd.Series)):
        yield from _collect_sources(value.to_dict(), excluded)
    elif hasattr(value, 'filenames'):
        yield from _collect_sources(value.filenames, excluded)
    elif isinstance(value, (str, os.PathLike)):
        path = Path(value)
        try:
            exists = path.exists()
        except (OSError, ValueError):
            return
        if not exists:
            return
        paths = path.rglob('*') if path.is_dir() else [path]
        for file in paths:
            absolute = os.path.abspath(file)
            if (absolute == excluded or absolute.startswith(excluded + os.sep)
                    or absolute == excluded + '.lock'
                    or any(part in ('.git', '__pycache__', '.torcheeg') for part in (
                        file.relative_to(path).parts if path.is_dir() else ()))
                    or any(part.startswith('.torcheeg-cache-') for part in file.parts)):
                continue
            if file.is_file():
                yield absolute


def compute_fingerprint(dataset_class, configuration, records, io_path):
    """Return a fingerprint for source files, processing configuration and record order."""
    excluded = os.path.abspath(io_path)
    sources = sorted(set(_collect_sources((configuration, records), excluded)))
    inputs = [(path, compute_checksum(path)) for path in sources]
    implementations = [
        getattr(dataset_class, name)
        for name in ('set_records', 'read_record', 'process_record')
    ]
    metadata = {}
    for name in ('channels', 'grid', 'adjacency'):
        try:
            metadata[name] = getattr(dataset_class, name)
        except NotImplementedError:
            continue
    value = (dataset_class.__module__, dataset_class.__qualname__,
             _describe_value(metadata),
             _describe_value(configuration), _describe_value(records), inputs,
             _describe_value(implementations))
    return _compute_fingerprint(joblib.hash(value).encode())


def get_regions(channel_list, region_list):
    output = []
    for region in region_list:
        region_channel_index_list = []
        for region_channel in region:
            try:
                channel_index = channel_list.index(region_channel)
            except ValueError:
                continue
            region_channel_index_list.append(channel_index)
        if len(region_channel_index_list) > 0:
            output.append(region_channel_index_list)
    return output


def get_locations(channel_list, location_list):
    location_list = np.array(location_list)
    output = {}
    for channel in channel_list:
        if len(np.argwhere(location_list == channel)):
            location = (np.argwhere(location_list == channel)[0]).tolist()
            output[channel] = location
    return output


def get_adjacency(channel_list: List,
                  adj_list: Dict[str, List[str]]) -> List[List]:
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


def get_grid_adjacency(location_list):
    adjacency_list = {}
    rows = len(location_list)
    columns = len(location_list[0]) if rows else 0
    for row, values in enumerate(location_list):
        for column, channel in enumerate(values):
            if channel == '-':
                continue
            neighbors = []
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == column_delta == 0:
                        continue
                    neighbor_row = row + row_delta
                    neighbor_column = column + column_delta
                    if 0 <= neighbor_row < rows and 0 <= neighbor_column < columns:
                        neighbor = location_list[neighbor_row][neighbor_column]
                        if neighbor != '-':
                            neighbors.append(neighbor)
            adjacency_list[channel] = neighbors
    return adjacency_list


def get_standard_adjacency(
    channel_list: List,
    standard_channel_location_dict: Dict,
    delta: float = 0.00056,
    global_channel_list: Tuple[Tuple[str, str], ...] = (
        ('FP1', 'FP2'), ('AF3', 'AF4'), ('F5', 'F6'), ('FC5', 'FC6'),
        ('C5', 'C6'), ('CP5', 'CP6'), ('P5', 'P6'), ('PO5', 'PO6'), ('O1', 'O2'))
) -> List[List]:
    r'''
    Creates an adjacency matrix based on the relative positions of electrodes in a standard system, allowing the addition of global electrode links to connect non-adjacent but symmetrical electrodes.

    - Paper: Zhong P, Wang D, Miao C. EEG-based emotion recognition using regularized graph neural networks[J]. IEEE Transactions on Affective Computing, 2020.
    - URL: https://ieeexplore.ieee.org/abstract/document/9091308
    - Related Project: https://github.com/zhongpeixiang/RGNN

    Args:
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, use the corresponding Dataset class's ``channels`` attribute.
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


def get_positions(channels, aliases=None):
    from .montage import STANDARD_1005_CHANNEL_LOCATION_DICT

    aliases = {} if aliases is None else aliases
    return {channel: STANDARD_1005_CHANNEL_LOCATION_DICT[
        aliases.get(channel, channel).upper()][:]
        for channel in channels}
