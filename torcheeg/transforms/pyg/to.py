from typing import Dict, List, Union

import numpy as np
import torch
from torch_geometric.data import Data

from ..base_transform import EEGTransform


class ToG(EEGTransform):
    r'''
    A transformation method for constructing a graph representation of EEG signals, the results of which are applied to the input of the :obj:`torch_geometric` model. In the graph, nodes correspond to electrodes, and edges correspond to associations between electrodes (eg, spatially adjacent or functionally connected)

    :obj:`TorchEEG` provides some common graph structures. Consider using the following adjacency matrices depending on the dataset (with different EEG acquisition systems):

    - datasets.constants.emotion_recognition.deap.DEAP_ADJACENCY_MATRIX
    - datasets.constants.emotion_recognition.dreamer.DREAMER_ADJACENCY_MATRIX
    - datasets.constants.emotion_recognition.seed.SEED_ADJACENCY_MATRIX
    - ...

    .. code-block:: python

        transform = ToG(adj=DEAP_ADJACENCY_MATRIX)
        transform(eeg=np.random.randn(32, 128))['eeg'].shape
        >>> (32, 4)

    Args:
        adj (list): An adjacency matrix represented by a 2D array, each element in the adjacency matrix represents the electrode-to-electrode edge weight. Please keep the order of electrodes in the rows and columns of the adjacency matrix consistent with the EEG signal to be transformed.
        complete_graph (bool): Whether to build as a complete graph. If False, only construct edges between electrodes based on non-zero elements; if True, construct variables between all electrodes and set the weight of non-existing edges to 0. (defualt: :obj:`False`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (defualt: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self, adj: List[List], complete_graph: bool = False, apply_to_baseline: bool = False):
        super(ToG, self).__init__(apply_to_baseline=apply_to_baseline)
        adj = torch.tensor(adj)
        if complete_graph:
            adj[adj == 0] = 1e-6
        self.adj = adj.to_sparse()
        self.complete_graph = complete_graph

    def __call__(self, *args, eeg: np.ndarray, baseline: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, Data]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (torch.Tensor, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            torch_geometric.data.Data: The graph representation data types that torch_geometric can accept. Nodes correspond to electrodes, and edges are determined via the given adjacency matrix.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> Data:
        data = Data(edge_index=self.adj._indices())
        data.x = torch.from_numpy(eeg).float()
        data.edge_weight = self.adj._values()

        return data
