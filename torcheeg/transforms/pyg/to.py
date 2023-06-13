from typing import Callable, Dict, List, Union

import numpy as np
import torch
from scipy.signal import hilbert
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
        transform(eeg=np.random.randn(32, 128))['eeg']
        >>> torch_geometric.data.Data

    Args:
        adj (list): An adjacency matrix represented by a 2D array, each element in the adjacency matrix represents the electrode-to-electrode edge weight. Please keep the order of electrodes in the rows and columns of the adjacency matrix consistent with the EEG signal to be transformed.
        add_self_loop (bool): Whether to add self-loop edges to the graph. (default: :obj:`True`)
        threshold (float, optional): Used to cut edges when not None. Edges whose weights exceed a threshold are retained. (default: :obj:`None`)
        top_k (int, optional): Used to cut edges when not None. Keep the k edges connected to each node with the largest weights. (default: :obj:`None`)
        binary (bool): Whether to binarize the weights on the edges to 0 and 1. If set to True, binarization are done after topk and threshold, the edge weights that still have values are set to 1, otherwise they are set to 0. (default: :obj:`False`)
        complete_graph (bool): Whether to build as a complete graph. If False, only construct edges between electrodes based on non-zero elements; if True, construct variables between all electrodes and set the weight of non-existing edges to 0. (default: :obj:`False`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 adj: List[List],
                 add_self_loop: bool = True,
                 threshold: Union[float, None] = None,
                 top_k: Union[int, None] = None,
                 binary: bool = False,
                 complete_graph: bool = False,
                 apply_to_baseline: bool = False):
        super(ToG, self).__init__(apply_to_baseline=apply_to_baseline)

        self.add_self_loop = add_self_loop
        self.threshold = threshold
        self.top_k = top_k
        self.binary = binary
        self.complete_graph = complete_graph

        adj = torch.tensor(adj)

        if add_self_loop:
            adj = adj + torch.eye(adj.shape[0])

        if not self.threshold is None:
            adj[adj < self.threshold] = 0

        if not self.top_k is None:
            rows = []
            for row in adj:
                vals, index = row.topk(self.top_k)
                topk = torch.zeros_like(row)
                topk[index] = vals
                rows.append(topk)
            adj = torch.stack(rows)

        if self.binary:
            adj[adj != 0] = 1.0

        if self.complete_graph:
            adj[adj == 0] = 1e-6

        self.adj = adj.to_sparse()

    def __call__(self,
                 *args,
                 eeg: Union[np.ndarray, torch.Tensor],
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, Data]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (torch.Tensor, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            torch_geometric.data.Data: The graph representation data types that torch_geometric can accept. Nodes correspond to electrodes, and edges are determined via the given adjacency matrix.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: Union[np.ndarray, torch.Tensor], **kwargs) -> Data:
        data = Data(edge_index=self.adj._indices())
        if isinstance(eeg, np.ndarray):
            data.x = torch.from_numpy(eeg).float()
        else:
            data.x = eeg
        data.edge_weight = self.adj._values()

        return data

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'adj': [...],
                'add_self_loop': self.add_self_loop,
                'threshold': self.threshold,
                'top_k': self.top_k,
                'binary': self.binary,
                'complete_graph': self.complete_graph
            })


class ToDynamicG(EEGTransform):
    r'''
    A transformation method for dynamically constructing the functional connections between electrodes according to the input EEG signals. The obtained graph structure based on functional connections can be applied to the input of the :obj:`torch_geometric` model. In the graph, nodes correspond to electrodes, and edges correspond to associations between electrodes (functionally connected)

    :obj:`TorchEEG` provides algorithms to dynamically calculate functional connections between electrodes:

    - Gaussian Distance
    - Absolute Pearson Correlation Coefficient
    - Phase Locking Value

    .. code-block:: python

        transform = ToDynamicG(edge_func='gaussian_distance', sigma=1.0, top_k=10, complete_graph=False)
        transform(eeg=np.random.randn(32, 128))['eeg']
        >>> Data(edge_index=[2, 320], x=[32, 128], edge_weight=[320])

        transform = ToDynamicG(edge_func='absolute_pearson_correlation_coefficient', threshold=0.1, binary=True)
        transform(eeg=np.random.randn(32, 128))['eeg']
        >>> Data(edge_index=[2, 310], x=[32, 128], edge_weight=[310])

        transform = ToDynamicG(edge_func='phase_locking_value')
        transform(eeg=np.random.randn(32, 128))['eeg']
        >>> Data(edge_index=[2, 992], x=[32, 128], edge_weight=[992])

        transform = ToDynamicG(edge_func=lambda x, y: (x * y).mean())
        transform(eeg=np.random.randn(32, 128))['eeg']
        >>> Data(edge_index=[2, 1024], x=[32, 128], edge_weight=[1024])

    Args:
        edge_func (str or Callable): Algorithms for computing functional connections. You can use the algorithms provided by TorchEEG, including gaussian_distance, absolute_pearson_correlation_coefficient and phase_locking_value. Or you can use custom functions by passing a callable object containing two parameters representing the signal of the two electrodes, and other named parameters (passed in when initializing the transform), and outputs the value of the functional connection between the two electrodes. (default: :obj:`gaussian_distance`)
        add_self_loop (bool): Whether to add self-loop edges to the graph. (default: :obj:`True`)
        threshold (float, optional): Used to cut edges when not None. Edges whose weights exceed a threshold are retained. (default: :obj:`None`)
        top_k (int, optional): Used to cut edges when not None. Keep the k edges connected to each node with the largest weights. (default: :obj:`None`)
        binary (bool): Whether to binarize the weights on the edges to 0 and 1. If set to True, binarization are done after topk and threshold, the edge weights that still have values are set to 1, otherwise they are set to 0. (default: :obj:`False`)
        complete_graph (bool): Whether to build as a complete graph. If False, only construct edges between electrodes based on non-zero elements; if True, construct variables between all electrodes and set the weight of non-existing edges to 0. (default: :obj:`False`)
        apply_to_baseline: (bool): Whether to act on the baseline signal at the same time, if the baseline is passed in when calling. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 edge_func: Union[str, Callable] = 'gaussian_distance',
                 add_self_loop: bool = True,
                 threshold: Union[float, None] = None,
                 top_k: Union[int, None] = None,
                 binary: bool = False,
                 complete_graph: bool = False,
                 apply_to_baseline: bool = False,
                 **kwargs):
        super(ToDynamicG, self).__init__(apply_to_baseline=apply_to_baseline)
        self.top_k = top_k
        self.threshold = threshold
        self.add_self_loop = add_self_loop
        self.binary = binary
        self.complete_graph = complete_graph
        self.edge_func = edge_func
        self.edge_func_list = {
            'gaussian_distance': self.gaussian_distance,
            'absolute_pearson_correlation_coefficient': self.absolute_pearson_correlation_coefficient,
            'phase_locking_value': self.phase_locking_value
        }
        self.kwargs = kwargs

    def opt(self, eeg: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(eeg, torch.Tensor):
            eeg = eeg.numpy()

        func = None

        if hasattr(self.edge_func, '__call__'):
            func = self.edge_func
        else:
            valid_edge_func = list(self.edge_func_list.keys())
            assert self.edge_func in valid_edge_func, f'Unimplemented calculation method {self.edge_func}, allowable selection of calculation methods include {valid_edge_func}'

            func = self.edge_func_list[self.edge_func]

        num_node = len(eeg)
        adj = np.zeros((num_node, num_node))

        for i in range(num_node):
            for j in range(num_node):
                adj[i][j] = func(eeg[i], eeg[j], **self.kwargs)

        return adj

    def adj(self, eeg: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        adj = torch.tensor(self.opt(eeg))

        if self.add_self_loop:
            adj = adj + torch.eye(adj.shape[0])

        if not self.threshold is None:
            adj[adj < self.threshold] = 0

        if not self.top_k is None:
            rows = []
            for row in adj:
                vals, index = row.topk(self.top_k)
                topk = torch.zeros_like(row)
                topk[index] = vals
                rows.append(topk)
            adj = torch.stack(rows)

        if self.binary:
            adj[adj != 0] = 1.0

        if self.complete_graph:
            adj[adj == 0] = 1e-6

        return adj.to_sparse()

    def gaussian_distance(self, x: np.ndarray, y: np.ndarray, sigma: float = 1.0, **kwargs):
        return np.exp(-np.linalg.norm(x - y, 2)**2 / (2. * sigma**2))

    def absolute_pearson_correlation_coefficient(self, x: np.ndarray, y: np.ndarray, **kwargs):
        return np.abs(np.corrcoef(x, y)[0][1])

    def phase_locking_value(self, x: np.ndarray, y: np.ndarray, **kwargs):
        x_hill = hilbert(x)
        y_hill = hilbert(y)
        pdt = (np.inner(x_hill, np.conj(y_hill)) /
               (np.sqrt(np.inner(x_hill, np.conj(x_hill)) * np.inner(y_hill, np.conj(y_hill)))))
        return np.angle(pdt)

    def __call__(self,
                 *args,
                 eeg: Union[np.ndarray, torch.Tensor],
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, Data]:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
            baseline (torch.Tensor, optional) : The corresponding baseline signal, if apply_to_baseline is set to True and baseline is passed, the baseline signal will be transformed with the same way as the experimental signal.

        Returns:
            torch_geometric.data.Data: The graph representation data types that torch_geometric can accept. Nodes correspond to electrodes, and edges are determined via the given adjacency matrix.
        '''
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: Union[np.ndarray, torch.Tensor], **kwargs) -> Data:
        adj = self.adj(eeg)
        data = Data(edge_index=adj._indices())
        if isinstance(eeg, np.ndarray):
            data.x = torch.from_numpy(eeg).float()
        else:
            data.x = eeg
        data.edge_weight = adj._values()

        return data

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'top_k': self.top_k,
                'threshold': self.threshold,
                'add_self_loop': self.add_self_loop,
                'binary': self.binary,
                'complete_graph': self.complete_graph,
                'edge_func': self.edge_func
            })
