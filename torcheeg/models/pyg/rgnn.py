import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add

from typing import Union, Tuple


def maybe_num_electrodes(index: torch.Tensor, num_electrodes: Union[int, None] = None) -> int:
    return index.max().item() + 1 if num_electrodes is None else num_electrodes


def add_remaining_self_loops(edge_index: torch.Tensor,
                             edge_weight: Union[torch.Tensor, None] = None,
                             fill_value: float = 1.0,
                             num_electrodes: Union[int, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    num_electrodes = maybe_num_electrodes(edge_index, num_electrodes)
    row, col = edge_index

    mask = row != col

    inv_mask = ~mask
    # print("inv_mask", inv_mask)

    loop_weight = torch.full((num_electrodes, ),
                             fill_value,
                             dtype=None if edge_weight is None else edge_weight.dtype,
                             device=edge_index.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]

        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight

        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)
    loop_index = torch.arange(0, num_electrodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 1,
                 cached: bool = False,
                 bias: bool = True):
        super(NewSGConv, self).__init__(in_channels, out_channels, K=num_layers, cached=cached, bias=bias)

    # allow negative edge weights
    @staticmethod
    def norm(edge_index: torch.Tensor,
             num_electrodes: int,
             adj: torch.Tensor,
             improved: bool = False,
             dtype: Union[torch.dtype, None] = None):
        if adj is None:
            adj = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, adj = add_remaining_self_loops(edge_index, adj, fill_value, num_electrodes)
        row, col = edge_index
        deg = scatter_add(torch.abs(adj), row, dim=0, dim_size=num_electrodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * adj * deg_inv_sqrt[col]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, adj: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(edge_index, x.size(0), adj, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # x_j: (batch_size*num_electrodes*num_electrodes, in_channels)
        # norm: (batch_size*num_electrodes*num_electrodes, )
        return norm.view(-1, 1) * x_j


class RGNN(torch.nn.Module):
    r'''
    Regularized Graph Neural Networks (RGNN). For more details, please refer to the following information.

    - Paper: Zhong P, Wang D, Miao C. EEG-based emotion recognition using regularized graph neural networks[J]. IEEE Transactions on Affective Computing, 2020.
    - URL: https://ieeexplore.ieee.org/abstract/document/9091308
    - Related Project: https://github.com/zhongpeixiang/RGNN

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = SEEDDataset(io_path=f'./seed',
                              root_path='./Preprocessed_EEG',
                              offline_transform=transforms.BandDifferentialEntropy(),
                              online_transform=ToG(SEED_STANDARD_ADJACENCY_MATRIX),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: int(x) + 1),
                              ]),
                              num_worker=8)
        model = RGNN(adj=torch.Tensor(SEED_STANDARD_ADJACENCY_MATRIX),
                     in_channels=5,
                     num_electrodes=62,
                     hid_channels=32,
                     num_layers=2,
                     num_classes=3,
                     dropout=0.7,
                     learn_edge_weights=True)

    Args:
        adj (torch.Tensor): The adjacency matrix corresponding to the EEG representation, where 1.0 means the node is adjacent and 0.0 means the node is not adjacent. The matrix shape should be [num_electrodes, num_electrodes].
        num_electrodes (int): The number of electrodes. (default: :obj:`62`)
        in_channels (int): The feature dimension of each electrode. (default: :obj:`5`)
        num_layers (int): The number of graph convolutional layers. (default: :obj:`2`)
        hid_channels (int): The number of hidden nodes in the first fully connected layer. (default: :obj:`32`)
        num_classes (int): The number of classes to predict. (default: :obj:`3`)
        dropout (float): Probability of an element to be zeroed in the dropout layers at the output fully-connected layer. (default: :obj:`0.7`)
        learn_edge_weights (bool): Whether to learn a set of parameters to adjust the adjacency matrix. (default: :obj:`True`)
    '''
    def __init__(self,
                 adj: Union[torch.Tensor, list],
                 num_electrodes: int = 62,
                 in_channels: int = 5,
                 num_layers: int = 2,
                 hid_channels: int = 32,
                 num_classes: int = 3,
                 dropout: float = 0.7,
                 learn_edge_weights: bool = True):

        super(RGNN, self).__init__()
        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.learn_edge_weights = learn_edge_weights

        self.xs, self.ys = torch.tril_indices(self.num_electrodes, self.num_electrodes, offset=0)
        if isinstance(adj, list):
            adj = torch.tensor(adj)
        adj = adj.reshape(self.num_electrodes, self.num_electrodes)[self.xs, self.ys]  # strict lower triangular values
        self.adj = nn.Parameter(adj, requires_grad=learn_edge_weights)

        self.conv1 = NewSGConv(in_channels=in_channels, out_channels=hid_channels, num_layers=num_layers)
        self.fc = nn.Linear(hid_channels, num_classes)

    def forward(self, data: Batch) -> torch.Tensor:
        r'''
        Args:
            data (torch_geometric.data.Batch): EEG signal representation, the ideal input shape of data.x is :obj:`[n, 62, 4]`. Here, :obj:`n` corresponds to the batch size, :obj:`62` corresponds to the number of electrodes, and :obj:`4` corresponds to :obj:`in_channels`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        batch_size = data.num_graphs
        x, edge_index = data.x, data.edge_index
        adj = torch.zeros((self.num_electrodes, self.num_electrodes), device=edge_index.device)
        adj[self.xs.to(adj.device), self.ys.to(adj.device)] = self.adj
        adj = adj + adj.transpose(1, 0) - torch.diag(adj.diagonal())  # copy values from lower tri to upper tri
        adj = adj.reshape(-1).repeat(batch_size)
        x = F.relu(self.conv1(x, edge_index, adj))

        x = global_add_pool(x, data.batch, size=batch_size)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)

        return x