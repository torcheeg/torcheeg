import torch
import torch.nn as nn
import torch.nn.functional as F


class LazyLoader:
    def __init__(self, lib_name):
        self._lib_name = lib_name
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            try:
                self._module = __import__(self._lib_name)
            except ImportError:
                raise ImportError(
                    f"To use this functionality, you need to install `{self._lib_name}`. "
                    f"Please refer to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
                )
        return getattr(self._module, name)


pyg = LazyLoader('torch_geometric')


class GIN(nn.Module):
    r'''
    A simple but effective graph isomorphism network (GIN) structure from the book of Zhang et al. For more details, please refer to the following information.

    - Book: Zhang X, Yao L. Deep Learning for EEG-Based Brain-Computer Interfaces: Representations, Algorithms and Applications[M]. 2021.
    - URL: https://www.worldscientific.com/worldscibooks/10.1142/q0282#t=aboutBook
    - Related Project: https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/pythonscripts/4-3_GIN.py

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.transforms.pyg import ToG
        from torcheeg.datasets.constants import SEED_STANDARD_ADJACENCY_MATRIX
        from torcheeg.models import GIN
        from torch_geometric.data import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.BandDifferentialEntropy(),
                              online_transform=ToG(SEED_STANDARD_ADJACENCY_MATRIX),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

        model = GIN(in_channels=4, hid_channels=64, num_classes=2)

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Args:
        in_channels (int): The feature dimension of each electrode. (default: :obj:`4`)
        hid_channels (int): The number of hidden nodes in the GRU layers and the fully connected layer. (default: :obj:`64`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''

    def __init__(self,
                 in_channels: int = 4,
                 hid_channels: int = 64,
                 num_classes: int = 3):
        super(GIN, self).__init__()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        nn1 = nn.Sequential(nn.Linear(in_channels, hid_channels), nn.ReLU(),
                            nn.Linear(hid_channels, hid_channels))
        self.conv1 = pyg.nn.GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(hid_channels)

        nn2 = nn.Sequential(nn.Linear(hid_channels, hid_channels), nn.ReLU(),
                            nn.Linear(hid_channels, hid_channels))
        self.conv2 = pyg.nn.GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(hid_channels)

        nn3 = nn.Sequential(nn.Linear(hid_channels, hid_channels), nn.ReLU(),
                            nn.Linear(hid_channels, hid_channels))
        self.conv3 = pyg.nn.GINConv(nn3)
        self.bn3 = nn.BatchNorm1d(hid_channels)

        self.fc1 = nn.Linear(hid_channels, hid_channels)
        self.fc2 = nn.Linear(hid_channels, num_classes)

    def forward(self, data: 'pyg.data.Batch') -> torch.Tensor:
        r'''
        Args:
            data (torch_geometric.data.Batch): EEG signal representation, the ideal input shape of data.x is :obj:`[n, 62, 4]`. Here, :obj:`n` corresponds to the batch size, :obj:`62` corresponds to the number of electrodes, and :obj:`4` corresponds to :obj:`in_channels`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x, edge_index, batch = data.x, data.edge_index, data.num_graphs

        x = x.reshape([-1, self.in_channels])

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        x = x.view(batch, -1, self.hid_channels)
        x = x.sum(dim=1)

        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)

        return x
