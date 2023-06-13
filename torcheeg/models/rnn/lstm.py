import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    r'''
    A simple but effective long-short term memory (LSTM) network structure from the book of Zhang et al. For more details, please refer to the following information.

    - Book: Zhang X, Yao L. Deep Learning for EEG-Based Brain-Computer Interfaces: Representations, Algorithms and Applications[M]. 2021.
    - URL: https://www.worldscientific.com/worldscibooks/10.1142/q0282#t=aboutBook
    - Related Project: https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/pythonscripts/4-1-1_LSTM.py

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = GRU(num_electrodes=32, hid_channels=64, num_classes=2)

    Args:
        num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (default: :obj:`32`)
        hid_channels (int): The number of hidden nodes in the GRU layers and the fully connected layer. (default: :obj:`64`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''
    def __init__(self,
                 num_electrodes: int = 32,
                 hid_channels: int = 64,
                 num_classes: int = 2):
        super(LSTM, self).__init__()

        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        self.gru_layer = nn.LSTM(input_size=num_electrodes,
                                 hidden_size=hid_channels,
                                 num_layers=2,
                                 bias=True,
                                 batch_first=True)

        self.out = nn.Linear(hid_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 32, 128]`. Here, :obj:`n` corresponds to the batch size, :obj:`32` corresponds to :obj:`num_electrodes`, and :obj:`128` corresponds to the number of data points included in the input EEG chunk.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = x.permute(0, 2, 1)

        r_out, (_, _) = self.gru_layer(x, None)
        r_out = F.dropout(r_out, 0.3)
        x = self.out(r_out[:, -1, :])  # choose r_out at the last time step
        return x