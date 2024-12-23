import torch
import torch.nn as nn


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    return torch.log(torch.clamp(x, min=eps))


class Expression(nn.Module):

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
                self.expression_fn, "kwargs"):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__,
                str(self.expression_fn.kwargs))
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (self.__class__.__name__ + "(expression=%s) " % expression_str)


class Conv2dNormWeight(nn.Conv2d):

    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dNormWeight, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data,
                                        p=2,
                                        dim=0,
                                        maxnorm=self.max_norm)
        return super(Conv2dNormWeight, self).forward(x)


class CSPNet(nn.Module):
    r'''
    CSP-empowered neural network (CSP-Net). For more details, please refer to the following information.

    - Paper: Jiang X, Meng L, Chen X, et al. CSP-Net: Common spatial pattern empowered neural networks for EEG-based motor imagery classification[J]. Knowledge-Based Systems, 2024, 305: 112668.
    - URL: https://www.sciencedirect.com/science/article/pii/S0950705124013029

    Below is a quick start example:

    .. code-block:: python

        from torcheeg.models import CSPNet

        model = CSPNet(chunk_size=1750,
                      num_electrodes=22,
                      num_classes=5,
                      num_filters_t=20,
                      filter_size_t=25)

        # batch_size, num_electrodes, n_electrodes, chunk_size
        x = torch.randn(10, 1, 22, 1750)
        model(x)

    Args:
        chunk_size (int): Number of data points included in each EEG chunk. (default: :obj:`1750`)
        num_electrodes (int): The number of electrodes, i.e., number of channels. (default: :obj:`22`)
        num_classes (int): The number of classes to predict. (default: :obj:`5`)
        dropout (float): Dropout rate. (default: :obj:`0.5`)
        num_filters_t (int): The number of temporal filters. (default: :obj:`20`)
        filter_size_t (int): The size of temporal filters. Must be smaller than chunk_size. (default: :obj:`25`)
        num_filters_s (int): The number of spatial filters per temporal filter. (default: :obj:`2`)
        filter_size_s (int): The size of spatial filters. If less than or equal to 0, it will be set to num_electrodes. (default: :obj:`-1`)
        pool_size_1 (int): The size of the average pooling layer. (default: :obj:`100`)
        pool_stride_1 (int): The stride of the average pooling layer. (default: :obj:`25`)
    '''

    def __init__(
        self,
        chunk_size: int = 1750,
        num_electrodes: int = 22,
        num_classes: int = 5,
        dropout: float = 0.5,
        num_filters_t: float = 20,
        filter_size_t: float = 25,
        num_filters_s: float = 2,
        filter_size_s: float = -1,
        pool_size_1: float = 100,
        pool_stride_1: float = 25,
    ):
        super().__init__()
        assert filter_size_t <= chunk_size, "Temporal filter size error"
        if filter_size_s <= 0:
            filter_size_s = num_electrodes

        self.features = nn.Sequential(
            nn.Conv2d(1,
                      num_filters_t, (filter_size_t, 1),
                      padding=(filter_size_t // 2, 0),
                      bias=False),
            nn.BatchNorm2d(num_filters_t),
            nn.Conv2d(num_filters_t,
                      num_filters_t * num_filters_s, (1, filter_size_s),
                      groups=num_filters_t,
                      bias=False),
            nn.BatchNorm2d(num_filters_t * num_filters_s),
            Expression(square),
            nn.AvgPool2d((pool_size_1, 1), stride=(pool_stride_1, 1)),
            Expression(safe_log),
            nn.Dropout(dropout),
        )

        n_features = (chunk_size - pool_size_1) // pool_stride_1 + 1
        n_filters_out = num_filters_t * num_filters_s
        self.feature_dim = n_filters_out

        self.classifier = nn.Sequential(
            Conv2dNormWeight(n_filters_out,
                             num_classes, (n_features, 1),
                             max_norm=0.5), nn.LogSoftmax(dim=1))

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.features(x)
        x = self.classifier(x)
        x = x[:, :, 0, 0]
        return x
