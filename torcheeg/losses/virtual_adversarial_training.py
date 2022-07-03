import contextlib
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model: nn.Module):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d: torch.Tensor, eps=1e-8):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + eps
    return d


class VirtualAdversarialTrainingLoss(nn.Module):
    r"""
    Virtual adversarial training loss is a regularization method based on virtual adversarial loss defined as the robustness of the conditional label distribution around each input data point against local perturbation. The virtual adversarial training loss smooth the model are only virtually adversarial and does not require label information and is hence applicable to semi-supervised learning.

    - Paper: Miyato T, Maeda S, Koyama M, et al. Virtual adversarial training: a regularization method for supervised and semi-supervised learning[J]. IEEE transactions on pattern analysis and machine intelligence, 2018, 41(8): 1979-1993.
    - URL: https://ieeexplore.ieee.org/abstract/document/8417973
    - Related Project: https://github.com/lyakaap/VAT-pytorch

    .. code-block:: python

        inputs = torch.randn(3, 5, requires_grad=True)
        model = nn.Linear(5, 5)
        loss = VirtualAdversarialTrainingLoss()
        output = loss(model, inputs)

    Args:
        xi (float): The hyperparameter xi in the focal loss. (defualt: :obj:`10.0`)
        eps (float): The hyperparameter eps in the focal loss. (defualt: :obj:`1.0`)
        iterations (int): iteration times of computing adversarial noise. (defualt: :obj:`1`)
    """
    def __init__(self, xi: float = 10.0, eps: float = 1.0, iterations: int = 1):
        super(VirtualAdversarialTrainingLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.iterations = iterations

    def forward(self,
                model: nn.Module,
                x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            model (nn.Module): For the model used for classification, the input of the forward function of the model should be a torch.Tensor, and the output should be a torch.Tensor corresponding to the predictions.
            x (torch.Tensor): The input data for the model.

        Returns:
            torch.Tensor: The loss value.
        '''
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):

            for _ in range(self.iterations):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

    @property
    def repr_body(self) -> Dict:
        return {'xi': self.xi, 'eps': self.eps, 'iterations': self.iterations}

    def __repr__(self) -> str:
        # init info
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            # line end
            if i:
                format_string += ', '
            # str param
            if isinstance(v, str):
                format_string += f"{k}='{v}'"
            else:
                format_string += f"{k}={v}"
        format_string += ')'
        return format_string
