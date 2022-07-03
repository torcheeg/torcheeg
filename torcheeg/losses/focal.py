import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


class FocalLoss(nn.Module):
    r"""
    Focal loss is a loss that adds a factor :math:`(1 - p_t)^{\gamma}` to the standard cross entropy criterion. Setting :math:`\gamma>0` reduces the relative loss for well-classified examples (:math:`p_t > 0.5`), putting more focus on hard, misclassified examples. As experiments demonstrate, the focal loss enables training highly accurate models in the presence of vast numbers of easy negative examples.

    - Paper: Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2980-2988.
    - URL: https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
    - Related Project: https://github.com/ailias/Focal-Loss-implement-on-Tensorflow

    .. code-block:: python

        inputs = torch.randn(3, 5, requires_grad=True)
        targets = torch.empty(3, dtype=torch.long).random_(5)
        loss = FocalLoss(num_classes=5)
        output = loss(inputs, targets)

    This version further adds support for multi-classification and label smoothing to the original implementation.

    Args:
        alpha (float): The hyperparameter alpha in the focal loss. (defualt: :obj:`1.0`)
        gamma (float): The hyperparameter gamma in the focal loss. (defualt: :obj:`2.0`)
        reduction (str): Specifies the reduction to apply to the output. Options include none, mean and sum. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed.
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
    """
    def __init__(
        self,
        alpha: int = 1.0,
        gamma: int = 2.0,
        reduction: str = 'mean',
        label_smooth: float = 0.05,
        num_classes: int = 2,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smooth = label_smooth
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            inputs (torch.Tensor): The predictions from the model.
            targets (torch.Tensor): The ground-truth labels.

        Returns:
            torch.Tensor: the loss value.
        '''

        targets = F.one_hot(targets, num_classes=self.num_classes)

        if self.label_smooth is not None:
            targets = (1 - self.label_smooth
                       ) * targets + self.label_smooth / self.num_classes

        loss = F.binary_cross_entropy_with_logits(inputs,
                                                  targets,
                                                  reduction='none')

        p_t = torch.exp(-loss)
        loss = self.alpha * (1 - p_t)**self.gamma * loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError

    @property
    def repr_body(self) -> Dict:
        return {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'reduction': self.reduction,
            'label_smooth': self.label_smooth,
            'num_classes': self.num_classes
        }

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