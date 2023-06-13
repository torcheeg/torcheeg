from typing import List

import torch.nn as nn

from .dann_like import _DANNLikeTrainer


class DANNTrainer(_DANNLikeTrainer):
    r'''
    The individual differences and nonstationary of EEG signals make it difficult for deep learning models trained on the training set of subjects to correctly classify test samples from unseen subjects, since the training set and test set come from different data distributions. Domain adaptation is used to address the problem of distribution drift between training and test sets and thus achieves good performance in subject-independent (cross-subject) scenarios. This class supports the implementation of Domain Adversarial Neural Networks (DANN) for deep domain adaptation.

    NOTE: DANN belongs to unsupervised domain adaptation methods, which only use labeled source and unlabeled target data. This means that the target dataset does not have to return labels.

    - Paper: Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks[J]. The journal of machine learning research, 2016, 17(1): 2096-2030.
    - URL: https://arxiv.org/abs/1505.07818
    - Related Project: https://github.com/fungtion/DANN/blob/master/train/main.py

    .. code-block:: python

        trainer = DANNTrainer(extractor,
                              classifier,
                              domain_classifier,
                              num_classes=10,
                              devices=1,
                              accelerator='gpu')
        trainer.fit(source_loader, target_loader, val_loader)
        trainer.test(test_loader)

    Args:
        extractor (nn.Module): The feature extraction model learns the feature representation of the EEG signal by forcing the correlation matrixes of source and target data to be close.
        classifier (nn.Module): The classification model learns the classification task with the source labeled data based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function.
        domain_classifier (nn.Module): The classification model, learning to discriminate between the source and target domains based on the feature of the feature extraction model. The dimension of its output should be equal to the number of categories in the dataset. The output layer does not need to have a softmax activation function or a gradient reverse layer (which is already included in the implementation).
        num_classes (int, optional): The number of categories in the dataset. (default: :obj:`None`)
        lr (float): The learning rate. (default: :obj:`0.0001`)
        weight_decay (float): The weight decay. (default: :obj:`0.0`)
        weight_domain (float): The weight of the DANN loss (default: :obj:`1.0`)
        alpha_scheduler (bool): Whether to use a scheduler for the alpha of the DANN loss, growing from 0 to 1 following the schedule from the DANN paper. (default: :obj:`True`)
        lr_scheduler (bool): Whether to use a scheduler for the learning rate, as defined in the DANN paper. (default: :obj:`False`)
        warmup_epochs (int): The number of epochs for the warmup phase, during which the weight of the DANN loss is 0. (default: :obj:`0`)
        devices (int): The number of devices to use. (default: :obj:`1`)
        accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
        metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy'. (default: :obj:`["accuracy"]`)

    .. automethod:: fit
    .. automethod:: test
    '''
    def __init__(self,
                 extractor: nn.Module,
                 classifier: nn.Module,
                 domain_classifier: nn.Module,
                 num_classes: int,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 weight_domain: float = 1.0,
                 alpha_scheduler: bool = True,
                 lr_scheduler: bool = False,
                 warmup_epochs: int = 0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"]):

        super(DANNTrainer, self).__init__(extractor=extractor,
                                          classifier=classifier,
                                          domain_classifier=domain_classifier,
                                          num_classes=num_classes,
                                          lr=lr,
                                          weight_decay=weight_decay,
                                          weight_domain=weight_domain,
                                          alpha_scheduler=alpha_scheduler,
                                          lr_scheduler=lr_scheduler,
                                          warmup_epochs=warmup_epochs,
                                          devices=devices,
                                          accelerator=accelerator,
                                          metrics=metrics)
