import math

import torch
from torch import nn
from torch.nn import functional as F

from .eegpt import EEGPT, _positive_int, _probability


class _LinearWithConstraint(nn.Linear):
    def __init__(self, in_features, out_features, max_norm):
        super().__init__(in_features, out_features)
        self.max_norm = max_norm

    def forward(self, x):
        weight = self.weight
        if self.max_norm is not None:
            if weight.requires_grad:
                with torch.no_grad():
                    weight.copy_(torch.renorm(weight, p=2, dim=0, maxnorm=self.max_norm))
                # Several forwards can safely contribute to one backward pass.
                weight = weight.clone()
            else:
                weight = torch.renorm(weight, p=2, dim=0, maxnorm=self.max_norm)
        return F.linear(x, weight, self.bias)


class _EEGPTDownstream(nn.Module):
    def __init__(self, encoder, freeze_encoder):
        super().__init__()
        if not isinstance(encoder, EEGPT) or encoder.return_attention_layer != -1:
            raise ValueError('encoder must be an EEGPT encoder that returns summary tokens.')
        if encoder.channel_ids.numel() != encoder.num_electrodes:
            raise ValueError('encoder must configure ordered channel_names.')
        if not isinstance(freeze_encoder, bool):
            raise ValueError('freeze_encoder must be a boolean.')
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.feature_dim = encoder.num_patches * encoder.embed_num * encoder.hid_channels
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def feature(self, x, flatten=False):
        r'''Extract encoder summaries before the task head and its dropout.

        Args:
            x (torch.Tensor): EEG with shape [batch, electrodes, samples].
            flatten (bool): Return [batch, feature_dim] instead of
                [batch, patches, summary_tokens, hid_channels]. (default: :obj:`False`)

        Frozen encoders stay in evaluation mode. Input gradients remain
        available when requested; freezing does not insert a no-grad region.
        '''
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        output = self.encoder(x)
        return output.flatten(start_dim=1) if flatten else output

    @classmethod
    def from_pretrained(cls, checkpoint_path, encoder_kwargs, prefix='target_encoder.', **kwargs):
        r'''Create a task model from a complete local encoder checkpoint.

        Args:
            checkpoint_path (str): Path to a tensor-only encoder state or a
                safely loadable Lightning state_dict checkpoint.
            encoder_kwargs (dict): Explicit EEGPT constructor settings,
                including ordered channel_names and input geometry.
            prefix (str): Prefix for encoder tensor keys. Use an empty string
                for a plain encoder state. (default: :obj:`'target_encoder.'`)
            **kwargs: Task-model constructor settings other than encoder.

        The task head starts from random weights. This does not load a trained
        classifier or regressor. Use load_state_dict for a complete task model.
        '''
        if not isinstance(encoder_kwargs, dict):
            raise ValueError('encoder_kwargs must be a dictionary of EEGPT settings.')
        encoder = EEGPT.from_pretrained(checkpoint_path, prefix=prefix, **encoder_kwargs)
        return cls(encoder=encoder, **kwargs)


def _validate_task_parameters(outputs, dropout, max_norm):
    _positive_int('outputs', outputs)
    _probability('dropout', dropout)
    if max_norm is not None and (isinstance(max_norm, bool) or
            not isinstance(max_norm, (int, float)) or not math.isfinite(max_norm) or max_norm <= 0):
        raise ValueError('max_norm must be finite and positive, or None.')



class EEGPTClassifier(_EEGPTDownstream):
    r'''Classify EEGPT summaries with a dropout and a constrained linear head.

    Input has shape :obj:`[batch, electrodes, samples]`; output contains
    unnormalized logits with shape :obj:`[batch, num_classes]`. This is a
    direct summary-token classifier. It does not add the original workspace's
    implicit resampling or learned 62-channel mixing, or the official
    repository's optional predictor/reconstructor task architecture.

    .. code-block:: python

        import torch
        from torcheeg.models import EEGPT, EEGPTClassifier

        encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
        model = EEGPTClassifier(encoder, num_classes=2)
        logits = model(torch.randn(2, 3, 1024))

    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        encoder (EEGPT): Configured summary-token encoder with ordered channel_names.
        num_classes (int): Number of output classes. (default: :obj:`2`)
        dropout (float): Task-head dropout probability. (default: :obj:`0.5`)
        max_norm (float, optional): Per-class weight norm limit; None disables projection. (default: :obj:`1.0`)
        freeze_encoder (bool): Freeze encoder parameters and keep its dropout in evaluation mode during head training. An existing encoder's parameter flags are otherwise preserved. (default: :obj:`False`)
    '''
    def __init__(self, encoder, num_classes=2, dropout=0.5, max_norm=1.0, freeze_encoder=False):
        _validate_task_parameters(num_classes, dropout, max_norm)
        super().__init__(encoder, freeze_encoder)
        self.num_classes = num_classes
        self.task_head = nn.Sequential(nn.Dropout(dropout),
            _LinearWithConstraint(self.feature_dim, num_classes, max_norm)).to(encoder.summary_token)

    def forward(self, x):
        return self.task_head(self.feature(x, flatten=True))


class EEGPTRegression(_EEGPTDownstream):
    r'''Predict continuous targets from flattened EEGPT summaries.

    Input has shape :obj:`[batch, electrodes, samples]`; output has shape
    :obj:`[batch, num_outputs]`, including the final singleton axis for scalar
    regression. No output activation or target scaling is applied. Set target
    units in the data pipeline and fit population statistics on training data
    only. This head is not an official pretrained regression checkpoint.

    .. code-block:: python

        import torch
        from torcheeg.models import EEGPT, EEGPTRegression

        encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
        model = EEGPTRegression(encoder)
        predictions = model(torch.randn(2, 3, 1024))

    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        encoder (EEGPT): Configured summary-token encoder with ordered channel_names.
        num_outputs (int): Number of continuous output variables. (default: :obj:`1`)
        dropout (float): Task-head dropout probability. (default: :obj:`0.5`)
        max_norm (float, optional): Per-output weight norm limit; None disables projection. (default: :obj:`1.0`)
        freeze_encoder (bool): Freeze encoder parameters and keep the encoder in evaluation mode during head training. (default: :obj:`False`)
    '''
    def __init__(self, encoder, num_outputs=1, dropout=0.5, max_norm=1.0, freeze_encoder=False):
        _validate_task_parameters(num_outputs, dropout, max_norm)
        super().__init__(encoder, freeze_encoder)
        self.num_outputs = num_outputs
        self.task_head = nn.Sequential(nn.Dropout(dropout),
            _LinearWithConstraint(self.feature_dim, num_outputs, max_norm)).to(encoder.summary_token)

    def forward(self, x):
        return self.task_head(self.feature(x, flatten=True))


class EEGPTEmbedding(_EEGPTDownstream):
    r'''Extract EEGPT embeddings without a learned task head.

    This wrapper returns the encoder summaries or their flattened form. It
    performs no resampling, channel mixing, normalization or feature fitting.
    It preserves input dtype/device and the encoder's parameter ownership.
    Randomly initialized encoder features are not pretrained representations.

    .. code-block:: python

        import torch
        from torcheeg.models import EEGPT, EEGPTEmbedding

        encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
        model = EEGPTEmbedding(encoder, freeze_encoder=True)
        features = model(torch.randn(2, 3, 1024))

    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        encoder (EEGPT): Configured summary-token encoder with ordered channel_names.
        flatten (bool): Return [batch, feature_dim] instead of the four-dimensional summary tensor. (default: :obj:`True`)
        freeze_encoder (bool): Freeze encoder parameters and keep it in evaluation mode. Input gradients remain available when requested. (default: :obj:`False`)
    '''
    def __init__(self, encoder, flatten=True, freeze_encoder=False):
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        super().__init__(encoder, freeze_encoder)
        self.flatten = flatten

    def forward(self, x):
        return self.feature(x, flatten=self.flatten)
