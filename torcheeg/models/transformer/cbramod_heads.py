from collections import OrderedDict

import torch
from torch import nn

from ..layers import FourierResample
from .cbramod import CBraMod
from ._utils import _positive_int, _probability


class _CBraModDownstream(nn.Module):
    def __init__(self, backbone, num_electrodes, chunk_size, num_patches, freeze_encoder):
        super().__init__()
        if not isinstance(backbone, CBraMod):
            raise ValueError('backbone must be a CBraMod instance.')
        for name, value in [('num_electrodes', num_electrodes), ('chunk_size', chunk_size),
                            ('num_patches', num_patches)]:
            _positive_int(name, value)
        if chunk_size % num_patches:
            raise ValueError('chunk_size must be divisible by num_patches.')
        if not isinstance(freeze_encoder, bool):
            raise ValueError('freeze_encoder must be a boolean.')
        self.backbone = backbone
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.num_patches = num_patches
        self.freeze_encoder = freeze_encoder
        self.feature_dim = num_electrodes * num_patches * backbone.hid_channels
        self.resample = FourierResample(backbone.patch_size)
        if freeze_encoder:
            backbone.requires_grad_(False)
            backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.backbone.eval()
        return self

    def feature(self, x, flatten=False, mask=None):
        r'''Return patch representations before the task head and its dropout.

        Input has shape [batch, num_electrodes, chunk_size]. Divide the final
        axis into num_patches equal, nonoverlapping segments, then resample
        EACH segment to backbone.patch_size samples. Return features shaped
        [batch, num_electrodes, num_patches, backbone.hid_channels], or flatten
        from axis one. This does not resample the complete window as one period.

        mask optionally selects [batch, num_electrodes, num_patches] patches
        for the backbone's native masking operation. A frozen backbone remains
        in evaluation mode, while input gradients are still available.
        '''
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3 or
                x.shape[0] < 1 or x.shape[1:] != (self.num_electrodes, self.chunk_size)):
            raise ValueError('Input must be real floating [batch, num_electrodes, chunk_size] with nonempty batches.')
        patches = x.reshape(x.shape[0], self.num_electrodes, self.num_patches, -1)
        output = self.backbone.forward_features(self.resample(patches), mask=mask)
        return output.flatten(1) if flatten else output

    @classmethod
    def from_pretrained(cls, checkpoint_path, backbone_kwargs, prefix='', **kwargs):
        r'''Build a task model from a complete local backbone tensor state.

        backbone_kwargs must explicitly configure the CBraMod architecture.
        The safe native loader requires every backbone key, including its
        reconstruction projection, and never downloads weights. The task
        head starts from random weights. This factory does not restore a
        trained task checkpoint: rebuild the same constructor settings and
        use load_state_dict for that purpose.
        '''
        if not isinstance(backbone_kwargs, dict):
            raise ValueError('backbone_kwargs must be a dictionary of CBraMod settings.')
        backbone = CBraMod.from_pretrained(checkpoint_path, prefix=prefix, **backbone_kwargs)
        return cls(backbone=backbone, **kwargs)


class _CBraModPrediction(_CBraModDownstream):
    def __init__(self, backbone, num_electrodes, chunk_size, num_patches,
                 outputs, pooling, head_channels, dropout, freeze_encoder):
        if not isinstance(backbone, CBraMod):
            raise ValueError('backbone must be a CBraMod instance.')
        _positive_int('outputs', outputs)
        _probability('dropout', dropout)
        if pooling not in ('flatten', 'mean'):
            raise ValueError('pooling must be flatten or mean.')
        if head_channels is None:
            head_channels = (4 * backbone.hid_channels, backbone.hid_channels)
        if not isinstance(head_channels, (tuple, list)):
            raise ValueError('head_channels must be a tuple/list of positive integers, or None.')
        for width in head_channels:
            _positive_int('head_channels entry', width)
        super().__init__(backbone, num_electrodes, chunk_size, num_patches, freeze_encoder)
        self.pooling = pooling
        self.head_channels = tuple(head_channels)
        width = self.feature_dim if pooling == 'flatten' else backbone.hid_channels
        layers = OrderedDict()
        for index, hidden in enumerate(head_channels):
            layers[f'linear{index}'] = nn.Linear(width, hidden)
            layers[f'activation{index}'] = nn.ELU()
            layers[f'dropout{index}'] = nn.Dropout(dropout)
            width = hidden
        layers['output'] = nn.Linear(width, outputs)
        self.task_head = nn.Sequential(layers).to(next(backbone.parameters()))

    def forward(self, x, mask=None):
        features = self.feature(x, mask=mask)
        features = features.flatten(1) if self.pooling == 'flatten' else features.mean(dim=(1, 2))
        return self.task_head(features)


class CBraModClassifier(_CBraModPrediction):
    r'''Classify raw EEG with CBraMod patch features and an explicit task head.

    Input is [batch, num_electrodes, chunk_size]; output is unnormalized
    [batch, num_classes] logits. Each equal temporal segment is Fourier
    resampled independently before feature extraction. Set sampling rate,
    physical window duration, channel order, filtering and units in the data
    pipeline. For the canonical pretrained backbone, use 200 samples per
    one-second patch, author-compatible channel order and units of 100 uV.
    A sample count alone cannot establish that preprocessing contract.

    The default flatten -> 4H -> H -> classes head follows the author's
    all_patch_reps classifier, where H is backbone.hid_channels. Select an
    empty head_channels tuple for a linear head; combine it with mean pooling
    for the author's average-pooling linear head. Use (H,) for the two-layer
    flattening head. No softmax or target normalization is applied.

    .. code-block:: python

        import torch
        from torcheeg.models import CBraMod, CBraModClassifier

        # Four seconds at 128 Hz -> four one-second, 200-sample patches.
        model = CBraModClassifier(CBraMod(), num_electrodes=19,
                                 chunk_size=512, num_patches=4, num_classes=2)
        logits = model(torch.randn(2, 19, 512))

    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        backbone (CBraMod): Configured backbone; its reconstruction projection remains in the state but is not used by the task head.
        num_electrodes (int): Ordered input electrodes.
        chunk_size (int): Raw samples per input window, divisible by num_patches.
        num_classes (int): Output classes. (default: :obj:`2`)
        num_patches (int): Equal nonoverlapping temporal segments per window. (default: :obj:`1`)
        pooling (str): Flatten all patch features, or mean over electrode and patch axes. (default: :obj:`'flatten'`)
        head_channels (tuple, optional): Hidden widths; None selects (4H, H), and () selects a linear head. (default: :obj:`None`)
        dropout (float): Dropout after each hidden ELU activation; a linear head has no dropout. (default: :obj:`0.1`)
        freeze_encoder (bool): Freeze backbone parameters and keep its dropout in evaluation mode. Otherwise preserve existing parameter flags. Input gradients remain available. (default: :obj:`False`)
    '''

    def __init__(self, backbone, num_electrodes, chunk_size, num_classes=2, num_patches=1,
                 pooling='flatten', head_channels=None, dropout=0.1, freeze_encoder=False):
        super().__init__(backbone, num_electrodes, chunk_size, num_patches, num_classes,
                         pooling, head_channels, dropout, freeze_encoder)
        self.num_classes = num_classes


class CBraModRegression(_CBraModPrediction):
    r'''Predict continuous targets from CBraMod patch representations.

    Input has shape [batch, num_electrodes, chunk_size]; output has shape
    [batch, num_outputs], retaining the last axis for scalar regression.
    Segment resampling and head configuration follow CBraModClassifier.
    Set physical target units explicitly; fit population target statistics
    on training data only. This head does not apply an output activation.

    .. code-block:: python

        import torch
        from torcheeg.models import CBraMod, CBraModRegression

        model = CBraModRegression(CBraMod(), num_electrodes=19,
                                 chunk_size=800, num_patches=4)
        predictions = model(torch.randn(2, 19, 800))

    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        backbone (CBraMod): Configured backbone, including its unused reconstruction projection.
        num_electrodes (int): Ordered input electrodes.
        chunk_size (int): Raw input samples per window.
        num_outputs (int): Continuous target dimensions. (default: :obj:`1`)
        num_patches (int): Equal temporal segments; chunk_size must be divisible. (default: :obj:`1`)
        pooling (str): Flatten, or mean over electrode and patch axes. (default: :obj:`'flatten'`)
        head_channels (tuple, optional): Hidden widths; None means (4H, H), () means linear. (default: :obj:`None`)
        dropout (float): Dropout after each hidden activation. (default: :obj:`0.1`)
        freeze_encoder (bool): Freeze the backbone and keep it in evaluation mode. (default: :obj:`False`)
    '''

    def __init__(self, backbone, num_electrodes, chunk_size, num_outputs=1, num_patches=1,
                 pooling='flatten', head_channels=None, dropout=0.1, freeze_encoder=False):
        super().__init__(backbone, num_electrodes, chunk_size, num_patches, num_outputs,
                         pooling, head_channels, dropout, freeze_encoder)
        self.num_outputs = num_outputs


class CBraModEmbedding(_CBraModDownstream):
    r'''Extract raw-window CBraMod embeddings without a learned task head.

    Divide [batch, electrodes, chunk_size] into equal segments and Fourier
    resample each to backbone.patch_size. Return flattened features by default,
    or [batch, electrodes, num_patches, backbone.hid_channels] when flatten=False.
    No learned channel mixing, normalization or feature fitting is performed.
    Randomly initialized backbone features are not pretrained representations.

    .. code-block:: python

        import torch
        from torcheeg.models import CBraMod, CBraModEmbedding

        model = CBraModEmbedding(CBraMod(), num_electrodes=19,
                                chunk_size=800, num_patches=4, freeze_encoder=True)
        features = model(torch.randn(2, 19, 800))

    .. automethod:: feature
    .. automethod:: from_pretrained

    Args:
        backbone (CBraMod): Configured backbone with its complete native state.
        num_electrodes (int): Ordered input electrodes.
        chunk_size (int): Raw input samples, divisible by num_patches.
        num_patches (int): Equal temporal segments per window. (default: :obj:`1`)
        flatten (bool): Flatten features from axis one. (default: :obj:`True`)
        freeze_encoder (bool): Freeze backbone parameters and keep it in evaluation mode while retaining input gradients. (default: :obj:`False`)
    '''

    def __init__(self, backbone, num_electrodes, chunk_size, num_patches=1, flatten=True, freeze_encoder=False):
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        super().__init__(backbone, num_electrodes, chunk_size, num_patches, freeze_encoder)
        self.flatten = flatten

    def forward(self, x, mask=None):
        return self.feature(x, flatten=self.flatten, mask=mask)
