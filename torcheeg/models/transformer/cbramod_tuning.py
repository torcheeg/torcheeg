from torch import nn

from .cbramod import CBraMod, _CrissCrossBlock
from .cbramod_heads import _CBraModDownstream
from ._tuning import _Adapter
from ._utils import _positive_int, _probability


class _AdapterBlock(_CrissCrossBlock):
    def __init__(self, block, adapter_size, dropout):
        nn.Module.__init__(self)
        # Preserve base component objects and their checkpoint paths.
        for name, module in block.named_children():
            self.add_module(name, module)
        self.adapter = _Adapter(block.norm1.normalized_shape[0], adapter_size, dropout).to(block.norm1.weight)
        self.training = block.training
        self.adapter.train(block.training)

    def forward(self, x):
        return self.adapter(super().forward(x))


def _select_blocks(model, blocks, freeze_base):
    if not isinstance(freeze_base, bool):
        raise ValueError('freeze_base must be a boolean.')
    if isinstance(model, CBraMod):
        encoder = model
    elif isinstance(model, _CBraModDownstream):
        if model.freeze_encoder:
            raise ValueError('Use freeze_encoder=False on a task model before adding trainable encoder modules.')
        encoder = model.backbone
    else:
        raise ValueError('model must be CBraMod or a CBraMod task model.')
    if blocks is None:
        indices = list(range(len(encoder.encoder.layers)))
    elif isinstance(blocks, (tuple, list)) and blocks:
        indices = list(blocks)
        if (any(isinstance(index, bool) or not isinstance(index, int) or
                not 0 <= index < len(encoder.encoder.layers) for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('blocks must contain unique valid zero-based block indices.')
    else:
        raise ValueError('blocks must be a nonempty list/tuple of indices, or None.')
    if any(not isinstance(encoder.encoder.layers[index], _CrissCrossBlock) for index in indices):
        raise ValueError('Selected encoder blocks have an unsupported custom implementation.')
    return encoder, indices


def add_cbramod_adapters(model, adapter_size=64, dropout=0.1, blocks=None, freeze_base=True):
    r'''Add a residual bottleneck adapter after selected CBraMod blocks in place.

    Each adapter follows the complete criss-cross block, after its attention
    and feed-forward residuals. Zero-initialized up-projections preserve the
    base function at insertion in deterministic evaluation. Training dropout
    consumes random numbers; bitwise equality of stochastic forwards is not
    promised. This is a post-block adapter, unlike EEGPT's two adapters per
    block. It retains the original workspace's CBraMod insertion position.

    Base component/parameter objects and checkpoint keys are retained. New
    modules inherit the supplied block's dtype, device and train/eval mode.
    Rebuild the same adapter structure before strict loading of augmented
    tensor state; create the optimizer after installing adapters. No weights
    are downloaded and no pretrained performance is implied.

    - Reference: Houlsby et al., Parameter-Efficient Transfer Learning for NLP, ICML 2019.
    - URL: https://proceedings.mlr.press/v97/houlsby19a.html

    .. code-block:: python

        from torcheeg.models import CBraMod, CBraModClassifier, add_cbramod_adapters

        model = CBraModClassifier(CBraMod(), num_electrodes=19,
                                 chunk_size=800, num_patches=4)
        add_cbramod_adapters(model, adapter_size=32)

    Args:
        model (nn.Module): CBraMod backbone or task model. Tasks must use freeze_encoder=False.
        adapter_size (int): Bottleneck hidden width. (default: :obj:`64`)
        dropout (float): Adapter hidden-activation dropout. (default: :obj:`0.1`)
        blocks (list of int, optional): Unique zero-based block indices, or all blocks. (default: :obj:`None`)
        freeze_base (bool): Freeze backbone base parameters and retain previously installed adapters as trainable. Preserve task-head flags. False preserves existing base parameter flags. This freezes parameters, not module training modes. (default: :obj:`True`)

    Returns:
        nn.Module: The supplied model modified in place.
    '''
    _positive_int('adapter_size', adapter_size)
    _probability('dropout', dropout)
    encoder, indices = _select_blocks(model, blocks, freeze_base)
    if any(isinstance(encoder.encoder.layers[index], _AdapterBlock) for index in indices):
        raise ValueError('A selected block already has an adapter.')
    replacements = [(index, _AdapterBlock(encoder.encoder.layers[index], adapter_size, dropout)) for index in indices]
    for index, block in replacements:
        encoder.encoder.layers[index] = block
    if freeze_base:
        encoder.requires_grad_(False)
        for module in encoder.modules():
            if isinstance(module, _Adapter):
                module.requires_grad_(True)
    return model
