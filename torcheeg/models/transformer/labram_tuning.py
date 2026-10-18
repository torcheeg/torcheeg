from torch import nn

from .labram_encoder import LaBraMEncoder, _Block
from .labram_heads import _LaBraMDownstream
from ._tuning import _Adapter
from ._utils import _positive_int, _probability


class _AdapterBlock(_Block):
    def __init__(self, block, adapter_size, dropout):
        nn.Module.__init__(self)
        # Preserve component/parameter ownership and native checkpoint paths.
        for name, module in block.named_children():
            self.add_module(name, module)
        self.register_parameter('gamma_1', block.gamma_1)
        self.register_parameter('gamma_2', block.gamma_2)
        width = block.norm1.normalized_shape[0]
        self.adapter_attn = _Adapter(width, adapter_size, dropout).to(block.norm1.weight)
        self.adapter_mlp = _Adapter(width, adapter_size, dropout).to(block.norm1.weight)
        self.training = block.training
        self.adapter_attn.train(block.training)
        self.adapter_mlp.train(block.training)

    def forward(self, x):
        update = self.attn(self.norm1(x))
        x = self.adapter_attn(
            x + self.drop_path(update if self.gamma_1 is None else self.gamma_1 * update)
        )
        update = self.mlp(self.norm2(x))
        return self.adapter_mlp(
            x + self.drop_path(update if self.gamma_2 is None else self.gamma_2 * update)
        )


def _select_blocks(model, blocks, freeze_base):
    if not isinstance(freeze_base, bool):
        raise ValueError('freeze_base must be a boolean.')
    if isinstance(model, LaBraMEncoder):
        encoder = model
    elif isinstance(model, _LaBraMDownstream):
        if model.freeze_encoder:
            raise ValueError(
                'Use freeze_encoder=False on task models before adding trainable encoder modules.'
            )
        encoder = model.backbone
    else:
        raise ValueError('model must be LaBraMEncoder or a LaBraM downstream task model.')
    if blocks is None:
        indices = list(range(len(encoder.blocks)))
    elif isinstance(blocks, (list, tuple)) and blocks:
        indices = list(blocks)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(encoder.blocks)
            for index in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError('blocks must contain unique valid zero-based block indices.')
    else:
        raise ValueError('blocks must be a nonempty list/tuple of indices, or None.')
    if any(type(encoder.blocks[index]) not in (_Block, _AdapterBlock) for index in indices):
        raise ValueError('Selected blocks have an unsupported custom implementation.')
    return encoder, indices


def _freeze_base(encoder, freeze_base):
    if not freeze_base:
        return
    encoder.requires_grad_(False)
    for module in encoder.modules():
        if isinstance(module, _Adapter):
            module.requires_grad_(True)


def add_labram_adapters(model, adapter_size=64, dropout=0.1, blocks=None, freeze_base=True):
    r"""Add two residual bottleneck adapters to each selected LaBraM block.

    One adapter follows the complete attention residual, including native
    layer scaling and stochastic depth. A second follows the complete MLP
    residual. This preserves the archived workspace's two insertion positions.
    Zero-initialized up-projections preserve deterministic base outputs at
    insertion. Training dropout consumes random numbers, so bitwise equality
    of stochastic training forwards is not promised.

    Native components and parameter objects, including optional gamma_1 and
    gamma_2 scales, retain their checkpoint paths. New modules inherit each
    selected block's dtype, device and train/eval mode. Frozen base parameters
    remain frozen; freeze_base does not change module training modes. Task
    wrappers must use freeze_encoder=False so their train() method can keep
    the new adapters active. Task-head parameter flags are preserved.

    Install modules before creating the optimizer. Load a complete native
    encoder before installation, or rebuild the same augmented structure
    before strict restoration of a trained task. No weights are downloaded
    and adding adapters does not establish pretrained benchmark performance.

    - Reference: Houlsby et al., Parameter-Efficient Transfer Learning for NLP, ICML 2019.
    - URL: https://proceedings.mlr.press/v97/houlsby19a.html

    .. code-block:: python

        from torcheeg.models import LaBraMEncoder, LaBraMClassifier, add_labram_adapters

        model = LaBraMClassifier(LaBraMEncoder(), ['Fp1', 'C3', 'C4'], chunk_size=800)
        add_labram_adapters(model, adapter_size=32)

    Args:
        model (nn.Module): LaBraMEncoder or LaBraM task wrapper; tasks require freeze_encoder=False.
        adapter_size (int): Positive bottleneck width. (default: :obj:`64`)
        dropout (float): Adapter hidden-activation dropout. (default: :obj:`0.1`)
        blocks (list of int, optional): Unique zero-based indices, or all blocks. (default: :obj:`None`)
        freeze_base (bool): Freeze all encoder base parameters and retain installed adapters as trainable. Preserve task-head flags. False preserves existing base/adapter flags. (default: :obj:`True`)

    Returns:
        nn.Module: The supplied model, modified in place.
    """
    _positive_int('adapter_size', adapter_size)
    _probability('dropout', dropout)
    encoder, indices = _select_blocks(model, blocks, freeze_base)
    if any(isinstance(encoder.blocks[index], _AdapterBlock) for index in indices):
        raise ValueError('A selected block already has adapters.')
    replacements = [
        (index, _AdapterBlock(encoder.blocks[index], adapter_size, dropout)) for index in indices
    ]
    for index, block in replacements:
        encoder.blocks[index] = block
    _freeze_base(encoder, freeze_base)
    return model
