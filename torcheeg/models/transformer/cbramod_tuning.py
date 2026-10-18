import math

import torch
from torch import nn
from torch.nn import functional as F

from .cbramod import CBraMod, _CrissCrossBlock, _Encoder, _Attention
from .cbramod_heads import _CBraModDownstream
from ._tuning import _Adapter, _LoRALinear
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
    if type(encoder.encoder) not in (_Encoder, _PromptEncoder):
        raise ValueError('The CBraMod encoder has an unsupported custom implementation.')
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
    if any(type(encoder.encoder.layers[index]) not in (_CrissCrossBlock, _AdapterBlock) for index in indices):
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
    _freeze_base(encoder, freeze_base)
    return model


class _PromptEncoder(_Encoder):
    def __init__(self, encoder, num_electrodes, num_prompts):
        nn.Module.__init__(self)
        self.layers = encoder.layers
        self.num_electrodes = num_electrodes
        self.num_prompts = num_prompts
        reference = self.layers[0].norm1.weight
        self.prompt_tokens = nn.Parameter(reference.new_empty(1, num_electrodes, num_prompts, reference.numel()))
        nn.init.normal_(self.prompt_tokens, std=.02)
        self.training = encoder.training

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.num_electrodes:
            raise ValueError('Prompted features must match the configured electrode count.')
        prompts = self.prompt_tokens.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        features = super().forward(torch.cat((prompts, x), dim=2))
        return features[:, :, self.num_prompts:, :]


def _freeze_base(encoder, freeze_base):
    if not freeze_base:
        return
    encoder.requires_grad_(False)
    for module in encoder.modules():
        if isinstance(module, _Adapter):
            module.requires_grad_(True)
        elif isinstance(module, _PromptEncoder):
            module.prompt_tokens.requires_grad_(True)
        elif isinstance(module, _LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
        elif isinstance(module, _LoRAUpdate):
            module.requires_grad_(True)


def add_cbramod_prompts(model, num_prompts=10, num_electrodes=None, freeze_base=True):
    r'''Prepend learned patch-feature tokens before the CBraMod encoder.

    Prompts have shape [1, electrodes, num_prompts, hid_channels]. They are
    inserted after native patch embedding and positional encoding, passed
    through every criss-cross block, then removed before returning original
    patch features. They add query, key and value positions inside the
    encoder. This preserves the SAME input-token operation implemented by
    both archived get_prefix_model and get_prompt_model; it is not a
    per-layer key/value-only prefix mechanism.

    Prompt insertion changes attention even before training. Zero prompt
    values would not preserve the original softmax. Modules are registered;
    no forward closure captures an external model. Recreate the same tuning
    structure before strict augmented-state loading, and construct the
    optimizer after installation. No input signals or weights are downloaded.

    .. code-block:: python

        from torcheeg.models import CBraMod, CBraModClassifier, add_cbramod_prompts

        model = CBraModClassifier(CBraMod(), num_electrodes=19,
                                 chunk_size=800, num_patches=4)
        add_cbramod_prompts(model, num_prompts=4)

    Args:
        model (nn.Module): CBraMod backbone or task model with freeze_encoder=False.
        num_prompts (int): Learned temporal feature tokens per electrode. (default: :obj:`10`)
        num_electrodes (int, optional): Required for a bare backbone; inferred from a task model. An explicit value must agree with the task. (default: :obj:`None`)
        freeze_base (bool): Freeze original backbone parameters while retaining installed adapters and prompts as trainable. Preserve task-head flags and module train/eval modes. False preserves existing flags. (default: :obj:`True`)

    Returns:
        nn.Module: The supplied model modified in place.
    '''
    _positive_int('num_prompts', num_prompts)
    encoder, _ = _select_blocks(model, None, freeze_base)
    if isinstance(model, _CBraModDownstream):
        if num_electrodes is None:
            num_electrodes = model.num_electrodes
        elif num_electrodes != model.num_electrodes:
            raise ValueError('num_electrodes must agree with the task input geometry.')
    _positive_int('num_electrodes', num_electrodes)
    if isinstance(encoder.encoder, _PromptEncoder):
        raise ValueError('The CBraMod encoder already has prompts.')
    if not isinstance(encoder.encoder, _Encoder):
        raise ValueError('The CBraMod encoder has an unsupported custom implementation.')
    replacement = _PromptEncoder(encoder.encoder, num_electrodes, num_prompts)
    encoder.encoder = replacement
    _freeze_base(encoder, freeze_base)
    return model


class _LoRAUpdate(nn.Module):
    def __init__(self, reference, in_features, out_features, rank, alpha, dropout):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(reference.new_empty(in_features, rank))
        self.lora_B = nn.Parameter(reference.new_zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.T, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(F.linear(self.dropout(x), self.lora_A.T), self.lora_B.T) * self.scaling


# Native packed attention layout follows CBraMod; see LICENSES/CBraMod.txt.
class _LoRAAttention(_Attention):
    def __init__(self, attention, projections, rank, alpha, dropout):
        nn.Module.__init__(self)
        self.num_heads = attention.num_heads
        self.dropout = attention.dropout
        self.in_proj_weight = attention.in_proj_weight
        self.in_proj_bias = attention.in_proj_bias
        self.out_proj = attention.out_proj
        width = attention.in_proj_weight.shape[1]
        for name in ('q', 'k', 'v'):
            attribute = 'lora_' + name
            if hasattr(attention, attribute):
                self.add_module(attribute, getattr(attention, attribute))
            if name + '_proj' in projections:
                update = _LoRAUpdate(attention.in_proj_weight, width, width, rank, alpha, dropout)
                update.train(attention.training)
                self.add_module(attribute, update)
        self.training = attention.training

    def forward(self, x):
        batch, tokens, width = x.shape
        projected = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        updates = []
        for name in ('q', 'k', 'v'):
            module = getattr(self, 'lora_' + name, None)
            updates.append(module(x) if module is not None else projected.new_zeros(x.shape))
        # Retain the native packed layout, including its batch/token axes.
        q, k, v = (projected + torch.cat(updates, dim=-1)).reshape(
            batch, tokens, 3, self.num_heads, width // self.num_heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.dropout if self.training else 0.)
        return self.out_proj(attended.transpose(1, 2).reshape(batch, tokens, width))


def add_cbramod_lora(model, rank=4, alpha=8.0, dropout=0.1, blocks=None,
                     target_modules=('linear1', 'linear2'), freeze_base=True):
    r'''Add registered low-rank updates to selected CBraMod projections.

    Compute x W^T + b + (alpha / rank) * dropout(x) A B. A is shaped
    [in_features, rank], B is [rank, out_features]. B starts at zero, preserving
    deterministic evaluation at insertion. Dropout is applied to INPUT
    features. No evaluation-time weight merging or monkey-patched closures
    are used. Native parameter objects and state keys remain unchanged.

    Exact targets are linear1/linear2, and self_attn_s or self_attn_t followed
    by .q_proj, .k_proj, .v_proj or .out_proj. Q/K/V targets are logical slices
    of the native packed projection; each selected slice has independent A/B
    parameters. The default targets only feed-forward projections, matching
    the original workspace's selection. Explicit attention targets use the
    native batch-first scaled-dot-product path, correcting the archived
    attention wrapper's batch/sequence layout and zero-update behavior.

    Rebuild the same structure before strict augmented tensor-state loading.
    Install modules before constructing the optimizer. New parameters inherit
    dtype/device/mode. Base parameter flags and task-head flags are otherwise
    preserved. Stochastic training dropout need not preserve bitwise outputs.

    - Reference: Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, ICLR 2022.
    - URL: https://github.com/microsoft/LoRA

    .. code-block:: python

        from torcheeg.models import CBraMod, CBraModClassifier, add_cbramod_lora

        model = CBraModClassifier(CBraMod(), num_electrodes=19,
                                 chunk_size=800, num_patches=4)
        add_cbramod_lora(model, rank=4)

    Args:
        model (nn.Module): CBraMod backbone or task model with freeze_encoder=False.
        rank (int): Rank per selected projection. (default: :obj:`4`)
        alpha (float): Positive scaling numerator. (default: :obj:`8.0`)
        dropout (float): Low-rank input dropout probability. (default: :obj:`0.1`)
        blocks (list of int, optional): Unique zero-based block indices, or all blocks. (default: :obj:`None`)
        target_modules (tuple of str): Unique exact block-relative projection names. Attention Q/K/V names select independent slices of packed native weights. (default: :obj:`('linear1', 'linear2')`)
        freeze_base (bool): Freeze original backbone weights and retain installed LoRA, adapters and prompts as trainable. Preserve task-head flags. False preserves existing base flags. (default: :obj:`True`)

    Returns:
        nn.Module: The supplied model, modified in place. No weights are loaded automatically.
    '''
    _positive_int('rank', rank)
    _probability('dropout', dropout)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive.')
    allowed = ('linear1', 'linear2') + tuple(
        axis + '.' + projection for axis in ('self_attn_s', 'self_attn_t')
        for projection in ('q_proj', 'k_proj', 'v_proj', 'out_proj'))
    if (not isinstance(target_modules, (list, tuple)) or not target_modules or
            any(not isinstance(name, str) or name not in allowed for name in target_modules) or
            len(set(target_modules)) != len(target_modules)):
        raise ValueError('target_modules must contain unique exact CBraMod projection paths.')
    encoder, indices = _select_blocks(model, blocks, freeze_base)
    # Check every requested target before allocating or replacing modules.
    for index in indices:
        block = encoder.encoder.layers[index]
        for name in target_modules:
            if name in ('linear1', 'linear2'):
                if type(getattr(block, name)) is not nn.Linear:
                    raise ValueError(f'Projection {index}.{name} already has LoRA or is unsupported.')
            else:
                axis, projection = name.split('.')
                attention = getattr(block, axis)
                if type(attention) not in (_Attention, _LoRAAttention):
                    raise ValueError(f'Attention {index}.{axis} has an unsupported implementation.')
                if projection == 'out_proj':
                    duplicate = type(attention.out_proj) is not nn.Linear
                else:
                    duplicate = hasattr(attention, 'lora_' + projection[0])
                if duplicate:
                    raise ValueError(f'Projection {index}.{name} already has LoRA or is unsupported.')
    replacements = []
    for index in indices:
        block = encoder.encoder.layers[index]
        for name in ('linear1', 'linear2'):
            if name in target_modules:
                linear = getattr(block, name)
                replacements.append((block, name, _LoRALinear(linear, rank, alpha, dropout)))
        for axis in ('self_attn_s', 'self_attn_t'):
            selected = [name.split('.')[1] for name in target_modules if name.startswith(axis + '.')]
            if not selected:
                continue
            attention = getattr(block, axis)
            qkv = [name for name in selected if name != 'out_proj']
            if qkv:
                replacement = _LoRAAttention(attention, qkv, rank, alpha, dropout)
                if 'out_proj' in selected:
                    replacement.out_proj = _LoRALinear(attention.out_proj, rank, alpha, dropout)
                replacements.append((block, axis, replacement))
            else:
                replacements.append((attention, 'out_proj', _LoRALinear(attention.out_proj, rank, alpha, dropout)))
    for parent, attribute, module in replacements:
        setattr(parent, attribute, module)
    _freeze_base(encoder, freeze_base)
    return model
