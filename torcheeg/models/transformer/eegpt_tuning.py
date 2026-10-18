import math

import torch
from torch import nn
from torch.nn import functional as F

from .eegpt import EEGPT, _Attention, _Block, _positive_int, _probability
from .eegpt_heads import _EEGPTDownstream
from ._tuning import _Adapter, _LoRALinear


class _AdapterBlock(nn.Module):
    def __init__(self, block, adapter_size, dropout):
        super().__init__()
        # Retain base module/parameter ownership and checkpoint key names.
        self.norm1 = block.norm1
        self.attn = block.attn
        self.drop_path = block.drop_path
        self.norm2 = block.norm2
        self.mlp = block.mlp
        width = block.norm1.normalized_shape[0]
        self.adapter_attn = _Adapter(width, adapter_size, dropout).to(block.norm1.weight)
        self.adapter_mlp = _Adapter(width, adapter_size, dropout).to(block.norm1.weight)
        self.training = block.training
        self.adapter_attn.train(block.training)
        self.adapter_mlp.train(block.training)

    def forward(self, x, return_attention=False):
        attention = self.attn(self.norm1(x), return_attention=return_attention)
        if return_attention:
            return attention
        x = self.adapter_attn(x + self.drop_path(attention))
        return self.adapter_mlp(x + self.drop_path(self.mlp(self.norm2(x))))


class _PrefixAttention(nn.Module):
    def __init__(self, attention, prefix_length):
        super().__init__()
        self.num_heads = attention.num_heads
        self.attn_dropout = attention.attn_dropout
        self.qkv = attention.qkv
        self.proj = attention.proj
        self.proj_drop = attention.proj_drop
        width = attention.qkv.in_features
        shape = (1, self.num_heads, prefix_length, width // self.num_heads)
        self.prefix_key = nn.Parameter(attention.qkv.weight.new_empty(shape))
        self.prefix_value = nn.Parameter(attention.qkv.weight.new_empty(shape))
        nn.init.normal_(self.prefix_key, std=0.02)
        nn.init.normal_(self.prefix_value, std=0.02)
        self.training = attention.training

    def forward(self, x, return_attention=False):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, width // self.num_heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        # Autocast can produce lower-precision Q/K/V from float32 parameters.
        k = torch.cat((self.prefix_key.to(dtype=k.dtype).expand(batch, -1, -1, -1), k), dim=-2)
        v = torch.cat((self.prefix_value.to(dtype=v.dtype).expand(batch, -1, -1, -1), v), dim=-2)
        if return_attention:
            return (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
        x = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.attn_dropout if self.training else 0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, tokens, width)))


def _select_blocks(model, blocks, freeze_base):
    if not isinstance(freeze_base, bool):
        raise ValueError('freeze_base must be a boolean.')
    if isinstance(model, EEGPT):
        encoder = model
    elif isinstance(model, _EEGPTDownstream):
        if model.freeze_encoder:
            raise ValueError('Use freeze_encoder=False on a task model before adding trainable encoder modules.')
        encoder = model.encoder
    else:
        raise ValueError('model must be EEGPT or an EEGPT task model.')
    if blocks is None:
        indices = list(range(len(encoder.blocks)))
    elif isinstance(blocks, (tuple, list)) and blocks:
        indices = list(blocks)
        if any(isinstance(index, bool) or not isinstance(index, int) or
               not 0 <= index < len(encoder.blocks) for index in indices) or len(set(indices)) != len(indices):
            raise ValueError('blocks must contain unique, valid zero-based block indices.')
    else:
        raise ValueError('blocks must be a nonempty list/tuple of indices, or None.')
    if any(not isinstance(encoder.blocks[index], (_Block, _AdapterBlock)) for index in indices):
        raise ValueError('Selected encoder blocks have an unsupported custom implementation.')
    return encoder, indices


def _freeze_base(encoder, freeze_base):
    if not freeze_base:
        return
    encoder.requires_grad_(False)
    # Keep previously installed tuning modules trainable when methods combine.
    for module in encoder.modules():
        if isinstance(module, _Adapter):
            module.requires_grad_(True)
        elif isinstance(module, _PrefixAttention):
            module.prefix_key.requires_grad_(True)
            module.prefix_value.requires_grad_(True)
        elif isinstance(module, _LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)


def add_eegpt_adapters(model, adapter_size=64, dropout=0.1, blocks=None, freeze_base=True):
    r'''Add residual bottleneck adapters to selected EEGPT blocks in place.

    Each selected block receives an adapter after its attention residual and
    after its MLP residual. Zero-initialized up-projections preserve the base
    function at insertion. Base module/parameter objects and their checkpoint
    keys are retained. Build the same tuning structure before restoring an
    augmented state dictionary. Create the optimizer after installing modules.

    - Reference: Houlsby et al., Parameter-Efficient Transfer Learning for NLP, ICML 2019.
    - URL: https://proceedings.mlr.press/v97/houlsby19a.html

    .. code-block:: python

        from torcheeg.models import EEGPT, EEGPTClassifier, add_eegpt_adapters

        encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
        model = EEGPTClassifier(encoder)
        add_eegpt_adapters(model, adapter_size=32)

    Args:
        model (nn.Module): EEGPT encoder or EEGPT task model. Task models must use freeze_encoder=False.
        adapter_size (int): Bottleneck width. (default: :obj:`64`)
        dropout (float): Adapter hidden-activation dropout. (default: :obj:`0.1`)
        blocks (list of int, optional): Unique zero-based block indices, or all blocks. (default: :obj:`None`)
        freeze_base (bool): Freeze encoder base weights and retain all installed adapter/LoRA/prefix parameters as trainable. Task-head flags are preserved. False preserves existing flags. (default: :obj:`True`)

    Returns:
        nn.Module: The supplied model, modified in place. No checkpoint is loaded automatically.
    '''
    _positive_int('adapter_size', adapter_size)
    _probability('dropout', dropout)
    encoder, indices = _select_blocks(model, blocks, freeze_base)
    if any(isinstance(encoder.blocks[index], _AdapterBlock) for index in indices):
        raise ValueError('A selected block already has adapters.')
    replacements = [(index, _AdapterBlock(encoder.blocks[index], adapter_size, dropout)) for index in indices]
    for index, block in replacements:
        encoder.blocks[index] = block
    _freeze_base(encoder, freeze_base)
    return model


def add_eegpt_prefix(model, prefix_length=10, blocks=None, freeze_base=True):
    r'''Prepend learned key/value vectors to selected EEGPT attention layers.

    Prefixes are direct per-layer key/value parameters, without a separate
    reparameterization network. They do not add query/output tokens. Attention
    inspection returns [batch * patches, heads, tokens, tokens + prefix_length]
    at prefixed layers. Prefix insertion changes attention even before
    training; zero prefix values would not preserve the base softmax.
    Modules participate in train/eval, dtype/device moves, copying and state
    dictionaries. Build this structure before loading an augmented state.
    Create the optimizer after installing modules.

    - Reference: Li and Liang, Prefix-Tuning: Optimizing Continuous Prompts for Generation, ACL 2021.
    - URL: https://aclanthology.org/2021.acl-long.353/

    .. code-block:: python

        from torcheeg.models import EEGPT, EEGPTClassifier, add_eegpt_prefix

        encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
        model = EEGPTClassifier(encoder)
        add_eegpt_prefix(model, prefix_length=8)

    Args:
        model (nn.Module): EEGPT encoder or EEGPT task model with freeze_encoder=False.
        prefix_length (int): Learned key/value positions per selected layer. (default: :obj:`10`)
        blocks (list of int, optional): Unique zero-based block indices, or all blocks. (default: :obj:`None`)
        freeze_base (bool): Freeze base encoder weights while retaining existing tuning parameters as trainable; preserve task-head flags. (default: :obj:`True`)

    Returns:
        nn.Module: The supplied model, modified in place. No pretrained performance is implied.
    '''
    _positive_int('prefix_length', prefix_length)
    encoder, indices = _select_blocks(model, blocks, freeze_base)
    if any(not isinstance(encoder.blocks[index].attn, _Attention) for index in indices):
        raise ValueError('Selected attention already has a prefix or uses an unsupported implementation.')
    replacements = [(index, _PrefixAttention(encoder.blocks[index].attn, prefix_length)) for index in indices]
    for index, attention in replacements:
        encoder.blocks[index].attn = attention
    _freeze_base(encoder, freeze_base)
    return model


def add_eegpt_lora(model, rank=4, alpha=8.0, dropout=0.1, blocks=None,
                   target_modules=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'), freeze_base=True):
    r'''Add registered low-rank updates to selected EEGPT linear projections.

    Compute x W^T + b + (alpha / rank) * dropout(x) A B, with input dropout,
    A shaped [in_features, rank] and B shaped [rank, out_features]. B starts
    at zero, so insertion preserves the base function. Base parameters keep
    their original objects and key names; no eval-time weight merging occurs.
    Target names are exact block-relative paths, avoiding substring matches
    against task heads or adapter projections. Install modules before creating
    an optimizer, and recreate their structure before loading augmented state.

    - Reference: Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, ICLR 2022.
    - URL: https://github.com/microsoft/LoRA

    .. code-block:: python

        from torcheeg.models import EEGPT, EEGPTClassifier, add_eegpt_lora

        encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
        model = EEGPTClassifier(encoder)
        add_eegpt_lora(model, rank=4)

    Args:
        model (nn.Module): EEGPT encoder or EEGPT task model with freeze_encoder=False.
        rank (int): Low-rank factor width. (default: :obj:`4`)
        alpha (float): Positive update scale numerator. (default: :obj:`8.0`)
        dropout (float): Input dropout probability for the low-rank branch. (default: :obj:`0.1`)
        blocks (list of int, optional): Unique zero-based block indices, or all blocks. (default: :obj:`None`)
        target_modules (tuple of str): Exact block-relative linear projection paths. (default: :obj:`('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')`)
        freeze_base (bool): Freeze base encoder weights while retaining existing tuning parameters as trainable; preserve task-head flags. (default: :obj:`True`)

    Returns:
        nn.Module: The supplied model, modified in place. No checkpoint is downloaded.
    '''
    _positive_int('rank', rank)
    _probability('dropout', dropout)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive.')
    allowed = ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')
    if (not isinstance(target_modules, (list, tuple)) or not target_modules or
            any(not isinstance(name, str) or name not in allowed for name in target_modules) or
            len(set(target_modules)) != len(target_modules)):
        raise ValueError('target_modules must contain unique exact EEGPT projection paths.')
    encoder, indices = _select_blocks(model, blocks, freeze_base)
    projections = []
    for index in indices:
        for name in target_modules:
            parent_name, attribute = name.split('.')
            parent = getattr(encoder.blocks[index], parent_name)
            linear = getattr(parent, attribute)
            if not isinstance(linear, nn.Linear):
                raise ValueError(f'Projection blocks.{index}.{name} already has LoRA or is unsupported.')
            projections.append((parent, attribute, linear))
    replacements = [(parent, name, _LoRALinear(linear, rank, alpha, dropout))
                    for parent, name, linear in projections]
    for parent, name, linear in replacements:
        setattr(parent, name, linear)
    _freeze_base(encoder, freeze_base)
    return model
