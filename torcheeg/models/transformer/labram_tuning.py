import math

import torch
from torch import nn

from .labram_encoder import LaBraMEncoder, _Attention, _Block
from .labram_heads import _LaBraMDownstream
from ._tuning import _Adapter, _LoRALinear
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


class _PrefixAttention(_Attention):
    def __init__(self, attention, prefix_length):
        nn.Module.__init__(self)
        self.num_heads = attention.num_heads
        self.qkv = attention.qkv
        self.q_bias = attention.q_bias
        self.v_bias = attention.v_bias
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.attn_drop = attention.attn_drop
        self.proj = attention.proj
        self.proj_drop = attention.proj_drop
        width = attention.qkv.in_features
        self.prefix_tokens = nn.Parameter(
            attention.qkv.weight.new_empty(
                prefix_length, 2, self.num_heads, width // self.num_heads
            )
        )
        nn.init.normal_(self.prefix_tokens, std=0.02)
        self.training = attention.training

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(x)
        if self.q_bias is not None:
            qkv = qkv + torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        q, k, v = (
            qkv.reshape(batch, tokens, 3, self.num_heads, width // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        if self.q_norm is not None:
            q = self.q_norm(q.flatten(0, 2)).reshape_as(q).to(v.dtype)
            k = self.k_norm(k.flatten(0, 2)).reshape_as(k).to(v.dtype)
        # Native keys are normalized first. Learned prefix keys bypass k_norm.
        prefix = self.prefix_tokens.to(dtype=v.dtype)
        prefix_k = prefix[:, 0].unsqueeze(0).expand(batch, -1, -1, -1).transpose(1, 2)
        prefix_v = prefix[:, 1].unsqueeze(0).expand(batch, -1, -1, -1).transpose(1, 2)
        k = torch.cat((prefix_k, k), dim=2)
        v = torch.cat((prefix_v, v), dim=2)
        attention = ((q * (width // self.num_heads) ** -0.5) @ k.transpose(-2, -1)).softmax(dim=-1)
        x = self.attn_drop(attention) @ v
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, tokens, width)))


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
        elif isinstance(module, _PrefixAttention):
            module.prefix_tokens.requires_grad_(True)
        elif isinstance(module, _LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)


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
        freeze_base (bool): Freeze native encoder parameters and retain installed adapters/prefixes/LoRA as trainable. Preserve task-head flags. False preserves existing base/adapter flags. (default: :obj:`True`)

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


def add_labram_prefix(model, prefix_length=10, blocks=None, freeze_base=True):
    r"""Prepend learned attention keys and values to selected LaBraM blocks.

    Native query/key normalization occurs before prefix insertion. Learned
    prefix keys bypass key normalization, matching the archived workspace
    implementation. Prefix vectors are not input tokens: they receive no
    temporal/channel positions, generate no queries, and do not change the
    output token count. Even zero prefix vectors change the softmax
    denominator; insertion does not preserve the base function.

    Each selected attention layer owns an independent prefix_tokens parameter
    of shape [prefix_length, 2, heads, head_width], initialized with normal
    standard deviation 0.02. Native modules, parameters, checkpoint paths and
    mixed child modes are preserved. Prefixes inherit the attention dtype and
    device; attention dropout follows normal train/eval calls.

    Install modules before the optimizer. Load native state before insertion,
    or rebuild the same augmented structure before strict restoration. Task
    models require freeze_encoder=False. Frozen base weights still transmit
    input gradients. This helper supports LaBraMEncoder and its task wrappers;
    the existing LaBraM classification API remains separate.

    - Reference: Li and Liang, Prefix-Tuning, ACL 2021.
    - URL: https://aclanthology.org/2021.acl-long.353/

    .. code-block:: python

        from torcheeg.models import LaBraMEncoder, LaBraMClassifier, add_labram_prefix

        model = LaBraMClassifier(LaBraMEncoder(), ['Fp1', 'C3', 'C4'], chunk_size=800)
        add_labram_prefix(model, prefix_length=10)

    Args:
        model (nn.Module): LaBraMEncoder or LaBraM task wrapper; tasks require freeze_encoder=False.
        prefix_length (int): Positive learned key/value length per layer. (default: :obj:`10`)
        blocks (list of int, optional): Unique zero-based indices, or all blocks. (default: :obj:`None`)
        freeze_base (bool): Freeze native encoder parameters and keep installed prefixes/adapters/LoRA trainable. Preserve task-head flags. False preserves existing flags. (default: :obj:`True`)

    Returns:
        nn.Module: The supplied model, modified in place.
    """
    _positive_int('prefix_length', prefix_length)
    encoder, indices = _select_blocks(model, blocks, freeze_base)
    if any(type(encoder.blocks[index].attn) is not _Attention for index in indices):
        raise ValueError('Selected attention already has prefixes or uses a custom implementation.')
    replacements = [
        (index, _PrefixAttention(encoder.blocks[index].attn, prefix_length)) for index in indices
    ]
    for index, attention in replacements:
        encoder.blocks[index].attn = attention
    _freeze_base(encoder, freeze_base)
    return model


def add_labram_lora(
    model,
    rank=4,
    alpha=8.0,
    dropout=0.1,
    blocks=None,
    target_modules=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    freeze_base=True,
):
    r"""Add registered low-rank updates to selected LaBraM projections.

    Compute x W^T + b + (alpha / rank) * dropout(x) A B. A is shaped
    [in_features, rank], B is [rank, out_features]. B starts at zero, so
    installation preserves deterministic evaluation outputs. Input dropout
    belongs to the registered module and follows normal train/eval calls.
    Native parameter objects and checkpoint names remain unchanged. There
    is no evaluation-time weight merging or monkey-patched forward closure.

    Exact block-relative paths avoid accidentally selecting task heads or
    adapter projections. attn.qkv covers the complete packed Q/K/V update,
    including the key projection; native optional Q/V biases remain separate.
    Both ordinary and prefix attention call the projection module, so QKV
    low-rank updates participate in computation. This corrects the archived
    helper's ineffective QKV forward patch under a functional linear call.

    Install before the optimizer. Load complete native weights before
    insertion, or recreate the complete augmented structure before strict
    restoration. New parameters inherit projection dtype, device and mode.
    Task wrappers require freeze_encoder=False. Task-head flags and mixed
    child modes are preserved. All installation orders with adapters and
    prefixes retain previously installed tuning parameters as trainable
    when freeze_base=True. False preserves existing flags.

    - Reference: Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, ICLR 2022.
    - URL: https://github.com/microsoft/LoRA

    .. code-block:: python

        from torcheeg.models import LaBraMEncoder, LaBraMClassifier, add_labram_lora

        model = LaBraMClassifier(LaBraMEncoder(), ['Fp1', 'C3', 'C4'], chunk_size=800)
        add_labram_lora(model, rank=4)

    Args:
        model (nn.Module): LaBraMEncoder or LaBraM task wrapper with freeze_encoder=False.
        rank (int): Positive rank per selected projection. (default: :obj:`4`)
        alpha (float): Finite positive update scaling numerator. (default: :obj:`8.0`)
        dropout (float): Low-rank input dropout probability. (default: :obj:`0.1`)
        blocks (list of int, optional): Unique zero-based indices, or all blocks. (default: :obj:`None`)
        target_modules (tuple of str): Unique exact block-relative projection paths. (default: :obj:`('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')`)
        freeze_base (bool): Freeze native encoder weights and retain installed LoRA, prefixes and adapters as trainable. Preserve task-head flags. False preserves existing flags. (default: :obj:`True`)

    Returns:
        nn.Module: The supplied model, modified in place. No weights are downloaded.
    """
    _positive_int('rank', rank)
    _probability('dropout', dropout)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError('alpha must be finite and positive.')
    allowed = ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')
    if (
        not isinstance(target_modules, (list, tuple))
        or not target_modules
        or any(not isinstance(name, str) or name not in allowed for name in target_modules)
        or len(set(target_modules)) != len(target_modules)
    ):
        raise ValueError('target_modules must contain unique exact LaBraM projection paths.')
    encoder, indices = _select_blocks(model, blocks, freeze_base)
    projections = []
    for index in indices:
        for name in target_modules:
            parent_name, attribute = name.split('.')
            parent = getattr(encoder.blocks[index], parent_name)
            linear = getattr(parent, attribute)
            if type(linear) is not nn.Linear:
                raise ValueError(
                    f'Projection blocks.{index}.{name} already has LoRA or is unsupported.'
                )
            projections.append((parent, attribute, linear))
    replacements = [
        (parent, name, _LoRALinear(linear, rank, alpha, dropout))
        for parent, name, linear in projections
    ]
    for parent, name, linear in replacements:
        setattr(parent, name, linear)
    _freeze_base(encoder, freeze_base)
    return model
