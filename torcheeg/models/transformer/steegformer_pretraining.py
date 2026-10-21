# ST-EEGFormer adaptation; see LICENSES/STEEGFormer.txt and its notice.
# Copyright (c) 2025 LIUYIN YANG.
import math
from argparse import Namespace
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from ._utils import _positive_int
from .steegformer import STEEGFormer, _Block, _ChannelEmbedding, _TemporalEmbedding


_VARIANTS = {
    'small': dict(
        hid_channels=512,
        mlp_channels=2048,
        depth=8,
        num_heads=8,
        decoder_channels=384,
        decoder_depth=4,
        decoder_num_heads=16,
    ),
    'base': dict(
        hid_channels=768,
        mlp_channels=3072,
        depth=12,
        num_heads=12,
        decoder_channels=512,
        decoder_depth=8,
        decoder_num_heads=16,
    ),
    'large': dict(
        hid_channels=1024,
        mlp_channels=4096,
        depth=24,
        num_heads=16,
        decoder_channels=512,
        decoder_depth=8,
        decoder_num_heads=16,
    ),
}


class STEEGFormerPretrainer(STEEGFormer):
    r"""Complete ST-EEGFormer masked autoencoder, including its native decoder.

    - Related Project: https://github.com/LiuyinYang1101/STEEGFormer

    Input is real floating EEG [batch, electrodes, samples], with exact
    learned channel indices. Native recordings use 128 Hz and 16-sample
    patches. No preprocessing is performed. Tokens follow temporal-major,
    channel-minor order. forward returns (loss, prediction, mask), where
    prediction is [batch, patches * electrodes, patch_size] and mask is
    zero for visible tokens and one for reconstructed tokens.

    encoder_position_offset=0 retains the author's regular training path.
    The author's demonstration path uses 1 instead; select it explicitly.
    Decoder positions always begin at 1. Inherited encoder feature methods
    retain STEEGFormer's position-1 downstream contract and do not mask.

    Small's decoder width is 384, despite the original factory name saying
    dec256. All encoder and decoder state keys are retained. Use for_variant
    or from_pretrained with an explicit variant for the three native sizes.

    Args:
        decoder_channels (int): Decoder width. (default: :obj:`384`)
        decoder_depth (int): Decoder transformer blocks. (default: :obj:`4`)
        decoder_num_heads (int): Decoder attention heads. (default: :obj:`16`)
        norm_pix_loss (bool): Normalize targets with native unbiased patch
            variance and epsilon 1e-6. (default: :obj:`False`)
        encoder_position_offset (int): Native regular (0) or demonstration
            (1) temporal positions. (default: :obj:`0`)
        **kwargs: Encoder settings accepted by :class:`STEEGFormer`.
    """

    def __init__(
        self,
        decoder_channels=384,
        decoder_depth=4,
        decoder_num_heads=16,
        norm_pix_loss=False,
        encoder_position_offset=0,
        **kwargs,
    ):
        for name, value in [
            ('decoder_channels', decoder_channels),
            ('decoder_depth', decoder_depth),
            ('decoder_num_heads', decoder_num_heads),
        ]:
            _positive_int(name, value)
        if decoder_channels % 2 or decoder_channels % decoder_num_heads:
            raise ValueError(
                'decoder_channels must be even and divisible by decoder_num_heads.'
            )
        if not isinstance(norm_pix_loss, bool):
            raise ValueError('norm_pix_loss must be a boolean.')
        if type(encoder_position_offset) is not int or encoder_position_offset not in (
            0,
            1,
        ):
            raise ValueError('encoder_position_offset must be 0 or 1.')
        # Native MAE has no dropout or stochastic depth. Do not silently
        # introduce a downstream regularization policy into pretraining.
        for key in ('dropout', 'attention_dropout', 'drop_path_rate'):
            if kwargs.get(key, 0.0) != 0.0:
                raise ValueError(f'{key} must be zero for native MAE pretraining.')
        if kwargs.get('qkv_bias', True) is not True:
            raise ValueError('qkv_bias must be True for native MAE pretraining.')
        super().__init__(**kwargs)
        if norm_pix_loss and self.patch_size < 2:
            raise ValueError('Unbiased target variance requires patch_size >= 2.')
        self.norm_pix_loss = norm_pix_loss
        self.encoder_position_offset = encoder_position_offset
        self.dec_channel_emd = _ChannelEmbedding(
            self.num_channel_embeddings, decoder_channels
        )
        self.dec_temporal_emd = _TemporalEmbedding(decoder_channels, self.max_patches)
        self.decoder_embed = nn.Linear(self.hid_channels, decoder_channels)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_channels))
        self.decoder_blocks = nn.ModuleList(
            [
                _Block(
                    decoder_channels,
                    decoder_num_heads,
                    4 * decoder_channels,
                    True,
                    0.0,
                    0.0,
                    0.0,
                    self.norm.eps,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_channels, eps=self.norm.eps)
        self.decoder_pred = nn.Linear(decoder_channels, self.patch_size)
        self.apply(self._init_pretraining_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _init_pretraining_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @classmethod
    def for_variant(cls, variant, **kwargs):
        r"""Construct the complete native small, base or large architecture.

        Geometry is fixed by variant. Signal and target policies remain
        explicit keyword arguments. This method does not load weights.
        """
        if not isinstance(variant, str) or variant not in _VARIANTS:
            raise ValueError('variant must be small, base or large.')
        geometry = _VARIANTS[variant]
        fixed = dict(
            patch_size=16,
            num_channel_embeddings=145,
            max_patches=511,
            qkv_bias=True,
            norm_eps=1e-6,
            **geometry,
        )
        for name, value in fixed.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(
                    f'{name} conflicts with the native {variant} architecture.'
                )
        return cls(**dict(kwargs, **fixed))

    def _validate_eeg(self, eeg, channel_indices):
        if (
            not isinstance(eeg, torch.Tensor)
            or not eeg.is_floating_point()
            or eeg.ndim != 3
            or min(eeg.shape) < 1
            or eeg.shape[-1] % self.patch_size
            or not 1 <= eeg.shape[-1] // self.patch_size <= self.max_patches
        ):
            raise ValueError(
                'EEG must be real floating [batch, electrodes, samples] with complete patches within max_patches.'
            )
        if (
            not isinstance(channel_indices, torch.Tensor)
            or channel_indices.dtype not in (torch.int32, torch.int64)
            or channel_indices.ndim not in (1, 2)
            or channel_indices.shape[-1] != eeg.shape[1]
        ):
            raise ValueError(
                'channel_indices must be integer [electrodes], [1, electrodes] or [batch, electrodes].'
            )
        if channel_indices.ndim == 1:
            channel_indices = channel_indices.unsqueeze(0)
        if channel_indices.shape[0] not in (1, eeg.shape[0]):
            raise ValueError('Channel-index batch size must be one or match EEG.')
        indices = channel_indices.to(device=eeg.device).expand(eeg.shape[0], -1)
        return torch.where(
            (indices < 0) | (indices >= self.num_channel_embeddings),
            self.num_channel_embeddings,
            indices,
        )

    def patchify(self, eeg):
        r"""Return complete temporal-major raw target patches [batch, tokens, patch_size]."""
        if (
            not isinstance(eeg, torch.Tensor)
            or not eeg.is_floating_point()
            or eeg.ndim != 3
            or min(eeg.shape) < 1
            or eeg.shape[-1] % self.patch_size
        ):
            raise ValueError('EEG must contain complete real floating patches.')
        return (
            eeg.unfold(-1, self.patch_size, self.patch_size)
            .permute(0, 2, 1, 3)
            .reshape(eeg.shape[0], -1, self.patch_size)
        )

    def unpatchify(self, patches, num_electrodes):
        r"""Invert patchify without overlap, truncation or a hardcoded patch size."""
        _positive_int('num_electrodes', num_electrodes)
        if (
            not isinstance(patches, torch.Tensor)
            or patches.ndim != 3
            or min(patches.shape) < 1
            or patches.shape[-1] != self.patch_size
            or patches.shape[1] % num_electrodes
        ):
            raise ValueError(
                'patches must have shape [batch, temporal_patches * electrodes, patch_size].'
            )
        return (
            patches.reshape(patches.shape[0], -1, num_electrodes, self.patch_size)
            .permute(0, 2, 1, 3)
            .reshape(patches.shape[0], num_electrodes, -1)
        )

    def random_masking(self, tokens, mask_ratio=0.75, generator=None):
        r"""Shuffle tokens per sample; return visible tokens, binary mask and inverse permutation.

        A supplied torch.Generator owns only masking randomness. Save its
        state alongside model and optimizer states for exact continuation.
        At least one token must be masked so the reconstruction loss exists.
        A ratio of one deliberately retains only CLS in the encoder.
        """
        if (
            isinstance(mask_ratio, bool)
            or not isinstance(mask_ratio, (float, int))
            or not math.isfinite(mask_ratio)
            or not 0.0 < mask_ratio <= 1.0
        ):
            raise ValueError('mask_ratio must be finite and in (0, 1].')
        if (
            not isinstance(tokens, torch.Tensor)
            or not tokens.is_floating_point()
            or tokens.ndim != 3
            or min(tokens.shape) < 1
        ):
            raise ValueError(
                'tokens must be nonempty real floating [batch, tokens, width].'
            )
        batch, length, width = tokens.shape
        keep = int(length * (1.0 - mask_ratio))
        noise = torch.rand(batch, length, device=tokens.device, generator=generator)
        shuffle = torch.argsort(noise, dim=1)
        restore = torch.argsort(shuffle, dim=1)
        visible = torch.gather(
            tokens, 1, shuffle[:, :keep].unsqueeze(-1).expand(-1, -1, width)
        )
        mask = torch.ones(batch, length, device=tokens.device)
        mask[:, :keep] = 0
        return visible, torch.gather(mask, 1, restore), restore

    def forward_encoder(self, eeg, channel_indices, mask_ratio=0.75, generator=None):
        r"""Encode visible tokens using the explicitly selected native temporal offset."""
        indices = self._validate_eeg(eeg, channel_indices)
        x = self.patch_embed(eeg)
        batch, patches, electrodes, width = x.shape
        offset = self.encoder_position_offset
        temporal = self.enc_temporal_emd.pe[:, offset : offset + patches].unsqueeze(2)
        x = (x + temporal + self.enc_channel_emd(indices).unsqueeze(1)).reshape(
            batch, -1, width
        )
        x, mask, restore = self.random_masking(x, mask_ratio, generator)
        cls = (self.cls_token + self.enc_temporal_emd.pe[:, :1]).expand(batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, restore, (batch, patches, electrodes, width)

    def forward_decoder(self, latent, channel_indices, patch_shape, ids_restore):
        r"""Restore original token order and reconstruct all patches, including visible ones."""
        batch, patches, electrodes, _ = patch_shape
        if channel_indices.ndim == 1:
            channel_indices = channel_indices.unsqueeze(0)
        indices = channel_indices.to(device=latent.device).expand(batch, -1)
        indices = torch.where(
            (indices < 0) | (indices >= self.num_channel_embeddings),
            self.num_channel_embeddings,
            indices,
        )
        x = self.decoder_embed(latent)
        masked = self.mask_token.expand(
            batch, ids_restore.shape[1] + 1 - x.shape[1], -1
        )
        x_patch = torch.cat((x[:, 1:], masked), dim=1)
        x_patch = torch.gather(
            x_patch, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        temporal = self.dec_temporal_emd.pe[:, 1 : patches + 1].unsqueeze(2)
        temporal = temporal.expand(batch, patches, electrodes, -1).reshape(
            batch, patches * electrodes, -1
        )
        channels = (
            self.dec_channel_emd(indices)
            .unsqueeze(1)
            .expand(-1, patches, -1, -1)
            .reshape(batch, patches * electrodes, -1)
        )
        # Retain the native left-associative addition for FP32 checkpoints.
        x_patch = x_patch + temporal + channels
        cls = x[:, :1] + self.dec_temporal_emd.pe[:, :1]
        x = torch.cat((cls, x_patch), dim=1)
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:]

    def forward_loss(self, eeg, prediction, mask):
        r"""Native masked mean-squared error; normalized targets use unbiased variance."""
        target = self.patchify(eeg)
        if (
            not isinstance(prediction, torch.Tensor)
            or not prediction.is_floating_point()
            or not isinstance(mask, torch.Tensor)
            or prediction.shape != target.shape
            or mask.shape != target.shape[:2]
        ):
            raise ValueError('Prediction or mask shape does not match the EEG target.')
        if not torch.all((mask == 0) | (mask == 1)) or mask.sum() <= 0:
            raise ValueError(
                'mask must be binary with at least one reconstructed token.'
            )
        return self._reconstruction_loss(eeg, prediction, mask)

    def _reconstruction_loss(self, eeg, prediction, mask):
        target = self.patchify(eeg)
        if self.norm_pix_loss:
            target = (target - target.mean(-1, keepdim=True)) / (
                target.var(-1, keepdim=True) + 1e-6
            ).sqrt()
        return ((prediction - target).square().mean(-1) * mask).sum() / mask.sum()

    def forward(self, eeg, channel_indices, mask_ratio=0.75, generator=None):
        latent, mask, restore, shape = self.forward_encoder(
            eeg, channel_indices, mask_ratio, generator
        )
        prediction = self.forward_decoder(latent, channel_indices, shape, restore)
        return self._reconstruction_loss(eeg, prediction, mask), prediction, mask

    @classmethod
    def from_pretrained(
        cls, checkpoint_path, variant=None, prefix='', allow_namespace=False, **kwargs
    ):
        r"""Load every encoder and decoder tensor, with strict keys, shapes and finite values.

        Accept a bare state or exactly one model/state_dict container.
        Unselected prefix components are ignored only with an explicit prefix.
        Loading uses CPU weights_only=True and mmap=True. Native Small/Base
        metadata needs explicit allow_namespace=True on runtimes providing
        safe_globals; no unsafe pickle fallback or persistent allowlist is used.
        No decoder exclusion, partial match, downloads or geometry inference.
        """
        if not isinstance(prefix, str) or not isinstance(allow_namespace, bool):
            raise ValueError('prefix must be a string and allow_namespace a boolean.')
        context = nullcontext()
        if allow_namespace:
            if not hasattr(torch.serialization, 'safe_globals'):
                raise RuntimeError(
                    'This runtime lacks safe_globals; convert the verified file to tensor-only state first.'
                )
            if Namespace not in torch.serialization.get_safe_globals():
                context = torch.serialization.safe_globals([Namespace])
        with context:
            checkpoint = torch.load(
                Path(checkpoint_path), map_location='cpu', weights_only=True, mmap=True
            )
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Checkpoint must contain a tensor state dictionary.')
        containers = [key for key in ('model', 'state_dict') if key in checkpoint]
        if len(containers) > 1:
            raise ValueError('Checkpoint has ambiguous model/state_dict containers.')
        state = checkpoint[containers[0]] if containers else checkpoint
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) for key in state
        ):
            raise ValueError('Checkpoint state keys must be strings.')
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        # Meta construction avoids allocating a second full Large model.
        with torch.device('meta'):
            model = (
                cls(**kwargs) if variant is None else cls.for_variant(variant, **kwargs)
            )
        expected = model.state_dict()
        missing, extra = set(expected) - set(selected), set(selected) - set(expected)
        if missing or extra:
            raise ValueError(
                f'Incomplete ST-EEGFormer MAE state: missing={sorted(missing)}, extra={sorted(extra)}.'
            )
        for name, value in selected.items():
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.shape != expected[name].shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f'Checkpoint tensor shape/type/finite mismatch: {name}.'
                )
        model.load_state_dict(selected, strict=True, assign=True)
        return model
