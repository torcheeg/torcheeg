"""Interfaces for an explicitly supplied, separately installed NeuroRVQ backend.

Author source and official weights are CC-BY-NC-4.0. Neither is bundled here.
"""
import hashlib
import json

import torch
from torch import nn

from ._utils import _positive_int


class _NeuroRVQInput(nn.Module):
    """Shared physical metadata; no neural backend or feature semantics."""

    def __init__(self, channel_names, channel_vocab, sampling_rate=200):
        super().__init__()
        _positive_int('sampling_rate', sampling_rate)
        if sampling_rate != 200:
            raise ValueError('The released backend requires EEG sampled at 200 Hz.')
        def names(values):
            if not isinstance(values, (tuple, list)) or not values or not all(isinstance(v, str) and v for v in values):
                raise ValueError('Require nonempty ordered string electrode names.')
            normalized = tuple(v.casefold() for v in values)
            if len(set(normalized)) != len(normalized):
                raise ValueError('Electrode names must be distinct ignoring case.')
            return normalized
        channel_names, channel_vocab = names(channel_names), names(channel_vocab)
        if any(name not in channel_vocab for name in channel_names):
            raise ValueError('Every input electrode must occur in the backend vocabulary.')
        self.channel_names, self.channel_vocab = channel_names, channel_vocab
        self.sampling_rate, self.num_electrodes = 200, len(channel_names)
        self.patch_size, self.max_patches = 200, 256
        self._spatial_ids = tuple(channel_vocab.index(v) for v in channel_names)
        self._digest = hashlib.sha256(json.dumps([channel_names, channel_vocab, 200]).encode()).digest()
        self.register_buffer('spatial_ids', torch.tensor(self._spatial_ids, dtype=torch.long))
        self.register_buffer('channel_fingerprint', torch.tensor(list(self._digest), dtype=torch.uint8))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        expected = {'spatial_ids': torch.tensor(self._spatial_ids),
                    'channel_fingerprint': torch.tensor(list(self._digest), dtype=torch.uint8)}
        for name, configured in expected.items():
            value = state_dict.get(prefix+name)
            if isinstance(value, torch.Tensor) and (value.dtype != configured.dtype or not torch.equal(value.cpu(), configured)):
                error_msgs.append('NeuroRVQ electrode order, vocabulary or physical metadata differs: '+name)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _inputs(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_floating_point():
            raise ValueError('Require floating EEG[batch,electrodes,samples].')
        batch, channels, samples = x.shape
        if batch < 1 or channels != self.num_electrodes or samples < 200 or samples % 200:
            raise ValueError('Require nonempty EEG in the declared order and complete 200-sample patches.')
        patches = samples // 200
        if patches > self.max_patches:
            raise ValueError('Input exceeds the complete temporal embedding table.')
        temporal = torch.arange(self.max_patches-patches, self.max_patches, device=x.device).repeat(channels).expand(batch, -1)
        spatial = self.spatial_ids.repeat_interleave(patches).expand(batch, -1)
        return x.reshape(batch, channels, patches, 200), temporal, spatial


class NeuroRVQ(_NeuroRVQInput):
    r'''Complete external NeuroRVQ with explicit ordered electrode metadata.

    The backend must retain all 32 released prediction heads. Construct it
    from the documented pinned author pretraining source. Parameter-free
    identity views expose the four raw feature branches; they add no random
    normalization parameters. Input is [batch,electrodes,samples], sampled
    at 200 Hz in the amplitude scale required by the chosen data workflow.
    Temporal positions are right-aligned as in the original inference code.
    Spatial positions use the original zero-based vocabulary without offset.

    Args:
        backend (nn.Module): Explicit complete local author backend.
        channel_names (list or tuple): Ordered names matching the input.
        channel_vocab (list or tuple): Original ordered electrode vocabulary.
        sampling_rate (int): Physical sampling rate, must be 200 Hz.
    '''
    def __init__(self, backend, channel_names, channel_vocab, sampling_rate=200):
        if not isinstance(backend, nn.Module):
            raise ValueError('Require an explicitly constructed nn.Module backend.')
        super().__init__(channel_names, channel_vocab, sampling_rate)
        if getattr(backend, 'patch_size', None) != 200 or getattr(backend, 'embed_dim', None) != 200:
            raise ValueError('Require the released 200-sample/200-wide backend.')
        if not getattr(backend, 'use_as_encoder', False) or not getattr(backend, 'use_for_pretraining', False):
            raise ValueError('Require the complete encoder with pretraining heads enabled.')
        positions, times = getattr(backend, 'pos_embed', None), getattr(backend, 'time_embed', None)
        if not isinstance(positions, torch.Tensor) or positions.shape != (len(channel_vocab)+1, 200):
            raise ValueError('Vocabulary must match the complete spatial embedding table.')
        if not isinstance(times, torch.Tensor) or times.shape != (256, 200):
            raise ValueError('Require the complete 256-position temporal table.')
        for index in range(1, 33):
            head = getattr(backend, 'head_pre_'+str(index), None)
            if not isinstance(head, nn.Linear) or (head.in_features, head.out_features) != (200, 8192):
                raise ValueError('Require all 32 complete 8192-class prediction heads.')
        self.backend = backend
        self.hid_channels = 800

    def forward(self, x, return_tokens=False):
        r'''Return mean features [B,800], or per-patch features [B,C,P,800].'''
        if not isinstance(return_tokens, bool):
            raise ValueError('return_tokens must be boolean.')
        patches, temporal, spatial = self._inputs(x)
        branches = self.backend(patches, temporal, spatial, return_patch_tokens=True)
        shape = (x.shape[0], self.num_electrodes*patches.shape[2], 200)
        if not isinstance(branches, (tuple, list)) or len(branches) != 5 or any(
                not isinstance(branch, torch.Tensor) or branch.shape != shape for branch in branches[:4]):
            raise ValueError('Backend must return four complete patch-feature branches and metadata.')
        tokens = torch.cat(branches[:4], dim=-1)
        if return_tokens:
            return tokens.reshape(x.shape[0], self.num_electrodes, patches.shape[2], 800)
        return tokens.mean(dim=1)

    def forward_masked(self, x, mask):
        r'''Return native symmetric prediction logits for all 32 RVQ heads.

        mask is boolean [B,C,P]. Outputs are [32,masked_count,8192] and
        [32,unmasked_count,8192], in the native boolean-index order. Both
        partitions must be nonempty. Obtain target RVQ IDs separately from
        the matching tokenizer; this method does not fabricate targets.
        '''
        patches, temporal, spatial = self._inputs(x)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != patches.shape[:3] or mask.device != x.device:
            raise ValueError('Require a same-device boolean mask[batch,electrodes,patches].')
        if not mask.any() or mask.all():
            raise ValueError('Both mask partitions must be nonempty.')
        return self.backend(patches, temporal, spatial, bool_masked_pos=mask.flatten(1), use_for_pretraining=True)

    @classmethod
    def from_pretrained(cls, checkpoint_path, backend, channel_names, channel_vocab, sampling_rate=200, prefix=''):
        r'''Strictly load a complete local flat tensor checkpoint on CPU.

        Construction is explicit. This method never imports author source,
        downloads files, drops prediction heads or uses unsafe pickle loading.
        '''
        model = cls(backend, channel_names, channel_vocab, sampling_rate)
        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        if str(checkpoint_path).endswith('.safetensors'):
            from safetensors.torch import load_file
            state = load_file(str(checkpoint_path), device='cpu')
        else:
            state = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError('Require a complete flat native tensor state dictionary.')
        if prefix and any(not key.startswith(prefix) for key in state):
            raise ValueError('Every checkpoint key must use the explicit prefix.')
        state = {k[len(prefix):]: v for k, v in state.items()}
        expected = backend.state_dict()
        if set(state) != set(expected):
            raise ValueError('Require every native state tensor, including all 32 prediction heads.')
        for name, value in state.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
                raise ValueError('Checkpoint shape/type/value mismatch: '+name)
        backend.load_state_dict(state, strict=True)
        return model


class NeuroRVQClassifier(nn.Module):
    r'''A new classification head over complete external NeuroRVQ features.

    Args:
        backbone (NeuroRVQ): Explicit complete native-backend adapter.
        num_classes (int): New task logit width. (default: :obj:`2`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
    '''
    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        if not isinstance(backbone, NeuroRVQ) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require a NeuroRVQ adapter and boolean freeze_backbone.')
        _positive_int('num_classes', num_classes)
        self.backbone, self.freeze_backbone = backbone, freeze_backbone
        self.classifier = nn.Linear(800, num_classes).to(backbone.backend.pos_embed)
        if freeze_backbone:
            backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        return self.classifier(self.backbone(x))
