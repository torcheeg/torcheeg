"""Independent interfaces for explicitly supplied complete SleepFM backends.

Author source and checkpoints are separately supplied CC-BY-NC-4.0 materials.
"""

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _positive_int


def _ordered_names(values, allowed=None):
    if (
        not isinstance(values, (tuple, list))
        or not values
        or not all(isinstance(v, str) and v for v in values)
    ):
        raise ValueError('Require nonempty ordered string names.')
    if len(set(values)) != len(values) or (
        allowed is not None and any(v not in allowed for v in values)
    ):
        raise ValueError('Names must be distinct and use the declared vocabulary.')
    return tuple(values)


def _load_complete(backend, path, prefix, state_key):
    if not isinstance(prefix, str) or (
        state_key is not None and not isinstance(state_key, str)
    ):
        raise ValueError('Require an explicit string prefix and optional state key.')
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file

        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, weights_only=True, map_location='cpu')
    if state_key is not None:
        if not isinstance(state, dict) or state_key not in state:
            raise ValueError('The declared checkpoint state key is absent.')
        state = state[state_key]
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError('Require a complete flat tensor state.')
    if prefix and any(not name.startswith(prefix) for name in state):
        raise ValueError('Every tensor must use the explicit prefix.')
    state = {name[len(prefix) :]: value for name, value in state.items()}
    expected = backend.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            'Require the complete native state without dropping or adding tensors.'
        )
    for name, value in state.items():
        if (
            value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError('Checkpoint shape/type/value mismatch: ' + name)
    backend.load_state_dict(state, strict=True)


class _SleepFMMetadata(nn.Module):
    def __init__(self, backend, metadata):
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise ValueError(
                'Require an explicitly constructed trusted local nn.Module backend.'
            )
        self.backend = backend
        self._digest = hashlib.sha256(json.dumps(metadata).encode()).digest()
        self.register_buffer(
            'metadata_fingerprint', torch.tensor(list(self._digest), dtype=torch.uint8)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        value = state_dict.get(prefix + 'metadata_fingerprint')
        expected = torch.tensor(list(self._digest), dtype=torch.uint8)
        if isinstance(value, torch.Tensor) and (
            value.dtype != expected.dtype or not torch.equal(value.cpu(), expected)
        ):
            error_msgs.append(
                'SleepFM configured physical metadata, order or variant differs.'
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class SleepFM(_SleepFMMetadata):
    r"""Native SleepFM upstream features for one explicit physiological modality.

    Input is [batch,electrodes,samples] at 128 Hz. Use complete five-second
    (640-sample) patches, at most 128 patches per call. Preprocessing and
    amplitude scaling are explicit caller responsibilities. Modality groups
    use the same complete native encoder, as in author embedding generation.

    Args:
        backend (nn.Module): Complete local author SetTransformer backend.
        channel_names (list or tuple): Ordered physical signal names.
        modality (str): One of BAS, RESP, EKG, EMG. (default: :obj:`'BAS'`)
    """

    def __init__(self, backend, channel_names, modality='BAS'):
        names = _ordered_names(channel_names)
        if modality not in ('BAS', 'RESP', 'EKG', 'EMG'):
            raise ValueError('Require an explicit supported physiological modality.')
        super().__init__(backend, ['upstream', names, modality, 128, 640])
        self.channel_names, self.modality = names, modality
        self.num_electrodes, self.sampling_rate, self.patch_size, self.hid_channels = (
            len(names),
            128,
            640,
            128,
        )

    def forward(self, x, padding_mask=None, return_tokens=False):
        r"""Return pooled [B,128] features, or contextual [B,patches,128].

        padding_mask is boolean [B,C]; True denotes an absent padded channel.
        At least one channel must remain in every example. No time padding
        is inferred or accepted by the upstream native encoder.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_floating_point():
            raise ValueError(
                'Require floating signals[batch,electrodes,samples] at128Hz.'
            )
        batch, channels, samples = x.shape
        if (
            batch < 1
            or channels != self.num_electrodes
            or samples < 640
            or samples % 640
            or samples > 640 * 128
        ):
            raise ValueError(
                'Require the declared channels and complete640-sample patches, at most128.'
            )
        if not isinstance(return_tokens, bool):
            raise ValueError('return_tokens must be boolean.')
        if padding_mask is None:
            padding_mask = torch.zeros(
                batch, channels, device=x.device, dtype=torch.bool
            )
        if (
            not isinstance(padding_mask, torch.Tensor)
            or padding_mask.dtype != torch.bool
            or padding_mask.shape != (batch, channels)
            or padding_mask.device != x.device
        ):
            raise ValueError('Require a same-device boolean channel padding mask[B,C].')
        if padding_mask.all(1).any():
            raise ValueError(
                'Each example must retain at least one physiological channel.'
            )
        pooled, tokens = self.backend(x, padding_mask)
        if pooled.shape != (batch, 128) or tokens.shape != (batch, samples // 640, 128):
            raise ValueError(
                'Require complete native pooled and contextual128-wide features.'
            )
        return tokens if return_tokens else pooled

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        backend,
        channel_names,
        modality='BAS',
        prefix='module.',
        state_key='state_dict',
    ):
        r"""Strictly load the complete separately supplied upstream checkpoint."""
        model = cls(backend, channel_names, modality)
        _load_complete(backend, checkpoint_path, prefix, state_key)
        return model


class _SleepFMSequence(_SleepFMMetadata):
    def __init__(
        self, backend, modality_names, variant, max_steps, demographic_names=None
    ):
        names = _ordered_names(modality_names, ('BAS', 'RESP', 'EKG', 'EMG'))
        super().__init__(
            backend, [variant, names, demographic_names, 128, '5s_features']
        )
        self.modality_names, self.max_steps = names, max_steps

    def _inputs(self, x, padding_mask):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or not x.is_floating_point():
            raise ValueError(
                'Require floating pretrained embeddings[B,modalities,steps,128].'
            )
        batch, modalities, steps, width = x.shape
        if (
            batch < 1
            or modalities != len(self.modality_names)
            or steps < 1
            or steps > self.max_steps
            or width != 128
        ):
            raise ValueError(
                'Embedding shape exceeds the complete native modality/time/width configuration.'
            )
        if padding_mask is None:
            padding_mask = torch.zeros(
                batch, modalities, steps, device=x.device, dtype=torch.bool
            )
        if (
            not isinstance(padding_mask, torch.Tensor)
            or padding_mask.dtype != torch.bool
            or padding_mask.shape != x.shape[:3]
            or padding_mask.device != x.device
        ):
            raise ValueError('Require a same-device boolean mask[B,modalities,steps].')
        # Native heads obtain temporal validity from modality0 and channel
        # validity from step0. Reject grids those two projections cannot encode.
        temporal, spatial = padding_mask[:, 0], padding_mask[:, :, 0]
        if spatial[:, 0].any() or temporal[:, 0].any():
            raise ValueError(
                'The first declared modality and first step must be valid in each example.'
            )
        if not torch.equal(padding_mask, spatial.unsqueeze(-1) | temporal.unsqueeze(1)):
            raise ValueError(
                'Padding must separate into modality padding and temporal padding.'
            )
        lengths = (~temporal).sum(1)
        if not torch.equal(
            temporal,
            torch.arange(steps, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1),
        ):
            raise ValueError('Temporal padding must be a contiguous right suffix.')
        return padding_mask, temporal, lengths


class SleepFMStaging(_SleepFMSequence):
    r"""Complete released five-class SleepFM staging head over upstream tokens.

    Input [B,modalities,steps,128] uses explicit five-second token boundaries.
    The original bidirectional LSTM is not packed. To keep right-padding out
    of valid predictions, this adapter calls it on each unpadded sequence and
    pads only returned logits with zero. This preserves the original model
    on each actual sequence, while excluding artificial padded future input.

    Args:
        backend (nn.Module): Complete local author SleepEventLSTMClassifier.
        modality_names (list or tuple): Ordered physiological modality names.
    """

    def __init__(self, backend, modality_names):
        super().__init__(backend, modality_names, 'sleep-staging', 8196)

    def forward(self, x, padding_mask=None):
        r"""Return logits[B,steps,5] and a boolean temporal padding mask[B,steps]."""
        mask, temporal, lengths = self._inputs(x, padding_mask)
        outputs = []
        for index, length in enumerate(lengths.tolist()):
            logits, _ = self.backend(
                x[index : index + 1, :, :length], mask[index : index + 1, :, :length]
            )
            if logits.shape != (1, length, 5):
                raise ValueError(
                    'Require all five complete native staging output columns.'
                )
            outputs.append(F.pad(logits, (0, 0, 0, x.shape[2] - length)))
        return torch.cat(outputs, dim=0), temporal

    @classmethod
    def from_pretrained(
        cls, checkpoint_path, backend, modality_names, prefix='module.', state_key=None
    ):
        r"""Strictly load every native staging tensor, including the five-class head."""
        model = cls(backend, modality_names)
        _load_complete(backend, checkpoint_path, prefix, state_key)
        return model


class SleepFMDiagnosis(_SleepFMSequence):
    r"""Complete released 1065-column SleepFM Cox log-risk head.

    Requires caller-supplied demographic columns already encoded in the
    native training convention. Names and order are mandatory metadata;
    this interface never invents demographic values, normalizes them or
    labels log-risk outputs as calibrated disease probabilities.

    Args:
        backend (nn.Module): Complete local author diagnosis model with demographics.
        modality_names (list or tuple): Ordered physiological modality names.
        demographic_names (list or tuple): Explicit names of the two native columns.
    """

    def __init__(self, backend, modality_names, demographic_names):
        names = _ordered_names(demographic_names)
        if len(names) != 2:
            raise ValueError(
                'Require explicit names for both native demographic columns.'
            )
        super().__init__(backend, modality_names, 'diagnosis', 6480, names)
        self.demographic_names = names

    def forward(self, x, demographics, padding_mask=None):
        r"""Return all original Cox log-risk columns[B,1065], without sigmoid."""
        mask, _, _ = self._inputs(x, padding_mask)
        if (
            not isinstance(demographics, torch.Tensor)
            or demographics.shape != (x.shape[0], 2)
            or not demographics.is_floating_point()
            or demographics.device != x.device
            or not torch.isfinite(demographics).all()
        ):
            raise ValueError(
                'Require two finite, explicitly encoded same-device demographic columns per example.'
            )
        output = self.backend(x, mask, demographics)
        if output.shape != (x.shape[0], 1065):
            raise ValueError(
                'Require every one of the1065native diagnosis output columns.'
            )
        return output

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        backend,
        modality_names,
        demographic_names,
        prefix='module.',
        state_key=None,
    ):
        r"""Strictly load every original diagnosis tensor and demographic projection."""
        model = cls(backend, modality_names, demographic_names)
        _load_complete(backend, checkpoint_path, prefix, state_key)
        return model


class SleepFMClassifier(nn.Module):
    r"""A new task head over complete native SleepFM upstream features.

    Args:
        backbone (SleepFM): Explicit complete native upstream adapter.
        num_classes (int): New task logit width. (default: :obj:`2`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
    """

    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        _positive_int('num_classes', num_classes)
        if not isinstance(backbone, SleepFM) or not isinstance(freeze_backbone, bool):
            raise ValueError(
                'Require a SleepFM upstream adapter and boolean freeze_backbone.'
            )
        self.backbone, self.freeze_backbone = backbone, freeze_backbone
        self.classifier = nn.Linear(128, num_classes).to(
            next(backbone.backend.parameters())
        )
        if freeze_backbone:
            backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x, padding_mask=None):
        return self.classifier(self.backbone(x, padding_mask))
