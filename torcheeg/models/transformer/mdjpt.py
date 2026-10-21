"""Explicit interfaces for the separately supplied complete mdJPT backend.

The pinned author source does not state a license. Neither its implementation
nor its weights are bundled or implicitly imported by this module.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from ._utils import _positive_int


def _patch_windows(data, patch_size=50, stride=25):
    """Device-neutral view of complete scalar-channel sliding windows.

    The explicit author factory uses this independently implemented helper
    instead of the original helper's hardcoded CUDA allocation. The complete
    original neural classes and every original state tensor remain unchanged.
    """
    return data[..., 0].unfold(1, patch_size, stride).contiguous()


class MDJPT(nn.Module):
    r"""Complete external mdJPT with explicit physical and grouping protocols.

    Ordinary inputs are float32 [batch,electrodes,625] at 125 Hz. Ordinary
    inference uses native two-row calls, so each native half-group contains
    one window. An odd final window is explicitly duplicated, then its second
    output is discarded. This provides sample-independent transfer features;
    it differs from original multiwindow group normalization. Native paired
    training is exposed separately and never inferred from class labels.

    Args:
        backend (nn.Module): Explicit complete original MultiModel_PL backend.
        channel_names (list or tuple): Ordered physical names of input rows.
        sampling_rate (int): Physical sampling rate, must be 125 Hz.
        allow_interpolation (bool): Permit the author's fixed nearest-neighbor
            projection of missing native sensors. (default: :obj:`False`)
        channel_aliases (dict, optional): Explicit physical-name equivalences,
            such as {'T3': 'T7'}. Unknown or duplicate mapped sensors fail.
        feature_mode (str): 'mean', 'de' or 'projector'. This fixes the native
            feature-map/projector branch for the lifetime of this instance.
            All unused native projection states remain loaded. (default:
            :obj:`'mean'`)
    """

    PRETRAINING_CHANNELS = (
        'FP1',
        'FPZ',
        'FP2',
        'AF3',
        'AF4',
        'F7',
        'F5',
        'F3',
        'F1',
        'FZ',
        'F2',
        'F4',
        'F6',
        'F8',
        'FT7',
        'FC5',
        'FC3',
        'FC1',
        'FCZ',
        'FC2',
        'FC4',
        'FC6',
        'FT8',
        'T7',
        'C5',
        'C3',
        'C1',
        'CZ',
        'C2',
        'C4',
        'C6',
        'T8',
        'TP7',
        'CP5',
        'CP3',
        'CP1',
        'CPZ',
        'CP2',
        'CP4',
        'CP6',
        'TP8',
        'P7',
        'P5',
        'P3',
        'P1',
        'PZ',
        'P2',
        'P4',
        'P6',
        'P8',
        'PO7',
        'PO5',
        'PO3',
        'POZ',
        'PO4',
        'PO6',
        'PO8',
        'O1',
        'OZ',
        'O2',
    )

    def __init__(
        self,
        backend,
        channel_names,
        sampling_rate=125,
        allow_interpolation=False,
        channel_aliases=None,
        feature_mode='mean',
    ):
        super().__init__()
        _positive_int('sampling_rate', sampling_rate)
        if sampling_rate != 125 or not isinstance(allow_interpolation, bool):
            raise ValueError('Require 125 Hz and boolean allow_interpolation.')
        if feature_mode not in ('mean', 'de', 'projector'):
            raise ValueError("feature_mode must be 'mean', 'de' or 'projector'.")
        if not isinstance(backend, nn.Module):
            raise ValueError(
                'Require an explicitly constructed complete author backend.'
            )
        self.allow_interpolation = allow_interpolation
        self.channel_names = (
            tuple(channel_names) if isinstance(channel_names, (tuple, list)) else ()
        )
        self.channel_aliases = (
            {}
            if channel_aliases is None
            else dict(channel_aliases)
            if isinstance(channel_aliases, dict)
            else None
        )
        self.projected_channel_names = self._map_names(
            channel_names, self.channel_aliases
        )
        cfg = getattr(backend, 'cfg', None)
        try:
            native = cfg.model.MLLA
            cnn_cfg = native.cnn
            if (
                cfg.model.encoder != 'MLLA'
                or tuple(backend.uni_channelname) != self.PRETRAINING_CHANNELS
                or tuple(native.uni_channels) != self.PRETRAINING_CHANNELS
                or (
                    native.patch_size,
                    native.patch_stride,
                    native.hidden_dim,
                    native.out_dim,
                    native.depth,
                    native.n_heads,
                )
                != (32, 6, 128, 32, 2, 8)
                or cfg.data_0.fs != 125
                or cfg.data_0.timeLen != 5
                or (
                    cnn_cfg.n_timeFilters,
                    cnn_cfg.n_msFilters,
                    cnn_cfg.n_channs,
                    cnn_cfg.msFilter_timeLen,
                    cnn_cfg.seg_att,
                    cnn_cfg.avgPoolLen,
                    cnn_cfg.timeSmootherLen,
                    cnn_cfg.multiFact,
                )
                != (32, 4, 64, 3, 15, 15, 3, 1)
                or tuple(cnn_cfg.dilation_array) != (1, 3, 6, 12)
                or tuple(cnn_cfg.stratified) != ('initial', 'middle1', 'middle2')
                or cnn_cfg.activ != 'softmax'
                or cnn_cfg.temp != 1.0
                or cnn_cfg.has_att is not True
                or cnn_cfg.global_att is not False
                or cnn_cfg.extract_mode != 'me'
            ):
                raise ValueError('Native geometry differs.')
            layers = backend.MLLA.encoder.blocks
            if len(layers) != 2 or any(
                layer.self_attn.n_heads != 8 for layer in layers
            ):
                raise ValueError('Keep both complete eight-head MLLA layers.')
            if (
                not callable(backend.channel_project)
                or not callable(backend.clisa_loss)
                or not callable(backend.cda_loss)
            ):
                raise ValueError(
                    'Keep original physical projection and both training losses.'
                )
            if cfg.train.loss.to_riem is not False:
                raise ValueError(
                    'The released configuration uses non-Riemannian native CDA.'
                )
        except (AttributeError, TypeError, KeyError, ValueError) as error:
            raise ValueError(
                'Require the complete released MLLA/CNN backend and explicit original configuration.'
            ) from error
        loss_cfg = cfg.train.loss
        if (
            not isinstance(loss_cfg.clisa_loss, bool)
            or not isinstance(loss_cfg.CDA_loss, bool)
            or not math.isfinite(float(loss_cfg.temp))
            or float(loss_cfg.temp) <= 0
            or not math.isfinite(float(loss_cfg.CDA_factor))
            or float(loss_cfg.CDA_factor) < 0
            or getattr(backend.clisa_loss, 'temperature', None) != float(loss_cfg.temp)
        ):
            raise ValueError(
                'Require explicit finite original loss settings and matching contrastive temperature.'
            )
        cnn = backend.cnn_encoder
        if (
            tuple(getattr(cnn, 'stratified', ())) != ('initial', 'middle1', 'middle2')
            or getattr(cnn, 'activ', None) != 'softmax'
            or getattr(cnn, 'temp', None) != 1.0
            or getattr(cnn, 'has_att', None) is not True
            or getattr(cnn, 'global_att', None) is not False
            or getattr(cnn, 'extract_mode', None) != 'me'
            or tuple(getattr(cnn, 'dilation_array', ())) != (1, 3, 6, 12)
            or getattr(backend.MLLA, 'patch_size', None) != 32
            or getattr(backend.MLLA, 'patch_stride', None) != 6
        ):
            raise ValueError(
                'Native non-tensor settings must match the complete released configuration.'
            )
        if any(
            isinstance(module, nn.Dropout) and module.p != 0
            for module in backend.modules()
        ):
            raise ValueError(
                'Keep the released zero-dropout neural configuration for this grouping protocol.'
            )
        table = torch.as_tensor(backend.channel_interpolate)
        if (
            table.shape != (60, 60)
            or table.dtype not in (torch.int32, torch.int64)
            or not torch.equal(
                table.sort(dim=1).values.cpu(), torch.arange(60).expand(60, -1)
            )
        ):
            raise ValueError('Require all sixty complete native neighbor permutations.')
        state = backend.state_dict()
        shapes = {
            'uni_mlp.fc.weight': (64, 60),
            'MLLA.encoder.read_in.weight': (128, 32),
            'MLLA.encoder.read_out.weight': (32, 128),
            'cnn_encoder.timeConv.weight': (32, 1, 1, 30),
            'cnn_encoder.att_conv.weight': (512, 1, 1, 15),
            'cnn_encoder.att_pointConv.weight': (512, 512, 1, 1),
            'cnn_encoder.timeConv1.weight': (512, 1, 1, 3),
            'cnn_encoder.timeConv2.weight': (512, 1, 1, 3),
        }
        for index in range(1, 5):
            shapes['cnn_encoder.msConv' + str(index) + '.weight'] = (128, 1, 64, 3)
        for index in range(2):
            for name in ['W_Q', 'W_K', 'W_V']:
                shapes[f'MLLA.encoder.blocks.{index}.self_attn.{name}.weight'] = (
                    128,
                    128,
                )
        if len(state) != 82 or any(
            key not in state or tuple(state[key].shape) != shape
            for key, shape in shapes.items()
        ):
            raise ValueError('Keep every complete released MLLA/CNN/projector tensor.')
        previous_mode = getattr(backend, '_torcheeg_mdjpt_feature_mode', feature_mode)
        if previous_mode != feature_mode:
            raise ValueError(
                'Construct separate complete backends for different fixed feature views.'
            )
        backend._torcheeg_mdjpt_feature_mode = feature_mode
        # Fix this branch once; forward calls do not mutate shared source flags.
        backend.save_fea = feature_mode != 'projector'
        backend.cnn_encoder.saveFea = backend.save_fea
        self.backend = backend
        self.feature_mode = feature_mode
        self.sampling_rate, self.chunk_size = 125, 625
        self.hid_channels = 1024 if feature_mode == 'projector' else 512
        behavior = dict(
            channel_names=self.channel_names,
            channel_aliases=self.channel_aliases,
            mapped_names=self.projected_channel_names,
            allow_interpolation=allow_interpolation,
            feature_mode=feature_mode,
            sampling_rate=125,
            chunk_size=625,
            native_vocabulary=self.PRETRAINING_CHANNELS,
            neighbor_table=table.tolist(),
            ordinary_grouping='two_rows_one_window_per_half_explicit_duplicate_odd_tail',
            patch_device_policy=getattr(
                backend, '_patch_device_policy', 'original_hardcoded_cuda'
            ),
            contrastive=bool(cfg.train.loss.clisa_loss),
            alignment=bool(cfg.train.loss.CDA_loss),
            temperature=float(cfg.train.loss.temp),
            alignment_factor=float(cfg.train.loss.CDA_factor),
            dropouts=[
                (name, module.p)
                for name, module in backend.named_modules()
                if isinstance(module, nn.Dropout)
            ],
            normalization_eps=[
                (name, module.eps)
                for name, module in backend.named_modules()
                if isinstance(module, nn.LayerNorm)
            ],
        )
        self._digest = hashlib.sha256(
            json.dumps(behavior, sort_keys=True).encode()
        ).digest()
        self.register_buffer(
            'metadata_fingerprint', torch.tensor(list(self._digest), dtype=torch.uint8)
        )

    def _map_names(self, names, aliases):
        if (
            not isinstance(names, (list, tuple))
            or not names
            or not all(isinstance(v, str) and v and v.strip() == v for v in names)
            or len({v.casefold() for v in names}) != len(names)
            or not isinstance(aliases, dict)
            or not all(
                isinstance(k, str) and isinstance(v, str) and k and v
                for k, v in aliases.items()
            )
        ):
            raise ValueError(
                'Require distinct ordered physical names and explicit alias dictionary.'
            )
        folded = {v.casefold() for v in names}
        if len({k.casefold() for k in aliases}) != len(aliases) or any(
            k.casefold() not in folded for k in aliases
        ):
            raise ValueError(
                'Every alias key must name a distinct declared input electrode.'
            )
        mapping = {k.casefold(): v.upper() for k, v in aliases.items()}
        mapped = tuple(mapping.get(v.casefold(), v.upper()) for v in names)
        if len(set(mapped)) != len(mapped) or any(
            v not in self.PRETRAINING_CHANNELS for v in mapped
        ):
            raise ValueError(
                'Unknown or duplicate mapped native sensors; declare physical aliases explicitly.'
            )
        if not self.allow_interpolation and set(mapped) != set(
            self.PRETRAINING_CHANNELS
        ):
            raise ValueError(
                'Missing native sensors require explicit allow_interpolation=True.'
            )
        return mapped

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
                'MDJPT physical projection, grouping or native configuration differs.'
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

    def _validate_input(self, x, channels):
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 3
            or x.dtype != torch.float32
            or x.shape[0] < 1
            or tuple(x.shape[1:]) != (channels, 625)
            or not torch.isfinite(x).all()
        ):
            raise ValueError(
                'Require finite float32 EEG [batch,declared_electrodes,625] at 125 Hz.'
            )

    def _summarize(self, features):
        if self.feature_mode == 'projector':
            return features
        features = features.squeeze(2)
        if self.feature_mode == 'mean':
            return features.mean(-1)
        # Original posthoc DE formula includes +1 inside the logarithm.
        return 0.5 * torch.log(
            2 * math.pi * math.e * features.var(-1, unbiased=False) + 1
        )

    def forward(self, x, return_tokens=False):
        r"""Return sample-independent features, or native feature maps [B,512,99].

        This explicit transfer protocol uses one window per native half-group.
        It is independent of other samples and batch partitions. Projector
        mode returns [B,1024] and does not have a temporal token return view.
        """
        self._validate_input(x, len(self.channel_names))
        if not isinstance(return_tokens, bool) or (
            return_tokens and self.feature_mode == 'projector'
        ):
            raise ValueError(
                'Temporal tokens require a feature-map mode and boolean return_tokens.'
            )
        outputs = []
        for chunk in x.split(2):
            size = chunk.shape[0]
            if size == 1:
                chunk = chunk.expand(2, -1, -1)
            projected = self.backend.channel_project(
                chunk.unsqueeze(1), self.projected_channel_names
            )
            features, _ = self.backend(projected)
            outputs.append(
                (features.squeeze(2) if return_tokens else self._summarize(features))[
                    :size
                ]
            )
        return torch.cat(outputs)

    def forward_native(self, paired_eeg):
        r"""Return original features for explicit groups [2,windows,C,625].

        Windows within each group share the author's normalization statistics.
        The group identities and pair alignment must be specified by the data
        protocol. Other windows can affect a target window in this method.
        """
        if (
            not isinstance(paired_eeg, torch.Tensor)
            or paired_eeg.ndim != 4
            or paired_eeg.shape[0] != 2
        ):
            raise ValueError(
                'Require explicit two-group EEG [2,windows,electrodes,625].'
            )
        values = paired_eeg.flatten(0, 1)
        self._validate_input(values, len(self.channel_names))
        projected = self.backend.channel_project(
            values.unsqueeze(1), self.projected_channel_names
        )
        features, covariance, tokens = self.backend(projected, returnMLLAout=True)
        result = dict(
            features=self._summarize(features),
            covariance_features=covariance,
            channel_tokens=tokens,
        )
        result['projector' if self.feature_mode == 'projector' else 'feature_map'] = (
            features
        )
        return result

    def forward_pretraining(self, domain_batches):
        r"""Compute both original native losses on explicit paired domains.

        Use projector mode. Each list item is a [2,pairs,C,625] tensor using
        the constructor's physical metadata, or a dictionary with 'eeg',
        'channel_names' and optional explicit 'channel_aliases'. Positive
        pairs are corresponding window ranks across the two groups, never
        class labels. Contrastive top-five reporting requires at least three
        pairs per domain. Alignment preserves the original CDA helper's
        reshape convention, not a silently corrected covariance estimator.
        Backpropagate aggregate 'loss' once; alignment_loss is unweighted.
        """
        if (
            self.feature_mode != 'projector'
            or not isinstance(domain_batches, (list, tuple))
            or not domain_batches
        ):
            raise ValueError(
                'Require projector mode and a nonempty list of paired domains.'
            )
        cfg = self.backend.cfg.train.loss
        if not cfg.clisa_loss and not cfg.CDA_loss:
            raise ValueError(
                'At least one original training objective must be enabled.'
            )
        contrastive, covariance = [], []
        for batch in domain_batches:
            if isinstance(batch, dict):
                if not {'eeg', 'channel_names'} <= set(batch) or set(batch) - {
                    'eeg',
                    'channel_names',
                    'channel_aliases',
                }:
                    raise ValueError(
                        'Declare domain EEG, physical names and optional aliases only.'
                    )
                values, names = batch['eeg'], batch['channel_names']
                mapped = self._map_names(names, batch.get('channel_aliases', {}))
            else:
                values, names, mapped = (
                    batch,
                    self.channel_names,
                    self.projected_channel_names,
                )
            if (
                not isinstance(values, torch.Tensor)
                or values.ndim != 4
                or values.shape[0] != 2
                or values.shape[1] < (3 if cfg.clisa_loss else 1)
            ):
                raise ValueError(
                    'Require two aligned nonempty groups and at least three contrastive pairs.'
                )
            flat = values.flatten(0, 1)
            self._validate_input(flat, len(names))
            features, cov = self.backend(
                self.backend.channel_project(flat.unsqueeze(1), mapped)
            )
            covariance.append(cov)
            if cfg.clisa_loss:
                contrastive.append(self.backend.clisa_loss(features)[0])
        zero = covariance[0].new_zeros(())
        contrastive_loss = sum(contrastive, zero)
        alignment_loss = self.backend.cda_loss(covariance) if cfg.CDA_loss else zero
        return dict(
            contrastive_loss=contrastive_loss,
            alignment_loss=alignment_loss,
            loss=contrastive_loss + float(cfg.CDA_factor) * alignment_loss,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        backend,
        channel_names,
        sampling_rate=125,
        allow_interpolation=False,
        channel_aliases=None,
        feature_mode='mean',
        prefix='',
        state_key='state_dict',
    ):
        r"""Strictly load all original tensors using restricted CPU loading.

        The official Lightning checkpoint uses state_key='state_dict' with
        no prefix. Flat complete exports use state_key=None. No unsafe pickle
        fallback, Lightning object reconstruction, partial head
        removal, or source/weight download is performed.
        """
        model = cls(
            backend,
            channel_names,
            sampling_rate,
            allow_interpolation,
            channel_aliases,
            feature_mode,
        )
        if not isinstance(prefix, str) or (
            state_key is not None and not isinstance(state_key, str)
        ):
            raise ValueError('Require explicit string prefix and optional state key.')
        path = Path(checkpoint_path)
        if path.suffix == '.safetensors':
            from safetensors.torch import load_file

            state = load_file(str(path), device='cpu')
        else:
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if state_key is not None:
            if not isinstance(state, dict) or state_key not in state:
                raise ValueError('The declared native state key is absent.')
            state = state[state_key]
        if not isinstance(state, dict) or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError('Require a complete native tensor state dictionary.')
        if prefix and any(not k.startswith(prefix) for k in state):
            raise ValueError('Every tensor must use the explicit common prefix.')
        state = {k[len(prefix) :]: v for k, v in state.items()}
        expected = backend.state_dict()
        if set(state) != set(expected):
            raise ValueError(
                'Keep all original MLLA/CNN/projector weights, including unused states.'
            )
        for key, value in state.items():
            if (
                value.shape != expected[key].shape
                or value.dtype != expected[key].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError('Checkpoint shape/type/value mismatch: ' + key)
        backend.load_state_dict(state, strict=True)
        return model


class MDJPTClassifier(nn.Module):
    r"""New task logits over an explicitly selected complete MDJPT feature view.

    Args:
        backbone (MDJPT): Complete original-backend adapter.
        num_classes (int): New task logit width. (default: :obj:`2`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
    """

    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        _positive_int('num_classes', num_classes)
        if not isinstance(backbone, MDJPT) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require an MDJPT adapter and boolean freeze_backbone.')
        self.backbone, self.freeze_backbone = backbone, freeze_backbone
        self.classifier = nn.Linear(backbone.hid_channels, num_classes).to(
            next(backbone.backend.parameters())
        )
        if freeze_backbone:
            backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        return self.classifier(self.backbone(x))
