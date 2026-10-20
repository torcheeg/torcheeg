"""Interfaces for a separately supplied complete Uni-DMFM author backend.

Author code and weights are not bundled. Obtain the source and model under
applicable original terms; the pinned source has no stated license.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from ._utils import _positive_int


class UniDMFM(nn.Module):
    r"""Complete external Uni-DMFM with explicit physical input metadata.

    Signals have shape [batch,electrodes,time] at 200 Hz, with between one
    and 100 complete 200-sample patches. Channel positions are ordered native
    ranks, not inferred physical coordinates. Padding is unsupported because
    native GroupNorm and attention mix sensors and patches.

    Args:
        backend (nn.Module): Explicit complete original author model.
        channel_names (list or tuple): Ordered physical electrode names.
        sampling_rate (int): Physical sampling rate, must be 200 Hz.
    """

    def __init__(self, backend, channel_names, sampling_rate=200):
        super().__init__()
        _positive_int('sampling_rate', sampling_rate)
        if sampling_rate != 200:
            raise ValueError('The documented pretrained protocol requires 200 Hz.')
        if (
            not isinstance(channel_names, (tuple, list))
            or not 1 <= len(channel_names) <= 256
            or not all(
                isinstance(v, str) and v.strip() == v and v for v in channel_names
            )
            or len({v.casefold() for v in channel_names}) != len(channel_names)
        ):
            raise ValueError(
                'Require one to 256 distinct ordered physical channel names.'
            )
        if not isinstance(backend, nn.Module):
            raise ValueError(
                'Require an explicitly constructed complete nn.Module backend.'
            )
        embedding = getattr(backend, 'patch_embedding', None)
        encoder = getattr(backend, 'transformer_encoder', None)
        layers = getattr(encoder, 'layers', [])
        if (
            getattr(backend, 'd_model', None) != 384
            or getattr(backend, 'patch_size', None) != 200
            or getattr(backend, 'stft_params', None) != {'n_fft': 64, 'hop_length': 16}
            or len(layers) != 12
            or embedding is None
            or tuple(getattr(embedding, 'channel_pos_emb', torch.empty(0)).shape)
            != (1, 256, 1, 384)
            or tuple(getattr(embedding, 'patch_pos_emb', torch.empty(0)).shape)
            != (1, 1, 100, 384)
        ):
            raise ValueError(
                'Require the complete released 12-layer/384-wide Uni-DMFM backend.'
            )
        rates = []
        for layer in layers:
            if (
                getattr(layer, 'd_model', None) != 384
                or getattr(layer, 'nhead', None) != 12
                or getattr(layer, 'group_size', None) != 4
                or getattr(getattr(layer, 'temporal_attn', None), 'embed_dim', None)
                != 384
                or not getattr(
                    getattr(layer, 'temporal_attn', None), 'batch_first', False
                )
            ):
                raise ValueError('Keep every complete native factored-attention layer.')
            drop = getattr(layer, 'drop_path', None)
            rate = (
                0.0
                if isinstance(drop, nn.Identity)
                else getattr(drop, 'drop_prob', None)
            )
            if rate is None:
                raise ValueError('Require an explicit native drop-path configuration.')
            rates.append(float(rate))
        expected_heads = {
            'time_reconstruction_head.weight': (200, 384),
            'freq_reconstruction_head.0.weight': (384, 384),
            'freq_reconstruction_head.2.weight': (429, 384),
            'contrastive_head.0.weight': (192, 384),
            'contrastive_head.2.weight': (256, 192),
        }
        state = backend.state_dict()
        if any(
            key not in state or tuple(state[key].shape) != shape
            for key, shape in expected_heads.items()
        ):
            raise ValueError('Keep all native time, frequency and contrastive heads.')
        if not callable(getattr(backend, 'forward_features', None)):
            raise ValueError('Require the original complete token feature method.')
        self.backend = backend
        self.channel_names = tuple(channel_names)
        self.num_electrodes = len(channel_names)
        self.sampling_rate, self.patch_size, self.hid_channels = 200, 200, 384
        self._digest = hashlib.sha256(
            json.dumps(
                {
                    'channel_names': self.channel_names,
                    'sampling_rate': 200,
                    'patch_size': 200,
                    'stft_params': backend.stft_params,
                    'drop_path_rates': rates,
                    'schema': [
                        (k, list(v.shape), str(v.dtype)) for k, v in state.items()
                    ],
                }
            ).encode()
        ).digest()
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
                'Uni-DMFM channel order, physical metadata or native configuration differs.'
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

    def _patches(self, x):
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 3
            or not x.is_floating_point()
            or x.shape[0] < 1
            or x.shape[1] != self.num_electrodes
            or not 200 <= x.shape[2] <= 20000
            or x.shape[2] % 200
            or not torch.isfinite(x).all()
        ):
            raise ValueError(
                'Require finite floating EEG [B,declared_electrodes,T], with one to 100 complete 200-sample patches.'
            )
        return x.reshape(x.shape[0], self.num_electrodes, -1, 200)

    def forward(self, x, return_tokens=False):
        r"""Return native token grids [B,C,P,384] or their mean [B,384].

        Mean reduction is an explicit transfer feature view. The original
        author classifier instead flattens all tokens into its new task head;
        use return_tokens=True when implementing that fixed-window protocol.
        """
        if not isinstance(return_tokens, bool):
            raise ValueError('return_tokens must be boolean.')
        tokens = self.backend.forward_features(self._patches(x))
        return tokens if return_tokens else tokens.mean(dim=(1, 2))

    def forward_pretraining(self, x):
        r"""Return all three complete native pretraining outputs.

        time_prediction [B,C,P,200], frequency_prediction [B,C,P,429] and
        contrastive_projection [B,256] retain original trainable heads.
        Supply explicitly augmented or masked signals as appropriate. This
        call does not invent augmentations, reconstruction targets or losses.
        Native pretraining uses squared error despite its loss_mae names.
        """
        time, frequency, contrastive = self.backend(self._patches(x), mode='full')
        return dict(
            time_prediction=time,
            frequency_prediction=frequency,
            contrastive_projection=contrastive,
        )

    def forward_contrastive(self, x):
        r"""Return the complete original contrastive projection [B,256]."""
        return self.backend(self._patches(x), mode='contrastive')

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        backend,
        channel_names,
        sampling_rate=200,
        prefix='',
        state_key='model',
    ):
        r"""Strictly load all native tensors from an explicitly supplied file.

        Uses restricted CPU tensor loading with no source import, download,
        unsafe fallback, decoder removal or partial loading. The release
        includes 144 profiling buffers in addition to 242 neural tensors;
        construct its matching documented backend before loading. No buffer
        is silently discarded. Flat exports require state_key=None.
        """
        model = cls(backend, channel_names, sampling_rate)
        if not isinstance(prefix, str) or (
            state_key is not None and not isinstance(state_key, str)
        ):
            raise ValueError(
                'Require an explicit string prefix and optional state key.'
            )
        path = Path(checkpoint_path)
        if path.suffix == '.safetensors':
            from safetensors.torch import load_file

            state = load_file(str(path), device='cpu')
        else:
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if state_key is not None:
            if not isinstance(state, dict) or state_key not in state:
                raise ValueError('The explicitly declared native state key is absent.')
            state = state[state_key]
        if not isinstance(state, dict) or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError('Require a complete native tensor state dictionary.')
        if prefix and any(not key.startswith(prefix) for key in state):
            raise ValueError('Every native tensor must use the explicit common prefix.')
        state = {key[len(prefix) :]: value for key, value in state.items()}
        expected = backend.state_dict()
        if set(state) != set(expected):
            raise ValueError(
                'Require every native tensor, including all three heads and supplied profiling buffers.'
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


class UniDMFMClassifier(nn.Module):
    r"""A new mean-token classification head over complete native features.

    Args:
        backbone (UniDMFM): Explicit complete native-backend adapter.
        num_classes (int): New task logit width. (default: :obj:`2`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
    """

    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        _positive_int('num_classes', num_classes)
        if not isinstance(backbone, UniDMFM) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require a UniDMFM adapter and boolean freeze_backbone.')
        self.backbone, self.freeze_backbone = backbone, freeze_backbone
        self.classifier = nn.Linear(384, num_classes).to(
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
