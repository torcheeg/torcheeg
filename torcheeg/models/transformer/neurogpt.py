"""Interfaces for an explicitly supplied complete NeuroGPT author backend.

Author code and weights are GPL-3.0 materials supplied separately. They are
not bundled, downloaded or imported by this library interface.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from ._utils import _positive_int


class NeuroGPT(nn.Module):
    r"""Complete external NeuroGPT with explicit sensor and segment metadata.

    Input is floating [batch,segments,22,500] at 250 Hz. Every segment is a
    complete two-second EEG window. Ordered channel names are required;
    no global electrode vocabulary, geometry, padding or interpolation is
    inferred. The caller defines segment onsets, overlap and preprocessing.
    The native encoder, embedder, GPT, unembedder and unused release states
    remain present.

    Args:
        backend (nn.Module): Explicit complete author Model instance.
        channel_names (list or tuple): Exactly 22 ordered physical names.
        sampling_rate (int): Physical sampling rate, must be 250 Hz.
        allow_montage_transfer (bool): Explicitly permit a different ordered
            physical montage. (default: :obj:`False`)
    """

    PRETRAINING_CHANNELS = (
        'FP1',
        'FP2',
        'F7',
        'F3',
        'FZ',
        'F4',
        'F8',
        'T1',
        'T3',
        'C3',
        'CZ',
        'C4',
        'T4',
        'T2',
        'T5',
        'P3',
        'PZ',
        'P4',
        'T6',
        'O1',
        'OZ',
        'O2',
    )

    def __init__(
        self, backend, channel_names, sampling_rate=250, allow_montage_transfer=False
    ):
        super().__init__()
        _positive_int('sampling_rate', sampling_rate)
        if sampling_rate != 250:
            raise ValueError('The documented NeuroGPT protocol requires 250 Hz.')
        if (
            not isinstance(channel_names, (tuple, list))
            or len(channel_names) != 22
            or not all(
                isinstance(v, str) and v and v.strip() == v for v in channel_names
            )
            or len({v.casefold() for v in channel_names}) != 22
        ):
            raise ValueError(
                'Require exactly 22 distinct ordered physical channel names.'
            )
        if not isinstance(allow_montage_transfer, bool):
            raise ValueError('allow_montage_transfer must be boolean.')
        native_montage = tuple(name.casefold() for name in channel_names) == tuple(
            name.casefold() for name in self.PRETRAINING_CHANNELS
        )
        if not native_montage and not allow_montage_transfer:
            raise ValueError(
                'Use the exact pretrained physical channel order or explicitly set allow_montage_transfer=True for a different montage.'
            )
        self.allow_montage_transfer = allow_montage_transfer
        self.is_pretraining_montage = native_montage
        if not isinstance(backend, nn.Module):
            raise ValueError(
                'Require an explicitly constructed complete nn.Module backend.'
            )
        for name in ['encoder', 'embedder', 'decoder', 'unembedder']:
            if not isinstance(getattr(backend, name, None), nn.Module):
                raise ValueError('Keep the complete native ' + name + '.')
        encoder, embedder, decoder = backend.encoder, backend.embedder, backend.decoder
        transformer = getattr(decoder, 'transformer', None)
        config = getattr(transformer, 'config', None)
        expected = dict(
            n_embd=1024,
            n_layer=6,
            n_head=16,
            n_inner=4096,
            n_positions=512,
            activation_function='gelu_new',
        )
        if (
            config is None
            or any(
                getattr(config, key, None) != value for key, value in expected.items()
            )
            or getattr(config, '_attn_implementation', 'eager') != 'eager'
            or getattr(transformer, '_attn_implementation', 'eager') != 'eager'
            or getattr(encoder, 'n_chans', None) != 22
            or getattr(encoder, 'n_times', None) != 500
            or len(getattr(encoder, 'transformer', [])) != 6
            or getattr(embedder, 'in_dim', None) != 1080
            or getattr(embedder, 'embed_dim', None) != 1024
            or not callable(getattr(embedder, 'duplicate_batch', None))
            or not callable(getattr(backend, 'compute_loss', None))
        ):
            raise ValueError(
                'Require the complete released six-layer Conformer and six-layer/1024-wide/16-head causal GPT model with eager attention.'
            )
        encoder_attention = [
            layer for layer in encoder.modules() if hasattr(layer, 'num_heads')
        ]
        if len(encoder_attention) != 6 or any(
            layer.num_heads != 10 or getattr(layer, 'emb_size', None) != 40
            for layer in encoder_attention
        ):
            raise ValueError(
                'Keep all six original width-40/ten-head Conformer attention layers.'
            )
        state = backend.state_dict()
        shapes = {
            'encoder.patch_embedding.shallownet.0.weight': (40, 1, 1, 25),
            'encoder.patch_embedding.shallownet.1.weight': (40, 40, 22, 1),
            'embedder.msk_embed': (1, 1, 1080),
            'embedder.cls_embed': (1, 1, 1080),
            'embedder.embed_model.model.0.weight': (1024, 1080),
            'decoder.transformer.wte.weight': (1, 1024),
            'decoder.transformer.wpe.weight': (512, 1024),
            'decoder.pooler_layer.0.weight': (1024, 1024),
            'unembedder.model.0.weight': (1080, 1024),
        }
        if any(
            key not in state or tuple(state[key].shape) != shape
            for key, shape in shapes.items()
        ):
            raise ValueError('Keep all complete native components and release states.')
        self.backend = backend
        self._check_native_mode()
        self.channel_names = tuple(channel_names)
        (
            self.num_electrodes,
            self.sampling_rate,
            self.segment_size,
            self.hid_channels,
        ) = 22, 250, 500, 1024
        behavior = {
            'gpt': {key: getattr(config, key) for key in expected},
            'gpt_dropout': [
                getattr(config, key, None)
                for key in ['resid_pdrop', 'attn_pdrop', 'embd_pdrop']
            ],
            'attention': 'eager',
            'prefix_method': 'native_csm_causal',
            'dropouts': [
                (name, layer.p)
                for name, layer in backend.named_modules()
                if isinstance(layer, nn.Dropout)
            ],
        }
        self._digest = hashlib.sha256(
            json.dumps(
                [
                    self.channel_names,
                    self.PRETRAINING_CHANNELS,
                    allow_montage_transfer,
                    250,
                    500,
                    behavior,
                ]
            ).encode()
        ).digest()
        self.register_buffer(
            'metadata_fingerprint', torch.tensor(list(self._digest), dtype=torch.uint8)
        )

    def _check_native_mode(self):
        if (
            getattr(self.backend, 'is_decoding_mode', None) is not False
            or getattr(self.backend, 'ft_only_encoder', None) is not False
            or getattr(self.backend.encoder, 'is_decoding_mode', None) is not False
            or getattr(self.backend.decoder, 'is_decoding_mode', None) is not False
            or getattr(self.backend.embedder, 'training_style', None) != 'CSM'
        ):
            raise ValueError(
                'Keep the original complete pretraining configuration; do not switch native decoding heads or encoder-only mode.'
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
                'NeuroGPT channel order, physical metadata or native configuration differs.'
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

    def _inputs(self, x):
        self._check_native_mode()
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 4
            or not x.is_floating_point()
            or x.shape[0] < 1
            or not 1 <= x.shape[1] <= 512
            or x.shape[2:] != (22, 500)
            or not torch.isfinite(x).all()
        ):
            raise ValueError(
                'Require finite floating EEG [batch,1..512_segments,22,500] at 250 Hz, without padded segments or channels.'
            )
        return {
            'inputs': x,
            'attention_mask': torch.ones(
                x.shape[:2], dtype=torch.long, device=x.device
            ),
        }

    def forward_encoder(self, x):
        r"""Return native within-segment Conformer features [B,S,27,40]."""
        self._inputs(x)
        encoded = self.backend.encoder(x)
        return encoded.reshape(x.shape[0], x.shape[1], 27, 40)

    def forward(self, x, return_tokens=False):
        r"""Return full native unmasked causal GPT tokens or final features.

        Tokens are [B,S,1024]; default output is the final segment [B,1024].
        This explicit feature view uses the original encoder, embedder and
        decoder without causal reconstruction masking or new CLS/pooler
        activations. The released CLS, vocabulary and pooler states remain
        loaded, although the native pretraining loss does not use them.
        """
        if not isinstance(return_tokens, bool):
            raise ValueError('return_tokens must be boolean.')
        batch = self._inputs(x)
        encoded = self.backend.encoder(x).reshape(x.shape[0], x.shape[1], 1080)
        batch['inputs_embeds'] = self.backend.embedder({'inputs': encoded})
        outputs = self.backend.decoder(batch)['outputs']
        return outputs if return_tokens else outputs[:, -1]

    def forward_reconstruction(self, x):
        r"""Run complete native causal prefix reconstruction and its loss.

        At least two segments are required. The author method generates
        every next-segment prefix: B*(S-1) reconstruction rows, each with
        S encoded positions and 1080 predicted values. Targets are detached
        native Conformer features, not raw EEG amplitudes. Return keys are
        prediction, reconstruction_loss and loss. Backpropagate loss once;
        reconstruction_loss names the same component, not an extra loss.
        Prefix expansion can require substantial memory for long sequences.
        """
        batch = self._inputs(x)
        if x.shape[1] < 2:
            raise ValueError(
                'Native causal reconstruction needs at least two complete segments.'
            )
        losses, outputs = self.backend.compute_loss(batch, return_outputs=True)
        return dict(
            prediction=outputs['outputs'],
            reconstruction_loss=losses['masking_loss'],
            loss=losses['loss'],
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        backend,
        channel_names,
        sampling_rate=250,
        prefix='',
        state_key=None,
        allow_montage_transfer=False,
    ):
        r"""Strictly load every native tensor from an explicit local file.

        Official weights are a flat dictionary. Reads restricted tensors on
        CPU without source import, download, unsafe pickle fallback, partial
        matching, encoder-only selection or head removal. Complete nested
        exports can specify an explicit state_key and common prefix.
        """
        model = cls(
            backend,
            channel_names,
            sampling_rate,
            allow_montage_transfer=allow_montage_transfer,
        )
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
                'Require every native encoder, embedder, GPT, pooler and unembedder state.'
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


class NeuroGPTClassifier(nn.Module):
    r"""A new task head over the complete causal GPT's final segment feature.

    Args:
        backbone (NeuroGPT): Explicit complete native-backend adapter.
        num_classes (int): New task logit width. (default: :obj:`4`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
    """

    def __init__(self, backbone, num_classes=4, freeze_backbone=False):
        super().__init__()
        _positive_int('num_classes', num_classes)
        if not isinstance(backbone, NeuroGPT) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require a NeuroGPT adapter and boolean freeze_backbone.')
        self.backbone, self.freeze_backbone = backbone, freeze_backbone
        self.classifier = nn.Linear(1024, num_classes).to(
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
