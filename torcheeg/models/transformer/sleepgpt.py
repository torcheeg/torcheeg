"""Interfaces for an explicitly supplied complete native SleepGPT backend.

No author code or weights are bundled. The pinned source supplies no license;
the separately obtained Figshare checkpoint specifies CC-BY-4.0.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from ._utils import _positive_int


class SleepGPT(nn.Module):
    r"""Complete released SleepGPT time-frequency model with explicit sensors.

    Input is floating [batch,electrodes,3000] at 100 Hz. Ordered names must
    use the native vocabulary C3, C4, EMG, EOG, F3, Fpz, O1, Pz. Unknown or
    bipolar names are not silently mapped. Missing native sensors are zero
    filled and masked. Filtering, referencing and amplitude scaling are
    caller responsibilities. Both native reconstruction heads remain loaded.

    Args:
        backend (nn.Module): Explicit complete author Model_Pre backend with
            the documented parameter-free Identity pooler feature view.
        channel_names (list or tuple): Ordered physical names matching input.
        sampling_rate (int): Physical sampling rate, must be 100 Hz.
    """

    CHANNEL_VOCAB = ('C3', 'C4', 'EMG', 'EOG', 'F3', 'Fpz', 'O1', 'Pz')

    def __init__(self, backend, channel_names, sampling_rate=100):
        super().__init__()
        _positive_int('sampling_rate', sampling_rate)
        if sampling_rate != 100:
            raise ValueError('The released SleepGPT model requires 100 Hz signals.')
        if not isinstance(backend, nn.Module):
            raise ValueError(
                'Require an explicitly constructed complete native nn.Module.'
            )
        if (
            not isinstance(channel_names, (tuple, list))
            or not channel_names
            or not all(isinstance(v, str) and v for v in channel_names)
        ):
            raise ValueError(
                'Require explicit nonempty ordered physical channel names.'
            )
        names = tuple(name.casefold() for name in channel_names)
        vocab = tuple(name.casefold() for name in self.CHANNEL_VOCAB)
        if len(set(names)) != len(names) or any(name not in vocab for name in names):
            raise ValueError(
                'Channels must be distinct members of the native sensor vocabulary.'
            )
        transformer = getattr(backend, 'transformer', None)
        expected = dict(
            embed_dim=768,
            num_heads=16,
            tfffn_start_layer_index=10,
            max_channels=8,
            num_patches=15,
            patch_size=200,
            fs=100,
            epoch_duration=30,
        )
        if (
            transformer is None
            or any(
                getattr(transformer, key, None) != value
                for key, value in expected.items()
            )
            or len(getattr(transformer, 'blocks', [])) != 12
        ):
            raise ValueError(
                'Require the complete released 12-layer/768-wide/16-head time-frequency model.'
            )
        config = getattr(getattr(backend, 'hparams', None), 'config', None)
        if (
            not isinstance(config, dict)
            or config.get('use_cb') is not False
            or config.get('time_only') is not False
            or config.get('fft_only') is not False
        ):
            raise ValueError(
                'Require the original dual-domain configuration with use_cb=False.'
            )
        if getattr(transformer, 'actual_channels', None) is not None:
            raise ValueError('Keep the full native eight-channel transformer layout.')
        if not isinstance(getattr(backend, 'pooler', None), nn.Identity):
            raise ValueError(
                'Use the documented parameter-free Identity view for the absent pretrained pooler.'
            )
        for name in [
            'infer',
            'get_attention_mask',
            'forward_masked_loss',
            'forward_masked_loss_2D',
        ]:
            if not callable(getattr(backend, name, None)):
                raise ValueError(
                    'The backend lacks a complete native pretraining method: ' + name
                )
        for name in ['Masked_docoder', 'Masked_docoder_fft']:
            if not isinstance(getattr(backend, name, None), nn.Module):
                raise ValueError('Keep both complete native reconstruction heads.')
        self.backend = backend
        self.channel_indices = tuple(vocab.index(name) for name in names)
        self.channel_names = tuple(self.CHANNEL_VOCAB[i] for i in self.channel_indices)
        self.num_electrodes, self.sampling_rate, self.chunk_size, self.hid_channels = (
            len(names),
            100,
            3000,
            1536,
        )
        behavior = {
            name: config.get(name)
            for name in [
                'use_cb',
                'time_only',
                'fft_only',
                'mask_ratio',
                'drop_path_rate',
                'loss_function',
            ]
        }
        self._digest = hashlib.sha256(
            json.dumps(
                [self.channel_names, self.CHANNEL_VOCAB, 100, 3000, expected, behavior]
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
                'SleepGPT sensor order, physical metadata or native configuration differs.'
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

    @staticmethod
    @torch.no_grad()
    def spectrogram(x):
        r"""Return detached native-standardized spectra [B,C,30,100].

        Uses a 256-point STFT, periodic Hann length 200, hop 100, reflection
        centering, the first 100 bins and all frames after the first. Values
        are 10*log10(magnitude+1e-8), standardized per frame across frequency
        with sample standard deviation. Zero-variance frames become zero.
        Complex STFT output keeps this preprocessing on the current PyTorch
        API. As in the author training forward, this branch is detached from
        raw-signal gradients; encoder/decoder parameter gradients still flow.
        """
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 3
            or not x.is_floating_point()
            or min(x.shape[:2]) < 1
            or x.shape[2] != 3000
        ):
            raise ValueError('Require floating signals[B,C,3000] at100Hz.')
        batch, channels, _ = x.shape
        window = torch.hann_window(200, device=x.device)
        spectrum = torch.stft(
            x.reshape(batch * channels, 3000),
            n_fft=256,
            hop_length=100,
            win_length=200,
            window=window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        parts = torch.view_as_real(spectrum)
        power = parts[..., 0].square() + parts[..., 1].square()
        values = (10 * torch.log10(power.sqrt()[:, :100, 1:] + 1e-8)).transpose(-2, -1)
        mean, spread = values.mean(-1, keepdim=True), values.std(-1, keepdim=True)
        nonzero = spread != 0
        normalized = (values - mean) / torch.where(
            nonzero, spread, torch.ones_like(spread)
        )
        return torch.where(nonzero, normalized, torch.zeros_like(normalized)).reshape(
            batch, channels, 30, 100
        )

    def _inputs(self, x, padding_mask):
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 3
            or not x.is_floating_point()
            or x.shape != (x.shape[0], self.num_electrodes, 3000)
            or x.shape[0] < 1
        ):
            raise ValueError(
                'Require nonempty floating EEG[B,declared_electrodes,3000].'
            )
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool)
        if (
            not isinstance(padding_mask, torch.Tensor)
            or padding_mask.dtype != torch.bool
            or padding_mask.shape != x.shape[:2]
            or padding_mask.device != x.device
        ):
            raise ValueError('Require a same-device boolean channel padding mask[B,C].')
        if padding_mask.all(1).any():
            raise ValueError(
                'Each example must have at least one available physical channel.'
            )
        signals = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        if not torch.isfinite(signals).all():
            raise ValueError('Available input signals must be finite.')
        index = torch.tensor(self.channel_indices, device=x.device, dtype=torch.long)
        mapped = x.new_zeros(x.shape[0], 8, 3000).index_copy(1, index, signals)
        present = torch.zeros(
            x.shape[0], 8, device=x.device, dtype=torch.bool
        ).index_copy(1, index, ~padding_mask)
        spectral = self.spectrogram(mapped)
        valid = present.to(dtype=x.dtype)
        batch = dict(
            epochs=(mapped, spectral),
            mask=self.backend.get_attention_mask(valid, valid),
        )
        return batch, present, index

    def forward(self, x, padding_mask=None):
        r"""Return both native class tokens concatenated into features[B,1536]."""
        batch, _, _ = self._inputs(x, padding_mask)
        result = self.backend.infer(batch, time_mask=False, stage='test')
        return torch.cat(
            [result['time_feats'][:, 0], result['fft_feats'][:, 0]], dim=-1
        )

    def forward_tokens(self, x, padding_mask=None):
        r"""Return both contextual token grids and canonical channel padding.

        time_tokens/frequency_tokens are[B,8,15,768] in CHANNEL_VOCAB order.
        Padded rows still have contextual activations and must be excluded
        from downstream token losses. padding_mask is boolean[B,8].
        """
        batch, present, _ = self._inputs(x, padding_mask)
        result = self.backend.infer(batch, time_mask=False, stage='test')
        return dict(
            time_tokens=result['time_feats'][:, 1:].reshape(x.shape[0], 8, 15, 768),
            frequency_tokens=result['fft_feats'][:, 1:].reshape(x.shape[0], 8, 15, 768),
            padding_mask=~present,
        )

    def forward_masked(self, x, mask, padding_mask=None):
        r"""Run both complete native reconstruction heads and masked losses.

        mask is boolean[B,input_electrodes,15], True for time patches to
        reconstruct. It may not select absent sensors and must select at
        least one patch per example. Frequency masks follow the original
        native stochastic sampler and use the normal Torch RNG. Returned
        predictions are[B,120,200]; masks are[B,8,15] in native sensor order.
        The loss uses original normalized frequency targets and supplied
        time amplitudes, preserving the native target-gradient convention.
        """
        batch, present, index = self._inputs(x, padding_mask)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != (x.shape[0], self.num_electrodes, 15)
            or mask.device != x.device
        ):
            raise ValueError(
                'Require same-device boolean time mask[B,input_electrodes,15].'
            )
        native_mask = torch.zeros(
            x.shape[0], 8, 15, device=x.device, dtype=torch.bool
        ).index_copy(1, index, mask)
        if (native_mask & ~present.unsqueeze(-1)).any() or not native_mask.flatten(
            1
        ).any(1).all():
            raise ValueError(
                'Select at least one available time patch per example, never absent sensors.'
            )
        batch['random_mask'] = [native_mask.flatten(1).to(dtype=x.dtype)]
        result = self.backend.infer(batch, time_mask=True, stage='train')
        time_mask, frequency_mask = result['time_mask_patch'], result['fft_mask_patch']
        if not frequency_mask.sum() > 0:
            raise ValueError(
                'Native random frequency mask selected no available patches; resample it.'
            )
        time_loss = self.backend.forward_masked_loss(
            result['cls_feats'], batch['epochs'][0], time_mask
        )
        frequency_loss = self.backend.forward_masked_loss_2D(
            result['cls_feats_fft'], batch['epochs'][1], frequency_mask
        )
        return dict(
            time_prediction=result['cls_feats'],
            frequency_prediction=result['cls_feats_fft'],
            time_mask=time_mask.reshape(x.shape[0], 8, 15),
            frequency_mask=frequency_mask.reshape(x.shape[0], 8, 15),
            time_loss=time_loss,
            frequency_loss=frequency_loss,
            loss=time_loss + frequency_loss,
            padding_mask=~present,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        backend,
        channel_names,
        sampling_rate=100,
        prefix='',
        state_key='state_dict',
    ):
        r"""Strictly load every native tensor from an explicit local checkpoint.

        Reads restricted tensors on CPU, with no source import, download,
        unsafe pickle fallback, partial matching or decoder removal. The
        official Lightning file uses state_key='state_dict'. Complete flat
        exports and safetensors use state_key=None. Backend construction is
        an explicit caller action, separate from checkpoint loading.
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
        if prefix and any(not k.startswith(prefix) for k in state):
            raise ValueError('Every tensor must use the explicit common prefix.')
        state = {k[len(prefix) :]: v for k, v in state.items()}
        expected = backend.state_dict()
        if set(state) != set(expected):
            raise ValueError(
                'Require the entire native model, including both reconstruction heads.'
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


class SleepGPTClassifier(nn.Module):
    r"""A new classification head over both complete native class features.

    Args:
        backbone (SleepGPT): Explicit complete native-backend adapter.
        num_classes (int): New task logit width. (default: :obj:`2`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
    """

    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        _positive_int('num_classes', num_classes)
        if not isinstance(backbone, SleepGPT) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require a SleepGPT adapter and boolean freeze_backbone.')
        self.backbone, self.freeze_backbone = backbone, freeze_backbone
        self.classifier = nn.Linear(1536, num_classes).to(
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
