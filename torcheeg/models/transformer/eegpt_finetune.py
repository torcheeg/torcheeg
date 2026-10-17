# Adapted from BINE022/EEGPT under Apache-2.0; see LICENSES/EEGPT.txt.
from collections.abc import Mapping
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .eegpt import EEGPT, _positive_int
from .eegpt_decoder import EEGPTPredictor, EEGPTReconstructor
from .eegpt_heads import _EEGPTDownstream, _LinearWithConstraint, _validate_task_parameters


class _Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, in_channels, out_channels, max_norm):
        super().__init__(in_channels, out_channels, kernel_size=1)
        self.max_norm = max_norm

    def forward(self, x):
        weight = self.weight
        if self.max_norm is not None:
            if weight.requires_grad:
                with torch.no_grad():
                    weight.copy_(torch.renorm(weight, p=2, dim=0, maxnorm=self.max_norm))
                weight = weight.clone()
            else:
                weight = torch.renorm(weight, p=2, dim=0, maxnorm=self.max_norm)
        return F.conv1d(x, weight, self.bias)


def _component_state(state, prefix, component):
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    expected = component.state_dict()
    missing, extra = set(expected) - set(selected), set(selected) - set(expected)
    if missing or extra:
        raise ValueError(f'Incomplete component {prefix}: missing={sorted(missing)}, extra={sorted(extra)}.')
    for name, value in selected.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.shape != expected[name].shape:
            raise ValueError(f'Checkpoint tensor shape/type mismatch: {prefix}{name}.')
        if not torch.isfinite(value).all():
            raise ValueError(f'Checkpoint tensor contains nonfinite values: {prefix}{name}.')
    return selected


class EEGPTFinetuneClassifier(_EEGPTDownstream):
    r'''Classify EEG with an EEGPT encoder and its temporal decoder.

    This is the official full-token downstream path: spatial summaries,
    reconstructor or predictor, patch-token mean pooling or a class token,
    LayerNorm, and a constrained linear head. Input is [batch, channels,
    samples], or [batch, channels, windows, samples_per_window] flattened in
    time order. Output contains [batch, num_classes] unnormalized logits.
    num_classes=0 returns pooled features through an identity head.

    Optional pointwise channel mixing and nearest-neighbor length
    normalization reproduce the corresponding official task-model choices.
    They must be selected explicitly. Learned channel slots do not recover
    physical electrodes. Length normalization does not infer a sampling rate
    or preserve a physical time interval; set the data protocol separately.
    For band-limited resampling, use a preprocessing transform instead.

    The default constructor trains both supplied components. Set
    freeze_encoder/freeze_decoder to freeze them and keep their dropout in
    evaluation mode. Input gradients remain available. train_cls_token=True
    additionally trains a frozen decoder's class token, matching the official
    optional decoder-freeze behavior. Otherwise existing parameter flags are
    preserved. Frozen components do not gain a no-grad region.

    .. code-block:: python

        import torch
        from torcheeg.models import EEGPT, EEGPTReconstructor, EEGPTFinetuneClassifier

        encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'],
                        hid_channels=32, depth=2, num_heads=4, embed_num=2)
        decoder = EEGPTReconstructor(num_patches=encoder.num_patches,
            in_features=32, embed_num=2, hid_channels=32, depth=2, num_heads=4)
        model = EEGPTFinetuneClassifier(encoder, decoder, num_classes=2)
        logits = model(torch.randn(2, 3, 1024))

    This example uses random weights. The module implements no masked
    pretraining loss, checkpoint download or benchmark training schedule.
    Direct EEGPTClassifier remains a separate summary-only architecture.

    .. automethod:: feature
    .. automethod:: forward_features
    .. automethod:: from_pretrained

    Args:
        encoder (EEGPT): Configured summary encoder with ordered channel_names.
        decoder (EEGPTReconstructor or EEGPTPredictor): Matching temporal decoder, on the same device and dtype as encoder.
        num_classes (int): Output classes; zero returns pooled features. (default: :obj:`2`)
        mean_pooling (bool): Pool non-class tokens; otherwise select the class token. (default: :obj:`True`)
        use_out_proj (bool): Apply decoder normalization and output projection. (default: :obj:`False`)
        norm_eps (float): Final pooling LayerNorm epsilon. (default: :obj:`1e-5`)
        max_norm (float, optional): Per-class weight norm limit; None disables it. (default: :obj:`1.0`)
        freeze_encoder (bool): Freeze encoder and keep it in evaluation mode. (default: :obj:`False`)
        freeze_decoder (bool): Freeze decoder and keep it in evaluation mode. (default: :obj:`False`)
        train_cls_token (bool): Train the frozen decoder class token; requires freeze_decoder. (default: :obj:`False`)
        in_channels (int, optional): When supplied, learn a pointwise mapping from this input channel count into the encoder's configured slots. None requires the encoder's channel count directly. (default: :obj:`None`)
        channel_max_norm (float, optional): Per-output channel-mixing weight norm limit; None disables it. (default: :obj:`1.0`)
        resize_input (bool): Normalize sample count to encoder.chunk_size with nearest-neighbor interpolation. (default: :obj:`False`)
    '''
    def __init__(self, encoder, decoder, num_classes=2, mean_pooling=True, use_out_proj=False,
                 norm_eps=1e-5, max_norm=1., freeze_encoder=False, freeze_decoder=False,
                 train_cls_token=False, in_channels=None, channel_max_norm=1., resize_input=False):
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 0:
            raise ValueError('num_classes must be a nonnegative integer.')
        _validate_task_parameters(1, 0., max_norm)
        _validate_task_parameters(1, 0., channel_max_norm)
        if isinstance(norm_eps, bool) or not isinstance(norm_eps, (int, float)) or not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError('norm_eps must be finite and positive.')
        for value in (mean_pooling, use_out_proj, freeze_decoder, train_cls_token, resize_input):
            if not isinstance(value, bool):
                raise ValueError('Pooling, projection, freezing, token-training and resizing settings must be booleans.')
        if train_cls_token and not freeze_decoder:
            raise ValueError('train_cls_token requires freeze_decoder=True.')
        if in_channels is not None:
            _positive_int('in_channels', in_channels)
        if not isinstance(encoder, EEGPT) or not isinstance(decoder, (EEGPTReconstructor, EEGPTPredictor)):
            raise ValueError('Supply an EEGPT encoder and an EEGPT temporal decoder.')
        if (decoder.num_patches != encoder.num_patches or decoder.embed_num != encoder.embed_num or
                decoder.in_features != encoder.hid_channels or decoder.return_attention_layer != -1):
            raise ValueError('Decoder patch count, summary count and input width must match the encoder, without attention inspection.')
        if decoder.cls_token.device != encoder.summary_token.device or decoder.cls_token.dtype != encoder.summary_token.dtype:
            raise ValueError('Encoder and decoder must have the same device and dtype.')
        super().__init__(encoder, freeze_encoder)
        self.decoder = decoder
        self.num_classes = num_classes
        self.use_out_proj = use_out_proj
        self.freeze_decoder = freeze_decoder
        self.resize_input = resize_input
        self.input_channels = encoder.num_electrodes if in_channels is None else in_channels
        self.decoder_feature_dim = decoder.output_features if use_out_proj else decoder.hid_channels
        if in_channels is not None:
            self.chan_conv = nn.Sequential(_Conv1dWithConstraint(in_channels, encoder.num_electrodes, channel_max_norm)).to(encoder.summary_token)
        else:
            self.chan_conv = nn.Identity()
        self.norm = nn.Identity() if mean_pooling else nn.LayerNorm(self.decoder_feature_dim, eps=norm_eps).to(encoder.summary_token)
        self.fc_norm = nn.LayerNorm(self.decoder_feature_dim, eps=norm_eps).to(encoder.summary_token) if mean_pooling else None
        self.head = _LinearWithConstraint(self.decoder_feature_dim, num_classes, max_norm).to(encoder.summary_token) if num_classes else nn.Identity()
        if freeze_decoder:
            self.decoder.requires_grad_(False)
            self.decoder.eval()
            if train_cls_token:
                self.decoder.cls_token.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_decoder:
            self.decoder.eval()
        return self

    def _prepare_input(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim not in (3, 4) or x.shape[1] != self.input_channels:
            raise ValueError('Input must be [batch, input_channels, samples] or its four-dimensional temporal partition.')
        if x.ndim == 4:
            x = x.flatten(2)
        if x.shape[-1] < 1:
            raise ValueError('Input must contain at least one sample.')
        if self.resize_input and x.shape[-1] != self.encoder.chunk_size:
            x = F.interpolate(x, size=self.encoder.chunk_size, mode='nearest')
        return self.chan_conv(x)

    def feature(self, x, flatten=False):
        r'''Extract raw spatial encoder summaries before the decoder.

        Explicit input mixing/resizing settings apply here too. As with the
        direct EEGPT task heads, flatten=True returns [batch, feature_dim];
        otherwise return [batch, patches, summaries, encoder_width]. For
        decoded/pool features use forward_features instead.
        '''
        if not isinstance(flatten, bool):
            raise ValueError('flatten must be a boolean.')
        output = self.encoder(self._prepare_input(x))
        return output.flatten(1) if flatten else output

    def forward_features(self, x, return_patch_tokens=False, return_all_tokens=False):
        r'''Return pooled decoder features or normalized decoder tokens.

        Default output is [batch, decoder_feature_dim]. With mean pooling,
        exclude the class token, average patch tokens, then apply LayerNorm.
        Otherwise normalize all tokens and select the class token. Optional
        token returns have the same final normalization, applied per token.
        Request at most one of the two token-return modes.
        '''
        if not isinstance(return_patch_tokens, bool) or not isinstance(return_all_tokens, bool):
            raise ValueError('Token-return settings must be booleans.')
        if return_patch_tokens and return_all_tokens:
            raise ValueError('Select patch tokens or all tokens, not both.')
        tokens = self.norm(self.decoder(self.feature(x), use_out_proj=self.use_out_proj))
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(tokens)
            patches = tokens[:, 1:]
            return self.fc_norm(patches) if return_patch_tokens else self.fc_norm(patches.mean(1))
        if return_all_tokens:
            return tokens
        return tokens[:, 1:] if return_patch_tokens else tokens[:, 0]

    def forward(self, x):
        return self.head(self.forward_features(x))

    @classmethod
    def from_pretrained(cls, checkpoint_path, encoder_kwargs, decoder_kwargs,
                        decoder_type='reconstructor', encoder_prefix='target_encoder.', decoder_prefix=None, **kwargs):
        r'''Load complete encoder and decoder states from one local checkpoint.

        encoder_kwargs and decoder_kwargs are explicit component constructor
        dictionaries, including ordered encoder channel_names and geometry.
        decoder_type selects 'reconstructor' or 'predictor'. decoder_prefix
        defaults to that component name plus a dot. encoder_prefix defaults to
        'target_encoder.'. Supply nonempty, non-overlapping explicit prefixes.

        Read tensors once with weights_only=True, map to CPU, and validate all
        component keys, shapes, floating dtypes and finite values before strict
        loading. Unrelated checkpoint components are ignored; missing or extra
        tensors inside either selected prefix are errors. Unsupported pickle
        objects are rejected without an unsafe fallback. The new task pooling,
        channel mixer and head retain random initialization. This is not a
        full trained-classifier loader; use strict load_state_dict for a full
        task tensor state and the same explicit constructor configuration.
        '''
        if not isinstance(encoder_kwargs, dict) or not isinstance(decoder_kwargs, dict):
            raise ValueError('Component constructor settings must be dictionaries.')
        if decoder_type not in ('reconstructor', 'predictor'):
            raise ValueError("decoder_type must be 'reconstructor' or 'predictor'.")
        if decoder_prefix is None:
            decoder_prefix = decoder_type + '.'
        if (not isinstance(encoder_prefix, str) or not isinstance(decoder_prefix, str) or
                not encoder_prefix or not decoder_prefix or encoder_prefix.startswith(decoder_prefix) or decoder_prefix.startswith(encoder_prefix)):
            raise ValueError('Component prefixes must be nonempty, distinct and non-overlapping.')
        checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError('Checkpoint must contain a tensor state dictionary.')
        state = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ValueError('Checkpoint state keys must be strings.')
        encoder = EEGPT(**encoder_kwargs)
        kind = EEGPTReconstructor if decoder_type == 'reconstructor' else EEGPTPredictor
        decoder = kind(**decoder_kwargs)
        encoder_state = _component_state(state, encoder_prefix, encoder)
        decoder_state = _component_state(state, decoder_prefix, decoder)
        encoder.load_state_dict(encoder_state, strict=True)
        decoder.load_state_dict(decoder_state, strict=True)
        return cls(encoder, decoder, **kwargs)
