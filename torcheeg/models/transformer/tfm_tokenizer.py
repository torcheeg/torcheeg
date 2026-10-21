"""TFM-Tokenizer/Encoder adapted from Jathurshan Pradeepkumar (MIT).

Source 2d6da482b16dabbb2ebec808fa9f505fc6f367c4; see LICENSES/TFMTokenizer.txt.
Uses the native noncausal linear attention already shared with BIOT.
"""
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.nn.functional as F
from torch import nn

from ._utils import _positive_int
from .biot import _Transformer, _PositionalEncoding


def _load_complete(model, checkpoint_path, prefix):
    if not isinstance(prefix, str):
        raise ValueError('prefix must be a string.')
    path = Path(checkpoint_path)
    if path.suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError('Install torcheeg[foundation] to load safetensors.') from error
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Require a complete flat tensor state dictionary.')
    if prefix and any(not name.startswith(prefix) for name in state):
        raise ValueError('Every state key must start with the explicit prefix.')
    selected = {name[len(prefix):]: value for name, value in state.items()}
    expected = model.state_dict()
    if set(selected) != set(expected):
        raise ValueError(f'Incomplete TFM state: missing={sorted(set(expected)-set(selected))}, '
                         f'extra={sorted(set(selected)-set(expected))}.')
    for name, value in selected.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError(f'Checkpoint tensor shape/type/value mismatch: {name}.')
        if name.endswith('cluster_size') and (value < 0).any():
            raise ValueError('EMA cluster counts must be nonnegative.')
        if name == 'index' and not torch.equal(value, expected[name]):
            raise ValueError('The positional index must preserve every channel row.')
    model.load_state_dict(selected, strict=True)
    return model


class _Encoder(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.transformer = _Transformer(width, 8, depth, .2)

    def forward(self, x):
        return self.transformer(x)


class _EMAQuantizer(nn.Module):
    def __init__(self, count, width, decay, update):
        super().__init__()
        self.count, self.width, self.decay, self.update = count, width, decay, update
        self.embedding = nn.Embedding(count, width)
        nn.init.uniform_(self.embedding.weight, -1/count, 1/count)
        self.embedding.weight.requires_grad_(False)
        self.register_buffer('cluster_size', torch.zeros(count))
        self.register_buffer('ema_w', self.embedding.weight.detach().clone())

    def forward(self, x):
        flat = x.reshape(-1, self.width)
        distances = flat.square().sum(-1, keepdim=True) - 2 * flat @ self.embedding.weight.t()
        distances = distances + self.embedding.weight.square().sum(-1).unsqueeze(0)
        indices = distances.argmin(-1)
        quantized = self.embedding(indices).reshape_as(x)
        if self.training and self.update:
            with torch.no_grad():
                assignments = F.one_hot(indices, self.count).to(flat.dtype)
                counts, totals = assignments.sum(0), assignments.t() @ flat.detach()
                if distributed.is_available() and distributed.is_initialized():
                    distributed.all_reduce(counts)
                    distributed.all_reduce(totals)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1-self.decay)
                self.ema_w.mul_(self.decay).add_(totals, alpha=1-self.decay)
                total = self.cluster_size.sum()
                smoothed = (self.cluster_size + 1e-5) / (total + self.count*1e-5) * total
                self.embedding.weight.copy_(self.ema_w / smoothed[:, None])
        return quantized, indices.reshape(x.shape[:2])


def _convolutions(width, temporal):
    output = width//2 if temporal else width
    first_kernel, first_stride = (200, 100) if temporal else (5, 5)
    return nn.Sequential(nn.Conv1d(1, width, first_kernel, stride=first_stride), nn.GELU(), nn.GroupNorm(width//4, width),
                         nn.Conv1d(width, width, 1), nn.GELU(), nn.GroupNorm(width//4, width),
                         nn.Conv1d(width, output, 1), nn.GELU(), nn.GroupNorm(width//4, output))


class TFMTokenizer(nn.Module):
    r'''Complete single-channel time-frequency motif tokenizer and decoder.

    The released CHB-MIT model has width64, vocabulary8192, frequency/time
    encoders of two blocks each and an eight-block reconstruction decoder.
    It uses the complete released linear reconstruction head. The multi-dataset release and current
    author source use an MLP head; choose that configuration
    explicitly rather than discarding mismatched reconstruction weights.

    Args:
        hid_channels (int): Embedding width, a multiple of eight. (default: :obj:`64`)
        num_tokens (int): Codebook vocabulary. (default: :obj:`8192`)
        encoder_depth (int): Depth of each frequency/time encoder. (default: :obj:`2`)
        decoder_depth (int): Reconstruction transformer depth. (default: :obj:`8`)
        reconstruction_head (str): ``linear`` or ``mlp``. (default: :obj:`linear`)
        decay (float): Codebook EMA factor in [0,1). (default: :obj:`0.99`)
        update_codebook (bool): Update EMA state during training. (default: :obj:`True`)
        beta (float): Nonnegative commitment-loss weight. (default: :obj:`1.0`)
    '''
    def __init__(self, hid_channels=64, num_tokens=8192, encoder_depth=2, decoder_depth=8,
                 reconstruction_head='linear', decay=.99, update_codebook=True, beta=1.):
        super().__init__()
        for name, value in [('hid_channels', hid_channels), ('num_tokens', num_tokens),
                            ('encoder_depth', encoder_depth), ('decoder_depth', decoder_depth)]:
            _positive_int(name, value)
        if hid_channels % 8 or reconstruction_head not in ('linear', 'mlp'):
            raise ValueError('Require width divisible by8 and a linear or mlp reconstruction head.')
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 <= decay < 1:
            raise ValueError('decay must be in [0,1).')
        if not isinstance(update_codebook, bool) or isinstance(beta, bool) or not isinstance(beta, (int, float)) or not 0 <= beta < float('inf'):
            raise ValueError('Require boolean update_codebook and finite nonnegative beta.')
        self.hid_channels, self.num_tokens = hid_channels, num_tokens
        self.encoder_depth, self.decoder_depth = encoder_depth, decoder_depth
        self.reconstruction_head, self.decay, self.update_codebook, self.beta = reconstruction_head, decay, update_codebook, beta
        self.freq_patch_embedding = _convolutions(hid_channels, False)
        self.trans_freq_encoder = _Encoder(hid_channels, encoder_depth)
        self.temporal_patch_embedding = _convolutions(hid_channels, True)
        self.freq_patch_embedding_2_atten = nn.Sequential(nn.Conv1d(hid_channels, hid_channels//8, 5, stride=5), nn.Sigmoid())
        self.freq_patch_embedding_2 = nn.Sequential(nn.Conv1d(hid_channels, hid_channels//8, 5, stride=5))
        self.trans_temporal_encoder = _Encoder(hid_channels, encoder_depth)
        self.quantizer = _EMAQuantizer(num_tokens, hid_channels, decay, update_codebook)
        self.trans_decoder = _Encoder(hid_channels, decoder_depth)
        self.decoder = nn.Linear(hid_channels, 100) if reconstruction_head == 'linear' else nn.Sequential(
            nn.Linear(hid_channels, hid_channels), nn.Tanh(), nn.Linear(hid_channels, 100))

    def tokenize(self, spectrum, temporal):
        r'''Return raw code vectors, integer IDs and normalized encoder vectors.

        Inputs: magnitude spectrum ``[batch,100,frames]`` and the matching
        waveform ``[batch,samples]`` at200Hz. Frames=(samples-200)//100+1.
        Train mode can update EMA state even under no_grad; use eval for a
        frozen tokenizer. No filtering or amplitude normalization is inferred.
        '''
        if not isinstance(spectrum, torch.Tensor) or not isinstance(temporal, torch.Tensor) or spectrum.ndim != 3 or temporal.ndim != 2:
            raise ValueError('Require spectrum[batch,100,frames] and temporal[batch,samples].')
        batch, frequencies, frames = spectrum.shape
        if not spectrum.is_floating_point() or not temporal.is_floating_point() or frequencies != 100 or batch < 1 or frames < 1:
            raise ValueError('Require nonempty floating tensors and exactly100frequency bins.')
        if temporal.shape[0] != batch or temporal.shape[1] < 200 or (temporal.shape[1]-200)//100+1 != frames:
            raise ValueError('Spectrum frames must align with 200-sample/100-stride temporal patches.')
        if spectrum.device != temporal.device or spectrum.dtype != temporal.dtype:
            raise ValueError('Spectrum and temporal tensors must share device/dtype.')
        frequency = self.freq_patch_embedding(spectrum.transpose(1, 2).reshape(batch*frames, 1, 100))
        frequency = self.trans_freq_encoder(frequency.transpose(1, 2)).transpose(1, 2)
        attention = self.freq_patch_embedding_2_atten(frequency)
        frequency = self.freq_patch_embedding_2(frequency) * attention
        frequency = frequency.reshape(batch, frames, self.hid_channels//2)
        time = self.temporal_patch_embedding(temporal[:, None]).transpose(1, 2)
        vectors = F.normalize(self.trans_temporal_encoder(torch.cat([frequency, time], dim=-1)), p=2, dim=-1)
        quantized, indices = self.quantizer(vectors)
        return quantized, indices, vectors

    def forward(self, spectrum, temporal):
        r'''Return reconstructed spectrum, IDs, straight-through codes and inputs.'''
        quantized, indices, vectors = self.tokenize(spectrum, temporal)
        straight_through = vectors + (quantized-vectors).detach()
        reconstruction = self.decoder(self.trans_decoder(straight_through)).transpose(1, 2)
        return reconstruction, indices, straight_through, vectors

    def quantization_loss(self, vectors, quantized):
        r'''EMA commitment loss; codebook vectors update through EMA, not Adam.

        Unlike adding the source's two opposing losses to straight-through
        vectors, this term retains the intended gradient into the encoder.
        Pass either raw or straight-through quantized vectors as target.
        '''
        return self.beta * F.mse_loss(vectors, quantized.detach())

    @classmethod
    def from_pretrained(cls, checkpoint_path, prefix='', reconstruction_head='linear', update_codebook=True):
        r'''Strictly load a complete local tokenizer with an explicit reconstruction head.'''
        return _load_complete(cls(reconstruction_head=reconstruction_head, update_codebook=update_codebook), checkpoint_path, prefix)


class TFMEncoder(nn.Module):
    r'''Complete pretrained token encoder, including the released output head.

    Input token IDs are [batch,electrodes,time], including an optional last
    vocabulary row reserved for masking. Up to16sequential channel rows and
    1000temporal positions are available. The released CHB-MIT binary head
    returns one logit, without sigmoid. The separate multi-dataset MTP release
    has a trained 8192-output head; select num_classes=8192 explicitly.

    Args:
        hid_channels (int): Embedding width, divisible by8. (default: :obj:`64`)
        num_tokens (int): Vocabulary without the mask row. (default: :obj:`8192`)
        num_layers (int): Linear attention layers. (default: :obj:`4`)
        num_classes (int): Output width. Official CHB-MIT uses1. (default: :obj:`1`)
    '''
    def __init__(self, hid_channels=64, num_tokens=8192, num_layers=4, num_classes=1):
        super().__init__()
        for name, value in [('hid_channels', hid_channels), ('num_tokens', num_tokens),
                            ('num_layers', num_layers), ('num_classes', num_classes)]:
            _positive_int(name, value)
        if hid_channels % 8:
            raise ValueError('hid_channels must be divisible by8.')
        self.hid_channels, self.num_tokens, self.num_layers, self.num_classes = hid_channels, num_tokens, num_layers, num_classes
        self.eeg_token_embedding = nn.Embedding(num_tokens+1, hid_channels)
        self.channel_embed = nn.Embedding(16, hid_channels)
        self.index = nn.Parameter(torch.arange(16), requires_grad=False)
        self.temporal_pos_embed = _PositionalEncoding(hid_channels, .1)
        self.pos_drop = nn.Dropout(.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hid_channels))
        self.LAT = _Transformer(hid_channels, 8, num_layers, .2)
        self.classification_head = nn.Linear(hid_channels, num_classes)

    def _tokens(self, ids, channel_ids):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 3 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('Require integer token IDs[batch,electrodes,time].')
        batch, channels, frames = ids.shape
        if batch < 1 or not 1 <= channels <= 16 or not 1 <= frames <= 1000:
            raise ValueError('Require nonempty batch,1--16electrodes and1--1000positions.')
        if (ids < 0).any() or (ids > self.num_tokens).any():
            raise ValueError('Token IDs exceed the vocabulary including the final mask row.')
        if channel_ids is None:
            rows = self.index[:channels]
        else:
            rows = torch.as_tensor(channel_ids, device=ids.device)
            if rows.shape != (channels,) or rows.dtype not in (torch.int32, torch.int64) or (rows < 0).any() or (rows >= 16).any() or rows.unique().numel() != channels:
                raise ValueError('Require one distinct channel row0--15 per electrode.')
        x = self.eeg_token_embedding(ids)
        # Loop order preserves the author's independent positional-dropout masks.
        parts = [self.temporal_pos_embed(x[:, index]+self.channel_embed(rows[index])) for index in range(channels)]
        x = torch.stack(parts, dim=1).reshape(batch, channels*frames, self.hid_channels)
        return self.LAT(torch.cat([self.cls_token.expand(batch, -1, -1), x], dim=1))

    def forward_features(self, ids, channel_ids=None):
        r'''Return the contextual class-token features [batch,hid_channels].'''
        return self._tokens(ids, channel_ids)[:, 0]

    def forward(self, ids, channel_ids=None):
        r'''Return raw class-token logits, preserving a one-logit binary head.'''
        return self.classification_head(self.forward_features(ids, channel_ids))

    def masked_prediction(self, ids, channel_ids=None):
        r'''Project each non-class token through the configured output head.'''
        return self.classification_head(self._tokens(ids, channel_ids)[:, 1:])

    @classmethod
    def from_pretrained(cls, checkpoint_path, prefix='', num_classes=1):
        r'''Strictly load every encoder tensor with an explicit original output width.

        The CHB-MIT binary release uses num_classes=1. The multi-dataset MTP
        release uses num_classes=8192, including its complete trained vocabulary
        head. No head tensors are inferred, replaced or silently discarded.
        '''
        return _load_complete(cls(num_classes=num_classes), checkpoint_path, prefix)


class TFMClassifier(nn.Module):
    r'''Frozen tokenizer, complete pretrained encoder and a new task head.

    Input is preprocessed floating EEG[batch,electrodes,samples] at200Hz.
    A periodic Hann window,200-point FFT,100-sample stride and no centering
    produce magnitude bins0--99. Hard IDs stop gradients into the tokenizer
    and raw EEG. The original encoder head remains in state but is unused.

    Args:
        tokenizer (TFMTokenizer): Complete tokenizer with EMA state/decoder.
        encoder (TFMEncoder): Complete encoder including its released head.
        num_classes (int): New task output width. (default: :obj:`2`)
        freeze_encoder (bool): Train only the new head. (default: :obj:`False`)
    '''
    def __init__(self, tokenizer, encoder, num_classes=2, freeze_encoder=False):
        super().__init__()
        if not isinstance(tokenizer, TFMTokenizer) or not isinstance(encoder, TFMEncoder):
            raise ValueError('Require complete TFMTokenizer and TFMEncoder models.')
        if not isinstance(freeze_encoder, bool) or tokenizer.num_tokens != encoder.num_tokens:
            raise ValueError('Require boolean freeze_encoder and matching vocabularies.')
        _positive_int('num_classes', num_classes)
        self.tokenizer, self.encoder, self.freeze_encoder = tokenizer, encoder, freeze_encoder
        self.classifier = nn.Linear(encoder.hid_channels, num_classes).to(encoder.cls_token)
        tokenizer.requires_grad_(False).eval()
        if freeze_encoder:
            encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    @torch.no_grad()
    def get_codebook_indices(self, x):
        r'''Return frozen clean-input teacher IDs[batch,electrodes,frames].'''
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_floating_point():
            raise ValueError('Require floating EEG[batch,electrodes,samples].')
        batch, channels, samples = x.shape
        if batch < 1 or not 1 <= channels <= 16 or samples < 200 or (samples-200)//100+1 > 1000:
            raise ValueError('Require nonempty batch,1--16channels and1--1000temporal frames.')
        temporal = x.reshape(batch*channels, samples)
        window = torch.hann_window(200, dtype=x.dtype, device=x.device)
        spectrum = torch.stft(temporal, n_fft=200, hop_length=100, window=window,
                              center=False, onesided=True, return_complex=True).abs()[:, :100]
        _, indices, _ = self.tokenizer.tokenize(spectrum, temporal)
        return indices.reshape(batch, channels, -1)

    def forward(self, x, channel_ids=None):
        r'''Return new-task logits, with no inferred normalization or filtering.'''
        ids = self.get_codebook_indices(x)
        return self.classifier(self.encoder.forward_features(ids, channel_ids))

    @classmethod
    def from_pretrained(cls, tokenizer_checkpoint, encoder_checkpoint, variant='multi-pretrained',
                        num_classes=2, freeze_encoder=False, tokenizer_prefix='', encoder_prefix=''):
        r'''Load a complete official tokenizer/encoder pair and add a new task head.

        multi-pretrained retains the 191-state MLP tokenizer and 59-state,
        8192-output MTP encoder. single-chbmit retains the 189-state linear
        tokenizer and 59-state, one-output binary encoder. Both preserve all
        original reconstruction and output heads, even though the new task
        uses class-token features. The new classifier is randomly initialized.
        Files must be supplied locally; weights and preprocessing are not
        downloaded or inferred. Pair selection is explicit and shape checked.
        '''
        variants = {'multi-pretrained': ('mlp', 8192), 'single-chbmit': ('linear', 1)}
        if not isinstance(variant, str) or variant not in variants:
            raise ValueError('variant must be multi-pretrained or single-chbmit.')
        reconstruction_head, original_classes = variants[variant]
        tokenizer = TFMTokenizer.from_pretrained(tokenizer_checkpoint, prefix=tokenizer_prefix,
            reconstruction_head=reconstruction_head, update_codebook=False)
        encoder = TFMEncoder.from_pretrained(encoder_checkpoint, prefix=encoder_prefix,
            num_classes=original_classes)
        return cls(tokenizer, encoder, num_classes=num_classes, freeze_encoder=freeze_encoder)
