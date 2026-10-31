# Adapted from pquochuy/SleepTransformer at 26167f26ddd8bb59ce787e56655c6e99ba446d2d.
# Copyright (c) 2023 pquochuy. MIT; see LICENSES/SLEEPTRANSFORMER-LICENSE.txt.
# Transformer blocks originate from Kyubyong Park (2019), ported to PyTorch here.
# Retain Apache-2.0 terms in LICENSES/SLEEPTRANSFORMER-TRANSFORMER-LICENSE.txt.

from collections import OrderedDict
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _position_table(length, width):
    positions = np.arange(length, dtype=np.float64)[:, None]
    dimensions = np.arange(width)
    angles = positions / np.power(10000.0, (dimensions - dimensions % 2) / width)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.from_numpy(angles.astype(np.float32))


class _SleepLayerNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).square().mean(-1, keepdim=True)
        return self.weight * ((x - mean) / (variance + 1e-8).pow(0.5)) + self.bias


class _SleepTransformerBlock(nn.Module):
    def __init__(self, width, heads, ffn_size, dropout, attention_dropout):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.attention_norm = _SleepLayerNorm(width)
        self.ffn_in = nn.Linear(width, ffn_size)
        self.ffn_out = nn.Linear(ffn_size, width)
        self.ffn_norm = _SleepLayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        for layer in [self.query, self.key, self.value, self.ffn_in, self.ffn_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        batch, length, width = x.shape
        head_width = width // self.heads

        def split_heads(value):
            # Match the reference's head-major packed batch arrangement.
            return (
                value.reshape(batch, length, self.heads, head_width)
                .permute(2, 0, 1, 3)
                .reshape(self.heads * batch, length, head_width)
            )

        query, key, value = [
            split_heads(layer(x)) for layer in [self.query, self.key, self.value]
        ]
        probabilities = (query @ key.transpose(-1, -2) / math.sqrt(head_width)).softmax(
            -1
        )
        mixed = self.attention_dropout(probabilities) @ value
        mixed = (
            mixed.reshape(self.heads, batch, length, head_width)
            .permute(1, 2, 0, 3)
            .reshape(batch, length, width)
        )
        # The original attention has no learned output projection.
        output = self.attention_norm(mixed + x)
        hidden = self.ffn_out(F.relu(self.ffn_in(output)))
        # Feedforward dropout follows residual normalization in the reference.
        output = self.dropout(self.ffn_norm(hidden + output))
        return output, probabilities.reshape(self.heads, batch, length, length).permute(
            1, 0, 2, 3
        )


class _SleepEncoder(nn.Module):
    def __init__(
        self, width, max_length, depth, heads, ffn_size, dropout, attention_dropout
    ):
        super().__init__()
        self.width = width
        self.register_buffer(
            'positions', _position_table(max_length, width), persistent=False
        )
        self.blocks = nn.ModuleList(
            [
                _SleepTransformerBlock(
                    width, heads, ffn_size, dropout, attention_dropout
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, return_attention=False):
        x = x * math.sqrt(self.width) + self.positions[: x.shape[1]].unsqueeze(0)
        attention = []
        for block in self.blocks:
            x, probabilities = block(x)
            if return_attention:
                attention.append(probabilities)
        return x, attention


class _SleepContextAttention(nn.Module):
    def __init__(self, width, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, hidden_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        self.vector = nn.Parameter(torch.empty(hidden_size))
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.1)

    def forward(self, x):
        hidden = torch.tanh(x @ self.weight + self.bias)
        exponential = torch.exp(hidden @ self.vector)
        weights = exponential / exponential.sum(-1, keepdim=True)
        return (x * weights.unsqueeze(-1)).sum(1), weights


class SleepTransformer(nn.Module):
    r"""Complete hierarchical SleepTransformer from the original TensorFlow model.

    Retains both within-epoch frame and across-epoch Transformer stacks,
    sinusoidal positions, learned frame-context attention and the full two-
    hidden-layer classifier. Input is floating
    :obj:`[batch, epochs, frames, frequencies, electrodes]`; output is
    :obj:`[batch, epochs, num_classes]` categorical logits. Spectral features,
    sampling rates, units, channels and normalization are supplied by the
    caller. There is no extra learned spectral condenser or attention output
    projection. The source is MIT-licensed; TensorFlow is not a runtime
    dependency and no pretrained weights are bundled.

    Args:
        num_electrodes (int): Signal channels in the last input axis. (default: 1)
        num_freqs (int): Spectral bins per channel. (default: 128)
        num_frames (int): Maximum spectral frames per epoch. (default: 29)
        sequence_length (int): Maximum epochs per context. (default: 20)
        num_classes (int): Output sleep stages. (default: 5)
        frame_depth (int): Frame Transformer blocks. (default: 2)
        sequence_depth (int): Epoch-context Transformer blocks. (default: 2)
        heads (int): Heads at both levels; divides frequencies times
            channels. (default: 8)
        frame_ffn_size (int): Frame feedforward hidden width. (default: 1024)
        sequence_ffn_size (int): Context feedforward hidden width. (default: 1024)
        attention_hidden_size (int): Frame-context pooling width. (default: 64)
        hid_channels (int): Width of both full classifier layers. (default: 1024)
        dropout (float): Feedforward and classifier dropout. (default: 0.1)
        attention_dropout (float): Attention-weight dropout. (default: 0.1)

    .. code-block:: python

        model = SleepTransformer()
        spectrogram = torch.randn(2, 20, 29, 128, 1)
        logits = model(spectrogram)
        targets = torch.randint(0, 5, (2, 20))
        loss = model.compute_loss(logits, targets)
        loss.backward()
    """

    SOURCE_REVISION = '26167f26ddd8bb59ce787e56655c6e99ba446d2d'

    def __init__(
        self,
        num_electrodes: int = 1,
        num_freqs: int = 128,
        num_frames: int = 29,
        sequence_length: int = 20,
        num_classes: int = 5,
        frame_depth: int = 2,
        sequence_depth: int = 2,
        heads: int = 8,
        frame_ffn_size: int = 1024,
        sequence_ffn_size: int = 1024,
        attention_hidden_size: int = 64,
        hid_channels: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        for name, value in [
            ('num_electrodes', num_electrodes),
            ('num_freqs', num_freqs),
            ('num_frames', num_frames),
            ('sequence_length', sequence_length),
            ('num_classes', num_classes),
            ('frame_depth', frame_depth),
            ('sequence_depth', sequence_depth),
            ('heads', heads),
            ('frame_ffn_size', frame_ffn_size),
            ('sequence_ffn_size', sequence_ffn_size),
            ('attention_hidden_size', attention_hidden_size),
            ('hid_channels', hid_channels),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        for name, value in [
            ('dropout', dropout),
            ('attention_dropout', attention_dropout),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError(f'{name} must be finite and in [0, 1).')
        width = num_electrodes * num_freqs
        if width % heads:
            raise ValueError('heads must divide num_freqs * num_electrodes.')
        self.num_electrodes = num_electrodes
        self.num_freqs = num_freqs
        self.num_frames = num_frames
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.frame_encoder = _SleepEncoder(
            width,
            num_frames,
            frame_depth,
            heads,
            frame_ffn_size,
            dropout,
            attention_dropout,
        )
        self.frame_attention = _SleepContextAttention(width, attention_hidden_size)
        self.sequence_encoder = _SleepEncoder(
            width,
            sequence_length,
            sequence_depth,
            heads,
            sequence_ffn_size,
            dropout,
            attention_dropout,
        )
        self.fc1 = nn.Linear(width, hid_channels)
        self.fc2 = nn.Linear(hid_channels, hid_channels)
        self.output = nn.Linear(hid_channels, num_classes)
        self.classifier_dropout = nn.Dropout(dropout)
        for layer in [self.fc1, self.fc2, self.output]:
            nn.init.xavier_uniform_(layer.weight)
            # tf.get_variable uses its default Glorot initializer for these biases.
            nn.init.uniform_(
                layer.bias,
                -math.sqrt(3.0 / layer.out_features),
                math.sqrt(3.0 / layer.out_features),
            )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        r"""Return all epoch logits and optionally both full attention hierarchies."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('x must be a floating spectral EEG tensor.')
        if (
            x.ndim != 5
            or x.shape[0] < 1
            or not 1 <= x.shape[1] <= self.sequence_length
            or not 1 <= x.shape[2] <= self.num_frames
            or x.shape[3:] != (self.num_freqs, self.num_electrodes)
        ):
            raise ValueError(
                'Expected [batch,epochs,frames,num_freqs,num_electrodes] within configured context limits.'
            )
        if not isinstance(return_attention, bool):
            raise TypeError('return_attention must be a boolean.')
        batch, epochs, frames = x.shape[:3]
        flat = x.reshape(batch * epochs, frames, self.num_freqs * self.num_electrodes)
        encoded, frame_scores = self.frame_encoder(flat, return_attention)
        context, frame_weights = self.frame_attention(encoded)
        encoded, sequence_scores = self.sequence_encoder(
            context.reshape(batch, epochs, -1), return_attention
        )
        hidden = self.classifier_dropout(F.relu(self.fc1(encoded)))
        hidden = self.classifier_dropout(F.relu(self.fc2(hidden)))
        logits = self.output(hidden)
        if return_attention:
            return logits, dict(
                frame_context=frame_weights.reshape(batch, epochs, frames),
                frame_self_attention=[
                    value.reshape(batch, epochs, *value.shape[1:])
                    for value in frame_scores
                ],
                sequence_self_attention=sequence_scores,
            )
        return logits

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, l2_weight: float = 1e-4
    ):
        r"""Native batch-summed epoch-mean CE plus half-squared L2 of all parameters.

        The reference averages over epochs and sums over batch, rather than
        averaging over both axes. L2 includes all learned weights and biases.
        Do not also add optimizer weight decay when reproducing this penalty.
        """
        if (
            isinstance(l2_weight, bool)
            or not isinstance(l2_weight, (int, float))
            or not math.isfinite(l2_weight)
            or l2_weight < 0
        ):
            raise ValueError('l2_weight must be finite and nonnegative.')
        if (
            logits.ndim != 3
            or logits.shape[-1] != self.num_classes
            or targets.shape != logits.shape[:2]
        ):
            raise ValueError(
                'Expected logits [batch,epochs,num_classes] and matching categorical targets.'
            )
        loss = (
            F.cross_entropy(
                logits.reshape(-1, self.num_classes),
                targets.reshape(-1),
                reduction='sum',
            )
            / logits.shape[1]
        )
        penalty = sum(parameter.square().sum() for parameter in self.parameters()) * 0.5
        return loss + l2_weight * penalty

    def _native_mapping(self):
        mapping = OrderedDict()
        for level, encoder in [
            ('frame_transformer', self.frame_encoder),
            ('seq_transformer', self.sequence_encoder),
        ]:
            prefix = (
                'frame_encoder' if level == 'frame_transformer' else 'sequence_encoder'
            )
            for i, _block in enumerate(encoder.blocks):
                native = f'{level}/encoder/num_blocks_{i}/'
                public = f'{prefix}.blocks.{i}.'
                for j, name in enumerate(['query', 'key', 'value']):
                    dense = 'dense' if j == 0 else f'dense_{j}'
                    mapping[native + 'multihead_attention/' + dense + '/kernel'] = (
                        public + name + '.weight',
                        True,
                    )
                    mapping[native + 'multihead_attention/' + dense + '/bias'] = (
                        public + name + '.bias',
                        False,
                    )
                for native_scope, name in [
                    ('multihead_attention', 'attention_norm'),
                    ('positionwise_feedforward', 'ffn_norm'),
                ]:
                    mapping[native + native_scope + '/ln/gamma'] = (
                        public + name + '.weight',
                        False,
                    )
                    mapping[native + native_scope + '/ln/beta'] = (
                        public + name + '.bias',
                        False,
                    )
                for j, name in enumerate(['ffn_in', 'ffn_out']):
                    dense = 'dense' if j == 0 else 'dense_1'
                    mapping[
                        native + 'positionwise_feedforward/' + dense + '/kernel'
                    ] = (public + name + '.weight', True)
                    mapping[native + 'positionwise_feedforward/' + dense + '/bias'] = (
                        public + name + '.bias',
                        False,
                    )
        for native, name in [
            ('Variable', 'weight'),
            ('Variable_1', 'bias'),
            ('Variable_2', 'vector'),
        ]:
            mapping['frame_attention_layer/' + native] = (
                'frame_attention.' + name,
                False,
            )
        for native, name in [('fc1', 'fc1'), ('fc2', 'fc2'), ('output', 'output')]:
            mapping['output_layer/' + native + '/weights'] = (name + '.weight', True)
            mapping['output_layer/' + native + '/biases'] = (name + '.bias', False)
        return mapping

    def native_state_dict(self):
        """Return every native TensorFlow variable layout without optimizer slots."""
        state = self.state_dict()
        return OrderedDict(
            (name, state[key].T if transpose else state[key])
            for name, (key, transpose) in self._native_mapping().items()
        )

    def load_native_state_dict(self, state_dict, strict: bool = True):
        """Load all TensorFlow model variables with explicit dense transpositions.

        Accept optional trailing :0 variable suffixes. No partial shape or
        guessed key matching is performed. Native optimizer slots are not
        model variables and should be supplied separately, not silently dropped.
        """
        normalized = {}
        for name, value in state_dict.items():
            canonical = name[:-2] if name.endswith(':0') else name
            if canonical in normalized:
                raise ValueError('Duplicate native variable name: ' + canonical)
            normalized[canonical] = value
        mapping = self._native_mapping()
        if strict and set(normalized) != set(mapping):
            raise ValueError('Native variables must exactly match the complete model.')
        converted = {}
        for name, (key, transpose) in mapping.items():
            if name in normalized:
                value = torch.as_tensor(normalized[name])
                expected = self.state_dict()[key].shape
                native_shape = tuple(reversed(expected)) if transpose else expected
                if value.shape != native_shape:
                    raise RuntimeError('Invalid native tensor shape: ' + name)
                converted[key] = value.T if transpose else value
        return self.load_state_dict(converted, strict=strict)
