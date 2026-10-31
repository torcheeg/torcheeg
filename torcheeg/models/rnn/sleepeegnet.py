"""SleepEEGNet's full CNN, recurrent encoder and attention sequence decoder.

New PyTorch implementation of the published architecture (arXiv:1903.02108).
The separately supplied author TensorFlow source is used only as a numerical
reference; its academic/non-commercial source is not included in this package.
"""

from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn import functional as F


def _same_padding(length, kernel, stride):
    total = max((math.ceil(length / stride) - 1) * stride + kernel - length, 0)
    return total // 2, total - total // 2


class _SleepTemporalBranch(nn.Module):
    def __init__(self, kernel, stride, first_pool, inner_kernel, last_pool, dropout):
        super().__init__()
        self.first = nn.Conv1d(1, 64, kernel, stride=stride)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(64 if i == 0 else 128, 128, inner_kernel) for i in range(3)]
        )
        self.first_pool = first_pool
        self.last_pool = last_pool
        self.dropout = nn.Dropout(dropout)
        for layer in [self.first] + list(self.convolutions):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def convolve(x, layer):
        left, right = _same_padding(x.shape[-1], layer.kernel_size[0], layer.stride[0])
        return F.relu(layer(F.pad(x, (left, right))))

    @staticmethod
    def pool(x, size):
        left, right = _same_padding(x.shape[-1], size, size)
        return F.max_pool1d(F.pad(x, (left, right), value=float('-inf')), size, size)

    def forward(self, x):
        x = self.dropout(self.pool(self.convolve(x, self.first), self.first_pool))
        for layer in self.convolutions:
            x = self.convolve(x, layer)
        x = self.pool(x, self.last_pool)
        # TensorFlow flattens time before channels, rather than channels first.
        return x.transpose(1, 2).reshape(x.shape[0], -1)


class _SleepSequenceCell(nn.Module):
    """Single-kernel LSTM, with i/j/f/o gates and an explicit unit forget bias."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel = nn.Parameter(
            torch.empty(input_size + hidden_size, 4 * hidden_size)
        )
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, x, state):
        cell, hidden = state
        incoming, candidate, forget, output = (
            torch.cat((x, hidden), -1) @ self.kernel + self.bias
        ).chunk(4, -1)
        cell = cell * torch.sigmoid(forget + 1) + torch.sigmoid(incoming) * torch.tanh(
            candidate
        )
        hidden = torch.tanh(cell) * torch.sigmoid(output)
        return hidden, (cell, hidden)


class SleepEEGNet(nn.Module):
    r"""Complete single-channel CNN/sequence-to-sequence SleepEEGNet.

    Two full temporal CNN branches feed a stacked recurrent encoder, learned
    token embedding, stacked recurrent decoder, input-fed Luong attention and
    seven-token output projection. This is a newly written PyTorch
    implementation of the published method; no author code or pretrained
    weights are redistributed. The separately supplied TensorFlow 1 reference
    uses academic/non-commercial terms.

    Input is floating :obj:`[batch, epochs, 1, chunk_size]` raw EEG. Channels,
    physical rates, units and normalization are supplied by the caller.
    :meth:`forward` requires decoder input tokens; :meth:`generate` performs
    autoregressive inference without ground-truth labels. Five sleep stages
    use zero through four, start-of-decoding is five, end-of-decoding is six.
    The default recurrent direction reproduces the released code; the paper's
    bidirectional encoder is available explicitly.

    Args:
        chunk_size (int): Samples per EEG epoch. (default: 3000)
        sequence_length (int): Maximum input epochs per context. (default: 10)
        num_classes (int): Sleep stage classes, excluding two special tokens.
            (default: 5)
        hid_channels (int): Hidden width of each encoder direction. (default: 128)
        num_layers (int): Encoder and decoder recurrent layers. (default: 2)
        embedding_size (int): Learned decoder token width. (default: 10)
        attention_size (int): Attentional decoder output width. (default: 64)
        bidirectional (bool): Independent forward/backward stacked encoders;
            decoder states concatenate both directions. (default: False)
        dropout (float): CNN dropout probability. (default: 0.5)

    .. code-block:: python

        model = SleepEEGNet()
        eeg = torch.randn(2, 10, 1, 3000)
        labels = torch.randint(0, 5, (2, 10))
        decoder_inputs, targets = model.prepare_targets(labels)
        logits = model(eeg, decoder_inputs)
        loss = model.compute_loss(logits, targets)
        loss.backward()
        model.eval()
        prediction = model.generate(eeg)
    """

    SOURCE_REVISION = '288ee50700ef34780c6e056a884cf22308aa6de2'

    def __init__(
        self,
        chunk_size: int = 3000,
        sequence_length: int = 10,
        num_classes: int = 5,
        hid_channels: int = 128,
        num_layers: int = 2,
        embedding_size: int = 10,
        attention_size: int = 64,
        bidirectional: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        for name, value in [
            ('chunk_size', chunk_size),
            ('sequence_length', sequence_length),
            ('num_classes', num_classes),
            ('hid_channels', hid_channels),
            ('num_layers', num_layers),
            ('embedding_size', embedding_size),
            ('attention_size', attention_size),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + ' must be a positive integer.')
        if not isinstance(bidirectional, bool):
            raise TypeError('bidirectional must be a boolean.')
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError('dropout must be finite and in [0,1).')
        self.chunk_size = chunk_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.start_token = num_classes
        self.end_token = num_classes + 1
        self.small_branch = _SleepTemporalBranch(50, 6, 8, 8, 4, dropout)
        self.large_branch = _SleepTemporalBranch(400, 50, 4, 6, 2, dropout)
        self.feature_dropout = nn.Dropout(dropout)
        small_steps = math.ceil(math.ceil(math.ceil(chunk_size / 6) / 8) / 4)
        large_steps = math.ceil(math.ceil(math.ceil(chunk_size / 50) / 4) / 2)
        self.feature_size = (small_steps + large_steps) * 128
        self.embedding = nn.Embedding(num_classes + 2, embedding_size)
        nn.init.uniform_(self.embedding.weight, -1, 1)
        self.encoder_forward = nn.ModuleList(
            [
                _SleepSequenceCell(
                    self.feature_size if i == 0 else hid_channels, hid_channels
                )
                for i in range(num_layers)
            ]
        )
        self.encoder_backward = (
            nn.ModuleList(
                [
                    _SleepSequenceCell(
                        self.feature_size if i == 0 else hid_channels, hid_channels
                    )
                    for i in range(num_layers)
                ]
            )
            if bidirectional
            else nn.ModuleList()
        )
        decoder_width = hid_channels * (2 if bidirectional else 1)
        self.decoder = nn.ModuleList(
            [
                _SleepSequenceCell(
                    embedding_size + attention_size if i == 0 else decoder_width,
                    decoder_width,
                )
                for i in range(num_layers)
            ]
        )
        self.memory_projection = nn.Linear(decoder_width, decoder_width, bias=False)
        self.attention_projection = nn.Linear(
            2 * decoder_width, attention_size, bias=False
        )
        self.output = nn.Linear(attention_size, num_classes + 2, bias=False)
        for layer in [self.memory_projection, self.attention_projection, self.output]:
            nn.init.xavier_uniform_(layer.weight)

    def _validate_input(self, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('x must be a floating raw EEG tensor.')
        if (
            x.ndim != 4
            or x.shape[0] < 1
            or not 1 <= x.shape[1] <= self.sequence_length
            or x.shape[2:] != (1, self.chunk_size)
        ):
            raise ValueError(
                'Expected [batch,epochs,1,chunk_size] within the context limit.'
            )

    @staticmethod
    def _encode_direction(features, layers, reverse=False):
        states = [
            (
                features.new_zeros(features.shape[0], layer.hidden_size),
                features.new_zeros(features.shape[0], layer.hidden_size),
            )
            for layer in layers
        ]
        outputs = []
        indices = (
            range(features.shape[1] - 1, -1, -1)
            if reverse
            else range(features.shape[1])
        )
        for i in indices:
            hidden = features[:, i]
            for j, layer in enumerate(layers):
                hidden, states[j] = layer(hidden, states[j])
            outputs.append(hidden)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, 1), states

    def encode(self, x):
        """Return full encoder outputs and every layer's (cell, hidden) state."""
        self._validate_input(x)
        batch, epochs = x.shape[:2]
        signal = x.reshape(batch * epochs, 1, self.chunk_size)
        features = self.feature_dropout(
            torch.cat((self.small_branch(signal), self.large_branch(signal)), -1)
        ).reshape(batch, epochs, self.feature_size)
        memory, states = self._encode_direction(features, self.encoder_forward)
        if self.bidirectional:
            backward, backward_states = self._encode_direction(
                features, self.encoder_backward, reverse=True
            )
            memory = torch.cat((memory, backward), -1)
            states = [
                (torch.cat((a[0], b[0]), -1), torch.cat((a[1], b[1]), -1))
                for a, b in zip(states, backward_states)
            ]
        return memory, states

    def _decode_step(self, tokens, states, previous_attention, memory, keys):
        hidden = torch.cat((self.embedding(tokens), previous_attention), -1)
        updated = []
        for layer, state in zip(self.decoder, states):
            hidden, state = layer(hidden, state)
            updated.append(state)
        alignment = torch.bmm(keys, hidden.unsqueeze(-1)).squeeze(-1).softmax(-1)
        context = torch.bmm(alignment.unsqueeze(1), memory).squeeze(1)
        attention = self.attention_projection(torch.cat((hidden, context), -1))
        return self.output(attention), updated, attention, alignment

    def forward(
        self,
        x: torch.Tensor,
        decoder_inputs: torch.Tensor,
        return_attention: bool = False,
    ):
        """Teacher-forced full decoder logits, optionally with encoder alignments."""
        memory, states = self.encode(x)
        if (
            not isinstance(decoder_inputs, torch.Tensor)
            or decoder_inputs.dtype != torch.long
        ):
            raise TypeError('decoder_inputs must be int64 tokens.')
        if (
            decoder_inputs.ndim != 2
            or decoder_inputs.shape[0] != x.shape[0]
            or not 1 <= decoder_inputs.shape[1] <= x.shape[1] + 1
        ):
            raise ValueError(
                'Expected decoder tokens [batch,steps], at most epochs+1 steps.'
            )
        if decoder_inputs.device != x.device:
            raise ValueError('EEG and decoder tokens must share a device.')
        if torch.any((decoder_inputs < 0) | (decoder_inputs > self.end_token)):
            raise ValueError('Decoder tokens must be stages, SOD or EOD.')
        keys = self.memory_projection(memory)
        attention = memory.new_zeros(x.shape[0], self.attention_projection.out_features)
        logits, alignments = [], []
        for tokens in decoder_inputs.unbind(1):
            scores, states, attention, alignment = self._decode_step(
                tokens, states, attention, memory, keys
            )
            logits.append(scores)
            alignments.append(alignment)
        result = torch.stack(logits, 1)
        return (result, torch.stack(alignments, 1)) if return_attention else result

    def prepare_targets(self, labels):
        """Prepend SOD to decoder inputs and append EOD to training targets."""
        if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
            raise TypeError('labels must be int64 sleep-stage targets.')
        if (
            labels.ndim != 2
            or labels.shape[0] < 1
            or not 1 <= labels.shape[1] <= self.sequence_length
            or torch.any((labels < 0) | (labels >= self.num_classes))
        ):
            raise ValueError('Expected categorical stage labels [batch,epochs].')
        start = labels.new_full((labels.shape[0], 1), self.start_token)
        end = labels.new_full((labels.shape[0], 1), self.end_token)
        return torch.cat((start, labels), 1), torch.cat((labels, end), 1)

    def compute_loss(self, logits, targets, l2_weight: float = 0.001):
        """Native per-sequence sum of class-mean CEs plus non-bias half L2.

        Average each class's token losses separately within each sequence,
        sum the present classes, then average sequences. EOD participates;
        absent classes contribute zero. This differs from ordinary token CE.
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
            or logits.shape[-1] != self.num_classes + 2
            or targets.shape != logits.shape[:2]
        ):
            raise ValueError(
                'Expected matching [batch,steps,vocabulary] logits and tokens.'
            )
        losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
        per_sequence = logits.new_zeros(logits.shape[0])
        for stage in range(self.num_classes + 2):
            mask = targets == stage
            per_sequence = per_sequence + (losses * mask).sum(1) / (mask.sum(1) + 1e-12)
        penalty = (
            sum(
                value.square().sum()
                for name, value in self.named_parameters()
                if not name.endswith('bias')
            )
            * 0.5
        )
        return per_sequence.mean() + l2_weight * penalty

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_steps=None):
        """Autoregress from SOD without labels, stopping after EOD or max_steps.

        Return tokens, logits, alignments and first-EOD-inclusive lengths.
        Following the reference's greedy decoder, rows finished earlier keep
        advancing until all rows finish; ignore tokens beyond each length.
        Calling ``eval()`` is the caller's responsibility.
        """
        memory, states = self.encode(x)
        if max_steps is None:
            max_steps = x.shape[1] + 2
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < 1
        ):
            raise ValueError('max_steps must be a positive integer.')
        tokens = torch.full(
            (x.shape[0],), self.start_token, device=x.device, dtype=torch.long
        )
        attention = memory.new_zeros(x.shape[0], self.attention_projection.out_features)
        keys = self.memory_projection(memory)
        finished = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        lengths = torch.full(
            (x.shape[0],), max_steps, dtype=torch.long, device=x.device
        )
        predictions, outputs, alignments = [], [], []
        for step in range(max_steps):
            scores, states, attention, alignment = self._decode_step(
                tokens, states, attention, memory, keys
            )
            tokens = scores.argmax(-1)
            newly_finished = ~finished & (tokens == self.end_token)
            lengths = torch.where(
                newly_finished, torch.full_like(lengths, step + 1), lengths
            )
            finished = finished | newly_finished
            predictions.append(tokens)
            outputs.append(scores)
            alignments.append(alignment)
            if bool(finished.all()):
                break
        return dict(
            tokens=torch.stack(predictions, 1),
            logits=torch.stack(outputs, 1),
            attention=torch.stack(alignments, 1),
            lengths=lengths,
        )

    def _native_mapping(self):
        mapping = OrderedDict()
        for branch_index, (branch_name, branch) in enumerate(
            [('small_branch', self.small_branch), ('large_branch', self.large_branch)]
        ):
            for index, name in enumerate(
                ['first', 'convolutions.0', 'convolutions.1', 'convolutions.2']
            ):
                number = branch_index * 4 + index
                scope = 'conv1d' if number == 0 else 'conv1d_' + str(number)
                mapping[scope + '/kernel'] = (
                    branch_name + '.' + name + '.weight',
                    (2, 1, 0),
                )
                mapping[scope + '/bias'] = (branch_name + '.' + name + '.bias', None)
        mapping['embeddin/dec_embedding'] = ('embedding.weight', None)
        encoders = [('encoding/rnn', 'encoder_forward')]
        if self.bidirectional:
            encoders = [
                ('encoding/bidirectional_rnn/fw', 'encoder_forward'),
                ('encoding/bidirectional_rnn/bw', 'encoder_backward'),
            ]
        for native, public in encoders:
            for i in range(self.num_layers):
                for variable in ['kernel', 'bias']:
                    mapping[
                        f'{native}/multi_rnn_cell/cell_{i}/lstm_cell/{variable}'
                    ] = (f'{public}.{i}.{variable}', None)
        mapping['decoding/memory_layer/kernel'] = ('memory_projection.weight', (1, 0))
        for i in range(self.num_layers):
            for variable in ['kernel', 'bias']:
                mapping[
                    f'decoding/decoder/attention_wrapper/multi_rnn_cell/cell_{i}/lstm_cell/{variable}'
                ] = (f'decoder.{i}.{variable}', None)
        mapping['decoding/decoder/attention_wrapper/attention_layer/kernel'] = (
            'attention_projection.weight',
            (1, 0),
        )
        mapping['decoding/decoder/dense/kernel'] = ('output.weight', (1, 0))
        return mapping

    def native_state_dict(self):
        """Export all exact TensorFlow model variable names and tensor layouts."""
        state = self.state_dict()
        return OrderedDict(
            (native, state[key].permute(order) if order else state[key])
            for native, (key, order) in self._native_mapping().items()
        )

    def load_native_state_dict(self, state_dict, strict: bool = True):
        """Strictly import full CNN/encoder/attention/decoder native variables.

        Convolutions transpose kernel/input/output axes explicitly. Recurrent
        kernels retain their native single-kernel i/j/f/o gate layout. Native
        optimizer slots must be handled separately; no guessed matching occurs.
        """
        normalized = {}
        for name, value in state_dict.items():
            key = name[:-2] if name.endswith(':0') else name
            if key in normalized:
                raise ValueError('Duplicate native variable name: ' + key)
            normalized[key] = value
        mapping = self._native_mapping()
        if strict and set(normalized) != set(mapping):
            raise ValueError('Native variables must exactly match the complete model.')
        state = self.state_dict()
        converted = {}
        for native, (key, order) in mapping.items():
            if native not in normalized:
                continue
            value = torch.as_tensor(normalized[native])
            expected = (
                tuple(state[key].shape[d] for d in order) if order else state[key].shape
            )
            if value.shape != expected:
                raise RuntimeError('Invalid native variable shape: ' + native)
            # All mappings used here are self-inverse transpositions.
            converted[key] = value.permute(order) if order else value
        return self.load_state_dict(converted, strict=strict)
