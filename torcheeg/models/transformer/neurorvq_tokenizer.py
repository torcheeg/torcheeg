"""Complete, explicitly supplied NeuroRVQ tokenizer (external CC-BY-NC-4.0).

No author source, checkpoint, or replacement neural network is bundled here.
"""

from contextlib import contextmanager

import torch
from torch import distributed, nn

from .neurorvq import _NeuroRVQInput


def _identity_reduce(value):
    return value


class NeuroRVQTokenizer(_NeuroRVQInput):
    r"""Complete external NeuroRVQ tokenizer with 32 ordered RVQ streams.

    Input is floating EEG [B,C,T] at 200 Hz, with complete one-second patches.
    Supply the full pinned author encoder, decoder, reconstruction heads and
    four eight-layer RVQs. Token stream ``8 * branch + residual`` uses the
    original branch order. Each stream has 8192 codes of width 128.

    Training preserves native EMA updates and all four native losses. In
    evaluation, ``track_usage=False`` restores the 32 usage counters after a
    call and requires initialized codebooks. This permits a frozen teacher
    without silently mutating its state. ``track_usage=True`` retains native
    evaluation statistics. Calls on one instance must be serialized.

    Args:
        backend (nn.Module): Explicit complete local author tokenizer.
        channel_names (list or tuple): Ordered names matching the input.
        channel_vocab (list or tuple): Original ordered electrode vocabulary.
        sampling_rate (int): Physical rate, must be 200 Hz.
        track_usage (bool): Retain usage updates during evaluation.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        backend,
        channel_names,
        channel_vocab,
        sampling_rate=200,
        track_usage=False,
    ):
        if not isinstance(backend, nn.Module) or not isinstance(track_usage, bool):
            raise ValueError('Require an explicit backend and boolean track_usage.')
        super().__init__(channel_names, channel_vocab, sampling_rate)
        if (
            getattr(backend, 'patch_size', None) != 200
            or getattr(backend, 'code_dim', None) != 128
        ):
            raise ValueError(
                'Require the complete 200-sample/128-code-width tokenizer.'
            )
        for name, depth, encoder in [('encoder', 12, True), ('decoder', 3, False)]:
            module = getattr(backend, name, None)
            if (
                not isinstance(module, nn.Module)
                or getattr(module, 'embed_dim', None) != 200
                or getattr(module, 'num_heads', None) != 10
                or len(getattr(module, 'blocks', [])) != depth
                or getattr(module, 'use_as_encoder', None) is not encoder
                or getattr(module, 'use_for_pretraining', True)
            ):
                raise ValueError('Require the complete native tokenizer ' + name + '.')
            for key, shape in [
                ('pos_embed', (len(self.channel_vocab) + 1, 200)),
                ('time_embed', (256, 200)),
            ]:
                if (
                    not isinstance(getattr(module, key, None), torch.Tensor)
                    or getattr(module, key).shape != shape
                ):
                    raise ValueError(
                        'Tokenizer spatial or temporal embedding table differs.'
                    )
        for name in [
            'encode',
            'decode',
            'std_norm',
            'calculate_signal_rec_loss',
            'calculate_rec_loss',
            'calculate_phase_loss',
        ]:
            if not callable(getattr(backend, name, None)):
                raise ValueError('Missing native tokenizer method: ' + name)
        for index in range(1, 5):
            layers = getattr(
                getattr(backend, 'quantize_' + str(index), None), 'layers', []
            )
            if len(layers) != 8:
                raise ValueError('Require all four eight-layer residual quantizers.')
            for layer in layers:
                embedding = getattr(layer, 'embedding', None)
                weight = getattr(embedding, 'weight', None)
                if (
                    not isinstance(weight, torch.Tensor)
                    or weight.shape != (8192, 128)
                    or not isinstance(
                        getattr(layer, 'cluster_size', None), torch.Tensor
                    )
                    or layer.cluster_size.shape != (8192,)
                    or not isinstance(getattr(embedding, 'initted', None), torch.Tensor)
                ):
                    raise ValueError(
                        'Require complete 8192-by-128 EMA codebooks and statistics.'
                    )
        heads = [('encode_task_layer_' + str(i), 200, 128) for i in range(1, 5)]
        heads += [
            (name, 800, 200)
            for name in [
                'decode_task_layer_amplitude',
                'decode_task_layer_angle_sin',
                'decode_task_layer_angle_cos',
            ]
        ]
        for name, width, output in heads:
            head = getattr(backend, name, None)
            if (
                not isinstance(head, nn.Sequential)
                or len(head) < 3
                or not isinstance(head[0], nn.Linear)
                or not isinstance(head[2], nn.Linear)
                or (head[0].in_features, head[0].out_features) != (width, 200)
                or (head[2].in_features, head[2].out_features) != (200, output)
            ):
                raise ValueError('Require complete native tokenizer head: ' + name)
        if len(backend.state_dict()) != 505:
            raise ValueError(
                'Require all 505 native tokenizer states, including decoder heads.'
            )
        self.backend = backend
        self.track_usage = track_usage
        self.register_buffer(
            'usage_policy', torch.tensor(int(track_usage), dtype=torch.uint8)
        )
        self.train(backend.training)

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
        value = state_dict.get(prefix + 'usage_policy')
        expected = torch.tensor(int(self.track_usage), dtype=torch.uint8)
        if isinstance(value, torch.Tensor) and (
            value.dtype != expected.dtype or not torch.equal(value.cpu(), expected)
        ):
            error_msgs.append('NeuroRVQ tokenizer evaluation usage policy differs.')
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _quantizers(self):
        return [
            layer
            for index in range(1, 5)
            for layer in getattr(self.backend, 'quantize_' + str(index)).layers
        ]

    @contextmanager
    def _ema_context(self):
        layers = self._quantizers()
        readonly = not self.training and not self.track_usage
        active = distributed.is_available() and distributed.is_initialized()
        initialized = all(bool(layer.embedding.initted.item()) for layer in layers)
        if not initialized and (
            readonly or (active and distributed.get_world_size() > 1)
        ):
            raise ValueError(
                'Require initialized codebooks for read-only evaluation or distributed training.'
            )
        # Native source captures the process group at construction. Refresh it
        # here so constructing a model before init_process_group is supported.
        reduce_fn = (
            distributed.all_reduce if active and not readonly else _identity_reduce
        )
        for layer in layers:
            if 'all_reduce_fn' in layer._modules:
                del layer._modules['all_reduce_fn']
            layer.all_reduce_fn = reduce_fn
        counters = (
            [layer.cluster_size.detach().clone() for layer in layers]
            if readonly
            else []
        )
        try:
            yield
        finally:
            with torch.no_grad():
                for layer, value in zip(layers, counters):
                    layer.cluster_size.copy_(value)

    @staticmethod
    def _indices(indices, shape):
        batch, channels, patches = shape
        native = torch.stack(
            [value.reshape(8, batch, channels * patches) for value in indices]
        )
        return native, native.reshape(32, batch, channels, patches)

    def get_tokens(self, x):
        r"""Return native IDs [4,8,B,C*P] and ``indices`` [32,B,C,P].

        ``quantized`` contains four complete [B,128,C,P] feature maps;
        ``code_loss`` and ``usage_ratios`` retain the original reductions.
        Training calls update the native codebooks once.
        """
        patches, temporal, spatial = self._inputs(x)
        with self._ema_context():
            quantized, indices, loss, usage = self.backend.encode(
                patches, temporal, spatial
            )
        native, ordered = self._indices(indices, patches.shape[:3])
        return {
            'token': native,
            'indices': ordered,
            'quantized': quantized,
            'code_loss': loss,
            'usage_ratios': usage,
        }

    def get_codebook_indices(self, x):
        r"""Return ordered IDs [32,B,C,P], aligned with the 32 native heads."""
        return self.get_tokens(x)['indices']

    def forward_pretraining(self, x):
        r"""Return the four original losses, their sum, and reconstructions.

        ``original`` and ``reconstruction`` have shape [B,C,T], standardized
        per sample across all electrodes and times, using native unbiased
        variance. They are not reconstructions in physical voltage units.
        ``amplitude``, ``phase_sin`` and ``phase_cos`` are [B,C*P,200].
        """
        patches, temporal, spatial = self._inputs(x)
        patches = patches.contiguous()
        fft = torch.fft.fft(patches, dim=-1)
        amplitude, amp_mean, amp_std = self.backend.std_norm(torch.log1p(fft.abs()))
        angle = torch.angle(fft)
        with self._ema_context():
            quantized, indices, code_loss, usage = self.backend.encode(
                patches, temporal, spatial
            )
            rec_amp, rec_sin, rec_cos = self.backend.decode(
                quantized, temporal, spatial
            )
            unscaled = torch.expm1(rec_amp.unsqueeze(2) * amp_std + amp_mean).flatten(
                1, 2
            )
            signal = torch.fft.ifft(
                torch.complex(unscaled * rec_cos, unscaled * rec_sin), dim=-1
            ).real
            original, _, _ = self.backend.std_norm(patches)
            reconstruction, _, _ = self.backend.std_norm(
                signal.unsqueeze(2).contiguous()
            )
            signal_loss = self.backend.calculate_signal_rec_loss(
                reconstruction, original
            )
            amplitude_loss = self.backend.calculate_rec_loss(rec_amp, amplitude)
            phase_loss = self.backend.calculate_phase_loss(
                rec_sin, angle.sin(), rec_cos, angle.cos()
            )
        native, ordered = self._indices(indices, patches.shape[:3])
        return {
            'loss': code_loss + amplitude_loss + phase_loss + signal_loss,
            'code_loss': code_loss,
            'amplitude_loss': amplitude_loss,
            'phase_loss': phase_loss,
            'signal_loss': signal_loss,
            'original': original.reshape_as(x),
            'reconstruction': reconstruction.reshape_as(x),
            'amplitude': rec_amp,
            'phase_sin': rec_sin,
            'phase_cos': rec_cos,
            'token': native,
            'indices': ordered,
            'usage_ratios': usage,
        }

    def forward(self, x, return_loss=False):
        r"""Return standardized original and reconstruction, both [B,C,T].

        Set ``return_loss=True`` to return the complete pretraining dictionary
        through ``Module.__call__`` (for example, in a training wrapper).
        """
        if not isinstance(return_loss, bool):
            raise ValueError('return_loss must be boolean.')
        result = self.forward_pretraining(x)
        if return_loss:
            return result
        return result['original'], result['reconstruction']

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        backend,
        channel_names,
        channel_vocab,
        sampling_rate=200,
        track_usage=False,
        prefix='',
    ):
        r"""Strictly load all 505 native states from an explicit local file.

        Accept a flat tensor dictionary or safetensors. No download, source
        import, discarded state or random fallback occurs. Wrapper metadata
        and usage policy are additional states in training checkpoints.
        """
        model = cls(backend, channel_names, channel_vocab, sampling_rate, track_usage)
        if not isinstance(prefix, str):
            raise ValueError('prefix must be a string.')
        if str(checkpoint_path).endswith('.safetensors'):
            from safetensors.torch import load_file

            state = load_file(str(checkpoint_path), device='cpu')
        else:
            state = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        if not isinstance(state, dict) or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError('Require a complete flat native tensor state dictionary.')
        if prefix and any(not key.startswith(prefix) for key in state):
            raise ValueError('Every checkpoint key must use the explicit prefix.')
        state = {key[len(prefix) :]: value for key, value in state.items()}
        expected = backend.state_dict()
        if set(state) != set(expected):
            raise ValueError('Require all 505 native tokenizer states.')
        for name, value in state.items():
            if (
                value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError('Checkpoint shape/type/value mismatch: ' + name)
        backend.load_state_dict(state, strict=True)
        return model
