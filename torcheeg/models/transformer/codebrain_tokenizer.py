"""CodeBrain TFDual tokenizer, adapted from Ma et al. (Apache-2.0).

Source22d350caf68246d2fda4f630ef837420db3fb130; see LICENSES/CodeBrain.txt.
"""
import math
import threading
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.nn.functional as F
from torch import nn

from ._utils import _positive_int, _probability
from .labram import Block


_checkpoint_lock = threading.RLock()


class _DiscardNamespace:
    __slots__ = ()

    def __setstate__(self, state):
        pass


class _DiscardDType:
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _discard_scalar(*args):
    return None


def _discard_scalar_new(*args):
    return None


def _metadata_bytes(value, encoding):
    # Protocol-2 NumPy scalar metadata stores bytes through this Latin-1 codec.
    if type(value) is not str or encoding != 'latin1':
        raise ValueError('Checkpoint metadata only supports Latin-1 byte strings.')
    return value.encode('latin1')


# These inert replacements discard training-only metadata. They do not call
# NumPy reconstruction functions, create dtypes or invoke argparse constructors.
# Naming aliases also work with the original safe_globals API in PyTorch2.4.
for _value, _module, _name in [(_DiscardNamespace, 'argparse', 'Namespace'),
                              (_DiscardDType, 'numpy', 'dtype'),
                              (_discard_scalar, 'numpy.core.multiarray', 'scalar'),
                              (_discard_scalar_new, 'numpy._core.multiarray', 'scalar'),
                              (_metadata_bytes, '_codecs', 'encode')]:
    _value.__module__ = _module
    _value.__name__ = _value.__qualname__ = _name


def _read_tensor_state(path):
    if Path(path).suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError('Install torcheeg[foundation] to load safetensors.') from error
        return load_file(str(path), device='cpu')
    metadata = [_DiscardNamespace, _DiscardDType, _discard_scalar, _discard_scalar_new, _metadata_bytes]
    context = torch.serialization.safe_globals(metadata) if hasattr(torch.serialization, 'safe_globals') else nullcontext()
    with _checkpoint_lock, context:
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
    return checkpoint


class _Convolutions(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 8, (1, 15), stride=(1, 8), padding=(0, 7)),
            nn.BatchNorm2d(8), nn.ReLU(), nn.Dropout(.3),
            nn.Conv2d(8, 4, (1, 3), padding=(0, 1)), nn.BatchNorm2d(4), nn.ReLU(), nn.Dropout(.3),
            nn.Conv2d(4, 4, (1, 3), padding=(0, 1)), nn.BatchNorm2d(4), nn.ReLU(), nn.Dropout(.3))

    def forward(self, x):
        return self.conv(x)


class _DualConvolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_convs = _Convolutions()
        self.f_convs = _Convolutions()

    def forward(self, x):
        batch, channels, patches, samples = x.shape
        frequency = torch.fft.fft(x, dim=-1).real
        temporal = self.t_convs(x.reshape(batch, 1, channels * patches, samples))
        frequency = self.f_convs(frequency.reshape(batch, 1, channels * patches, samples))
        temporal = temporal.permute(0, 2, 3, 1).reshape(batch, channels * patches, 100)
        frequency = frequency.permute(0, 2, 3, 1).reshape(batch, channels * patches, 100)
        return torch.cat([temporal, frequency], dim=-1)


class _DecoderPatch(nn.Module):
    def __init__(self, code_dim):
        super().__init__()
        self.proj = nn.Conv2d(code_dim, 200, 1)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _TokenizerTransformer(nn.Module):
    def __init__(self, depth, code_dim=None):
        super().__init__()
        self.is_decoder = code_dim is not None
        self.patch_embed = _DecoderPatch(code_dim) if self.is_decoder else _DualConvolution()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 200))
        self.pos_embed = nn.Parameter(torch.zeros(1, 129, 200))
        self.time_embed = nn.Parameter(torch.zeros(1, 60, 200))
        self.pos_drop = nn.Dropout(0)
        self.blocks = nn.ModuleList([Block(dim=200, num_heads=10, mlp_ratio=4, qkv_bias=True,
            qk_norm=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.) for _ in range(depth)])
        self.norm = nn.Identity()
        self.fc_norm = nn.LayerNorm(200, eps=1e-6)
        self.head = nn.Identity()
        self.apply(_initialize)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.time_embed, std=.02)
        with torch.no_grad():
            for index, block in enumerate(self.blocks):
                block.attn.proj.weight.div_(math.sqrt(2 * (index + 1)))
                block.mlp.fc2.weight.div_(math.sqrt(2 * (index + 1)))

    def forward(self, x, channel_ids):
        if self.is_decoder:
            batch, _, channels, patches = x.shape
        else:
            batch, channels, patches, _ = x.shape
        tokens = self.patch_embed(x)
        tokens = torch.cat([self.cls_token.expand(batch, -1, -1), tokens], dim=1)
        positions = self.pos_embed[:, channel_ids].unsqueeze(2).expand(batch, -1, patches, -1).flatten(1, 2)
        positions = torch.cat([self.pos_embed[:, :1].expand(batch, -1, -1), positions], dim=1)
        tokens = tokens + positions
        temporal = self.time_embed[:, :patches].unsqueeze(1).expand(batch, channels, -1, -1).flatten(1, 2)
        tokens = torch.cat([tokens[:, :1], tokens[:, 1:] + temporal], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.fc_norm(self.norm(tokens)[:, 1:])


def _initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def _sample_vectors(samples, count):
    indices = torch.randperm(len(samples), device=samples.device)[:count] if len(samples) >= count else torch.randint(
        0, len(samples), (count,), device=samples.device)
    return samples[indices]


@torch.no_grad()
def _kmeans(samples, count):
    means = _sample_vectors(samples, count)
    for _ in range(10):
        buckets = (samples @ means.t()).max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=count)
        empty = bins == 0
        updated = samples.new_zeros(count, samples.shape[-1])
        updated.scatter_add_(0, buckets[:, None].expand(-1, samples.shape[-1]), samples)
        updated = F.normalize(updated / bins.masked_fill(empty, 1)[:, None], p=2, dim=-1)
        means = torch.where(empty[:, None], means, updated)
    return means, bins


class _EmbeddingEMA(nn.Module):
    def __init__(self, count, code_dim, kmeans_init, update):
        super().__init__()
        weight = torch.zeros(count, code_dim) if kmeans_init else F.normalize(torch.randn(count, code_dim), dim=-1)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(count), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.register_buffer('initted', torch.tensor([float(not kmeans_init)]))
        self.update = update
        self._initialized = not kmeans_init

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._initialized = bool(self.initted.item())

    @torch.no_grad()
    def initialize(self, samples):
        if not self._initialized:
            synchronize = distributed.is_available() and distributed.is_initialized()
            if not synchronize or distributed.get_rank() == 0:
                means, counts = _kmeans(samples, self.weight.shape[0])
                self.weight.copy_(means)
                self.cluster_size.copy_(counts)
            if synchronize:
                # Every rank must attach the same meaning to each token ID.
                distributed.broadcast(self.weight, src=0)
                distributed.broadcast(self.cluster_size, src=0)
            self.initted.fill_(1)
            self._initialized = True

    def forward(self, indices):
        return F.embedding(indices, self.weight)


class _NormQuantizer(nn.Module):
    def __init__(self, count, code_dim, decay, kmeans_init, update, track_usage):
        super().__init__()
        self.embedding = _EmbeddingEMA(count, code_dim, kmeans_init, update)
        self.register_buffer('cluster_size', torch.zeros(count))
        self.decay = decay
        self.track_usage = track_usage

    def forward(self, z):
        z = F.normalize(z.permute(0, 2, 3, 1), p=2, dim=-1)
        flattened = z.reshape(-1, z.shape[-1])
        self.embedding.initialize(flattened)
        distances = flattened.pow(2).sum(dim=1, keepdim=True) + self.embedding.weight.pow(2).sum(dim=1) - 2 * torch.einsum(
            'bd,nd->bn', flattened, self.embedding.weight)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(z.shape)
        counts = F.one_hot(indices, self.embedding.weight.shape[0]).to(z.dtype)
        if self.track_usage or (self.training and self.embedding.update):
            with torch.no_grad():
                bins = counts.sum(0)
                _distributed_sum(bins)
                if self.track_usage:
                    self.cluster_size.mul_(self.decay).add_(bins, alpha=1-self.decay)
                if self.training and self.embedding.update:
                    empty = bins == 0
                    totals = flattened.t() @ counts
                    _distributed_sum(totals)
                    updated = F.normalize((totals / bins.masked_fill(empty, 1).unsqueeze(0)).t(), p=2, dim=-1)
                    updated = torch.where(empty[:, None], self.embedding.weight, updated)
                    self.embedding.weight.mul_(self.decay).add_(updated, alpha=1-self.decay)
                    self.embedding.weight.copy_(F.normalize(self.embedding.weight, p=2, dim=-1))
        loss = F.mse_loss(quantized.detach(), z)
        quantized = z + (quantized-z).detach()
        return quantized.permute(0, 3, 1, 2), loss, indices


def _distributed_sum(tensor):
    # Process groups may be initialized after the module is constructed.
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(tensor)


def _contrastive_loss(first, second, mode):
    first, second = first.flatten(1), second.flatten(1)
    features = torch.cat([second, first], dim=0)
    similarities = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
    batch = first.shape[0]
    if mode == 'infonce':
        # A fixed-size logits matrix avoids dynamic boolean-indexed shapes
        # during compilation. Only self-similarities are excluded here; the
        # paired positive is the target class and appears exactly once.
        diagonal = torch.eye(2*batch, device=features.device, dtype=torch.bool)
        logits = similarities.masked_fill(diagonal, float('-inf')) / .1
        labels = (torch.arange(2*batch, device=features.device) + batch) % (2*batch)
        return F.cross_entropy(logits, labels, reduction='sum') / (2*batch)
    positive = torch.cat([similarities.diag(batch), similarities.diag(-batch)]).reshape(2*batch, 1)
    eye = torch.eye(2*batch, device=features.device)
    # The author subtracts the identical +/-batch masks twice, leaving
    # negative ones that become True and include positive pairs again.
    mask = (1-eye-eye.roll(batch, dims=1)-eye.roll(-batch, dims=1)).bool()
    negative = similarities[mask].reshape(2*batch, -1)
    logits = torch.cat([positive, negative], dim=1) / .1
    return F.cross_entropy(logits, torch.zeros(2*batch, dtype=torch.long, device=features.device), reduction='sum') / (2*batch)


def _validate_eeg(x):
    if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 4 or min(x.shape) < 1 or
            x.shape[-1] != 200 or x.shape[2] > 60):
        raise ValueError('EEG must be nonempty floating [batch,electrodes,1..60patches,200].')


class CodeBrainTokenizer(nn.Module):
    r'''Complete CodeBrain TFDual time/frequency tokenizer and reconstruction objective.

    Input [batch,electrodes,patches,200] uses nonoverlapping one-second patches
    at200Hz. Native positional rows are numeric IDs, not a named montage.
    Default rows1..electrodes reproduce the author's sequential assignment;
    explicit channel_ids selects one distinct row1..128 per electrode. No
    resampling, scaling or channel-name mapping occurs inside the model.

    Args:
        encoder_depth (int): Transformer encoder blocks. (default: :obj:`12`)
        decoder_depth (int): Blocks per domain decoder. (default: :obj:`3`)
        code_dim (int): Quantized latent width. (default: :obj:`32`)
        num_time_tokens (int): Time vocabulary size. (default: :obj:`4096`)
        num_frequency_tokens (int): Frequency vocabulary size. (default: :obj:`4096`)
        decay (float): Codebook/usage EMA decay. (default: :obj:`0.99`)
        quantize_kmeans_init (bool): Initialize new codebooks from data. (default: :obj:`True`)
        update_codebook (bool): Update codebook weights in train mode. (default: :obj:`True`)
        track_usage (bool): Track EMA counts, including eval as native. (default: :obj:`True`)
        contrastive_mode (str): Correct paired InfoNCE, or native positive-pair
            duplication for source reproduction. (default: :obj:`'infonce'`)
    '''
    def __init__(self, encoder_depth=12, decoder_depth=3, code_dim=32, num_time_tokens=4096,
                 num_frequency_tokens=4096, decay=.99, quantize_kmeans_init=True,
                 update_codebook=True, track_usage=True, contrastive_mode='infonce'):
        super().__init__()
        for name, value in [('encoder_depth', encoder_depth), ('decoder_depth', decoder_depth), ('code_dim', code_dim),
                            ('num_time_tokens', num_time_tokens), ('num_frequency_tokens', num_frequency_tokens)]:
            _positive_int(name, value)
        _probability('decay', decay)
        if not all(isinstance(value, bool) for value in [quantize_kmeans_init, update_codebook, track_usage]):
            raise ValueError('Codebook initialization, update and usage switches must be boolean.')
        if contrastive_mode not in ('infonce', 'native'):
            raise ValueError('contrastive_mode must be infonce or native.')
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.code_dim = code_dim
        self.num_time_tokens = num_time_tokens
        self.num_frequency_tokens = num_frequency_tokens
        self.decay = decay
        self.quantize_kmeans_init = quantize_kmeans_init
        self.update_codebook = update_codebook
        self.track_usage = track_usage
        self.contrastive_mode = contrastive_mode
        self.encoder = _TokenizerTransformer(encoder_depth)
        self.decoder_t = _TokenizerTransformer(decoder_depth, code_dim)
        self.decoder_f = _TokenizerTransformer(decoder_depth, code_dim)
        self.quantize_t = _NormQuantizer(num_time_tokens, code_dim, decay, quantize_kmeans_init, update_codebook, track_usage)
        self.quantize_f = _NormQuantizer(num_frequency_tokens, code_dim, decay, quantize_kmeans_init, update_codebook, track_usage)
        self.encode_task_layer_t = nn.Sequential(nn.Linear(200, 200), nn.Tanh(), nn.Linear(200, code_dim))
        self.encode_task_layer_f = nn.Sequential(nn.Linear(200, 200), nn.Tanh(), nn.Linear(200, code_dim))
        self.decode_task_layer_temp = nn.Sequential(nn.Linear(200, 200), nn.Tanh(), nn.Linear(200, 200))
        self.decode_task_layer_xrec = nn.Sequential(nn.Linear(200, 200), nn.Tanh(), nn.Linear(200, 200))
        self.decode_task_layer_angle = nn.Sequential(nn.Linear(200, 200), nn.Tanh(), nn.Linear(200, 200))
        for module in [self.encode_task_layer_t, self.encode_task_layer_f, self.decode_task_layer_temp,
                       self.decode_task_layer_xrec, self.decode_task_layer_angle]:
            module.apply(_initialize)

    def _ids(self, channel_ids, channels, device):
        if not 1 <= channels <= 128:
            raise ValueError('Require between1 and128 electrodes.')
        if channel_ids is None:
            return torch.arange(1, channels+1, device=device)
        if isinstance(channel_ids, (list, tuple)):
            if (len(channel_ids) != channels or any(isinstance(value, bool) or not isinstance(value, int) for value in channel_ids) or
                    len(set(channel_ids)) != channels or min(channel_ids) < 1 or max(channel_ids) > 128):
                raise ValueError('channel_ids must give one unique integer row1..128 per electrode.')
            return torch.tensor(channel_ids, device=device)
        if (not isinstance(channel_ids, torch.Tensor) or channel_ids.dtype not in (torch.int32, torch.int64) or
                channel_ids.shape != (channels,) or channel_ids.unique().numel() != channels or
                channel_ids.min() < 1 or channel_ids.max() > 128):
            raise ValueError('channel_ids must give one unique integer row1..128 per electrode.')
        return channel_ids.to(device=device, dtype=torch.long)

    def encode(self, x, channel_ids=None, contrastive=None):
        r'''Return quantized time/IDs/loss, quantized frequency/IDs/loss and contrastive loss.

        Quantized tensors are [batch,code_dim,electrodes,patches]; IDs are
        flattened in batch/electrode/patch order. contrastive=None follows
        native behavior: enabled whenever gradients are enabled, independently
        of train/eval mode. True requires an even patch count of at least2.
        False allows one/odd-patch tokenization and reconstruction with gradients.
        '''
        _validate_eeg(x)
        if contrastive is not None and not isinstance(contrastive, bool):
            raise ValueError('contrastive must be boolean or None.')
        contrastive = torch.is_grad_enabled() if contrastive is None else contrastive
        if contrastive and (x.shape[2] < 2 or x.shape[2] % 2):
            raise ValueError('Contrastive loss requires an even patch count of at least2; use contrastive=False otherwise.')
        ids = self._ids(channel_ids, x.shape[1], x.device)
        features = self.encoder(x, ids)
        if contrastive:
            first = self.encoder(x[:, :, :x.shape[2]//2], ids)
            second = self.encoder(x[:, :, x.shape[2]//2:], ids)
            contrastive_loss = _contrastive_loss(first, second, self.contrastive_mode)
        else:
            contrastive_loss = x.new_zeros(())
        with torch.autocast(device_type=x.device.type, enabled=False):
            time = self.encode_task_layer_t(features.to(self.encode_task_layer_t[-1].weight.dtype))
            frequency = self.encode_task_layer_f(features.to(self.encode_task_layer_f[-1].weight.dtype))
        shape = (x.shape[0], x.shape[1], x.shape[2], self.code_dim)
        time, time_loss, time_ids = self.quantize_t(time.reshape(shape).permute(0, 3, 1, 2))
        frequency, frequency_loss, frequency_ids = self.quantize_f(frequency.reshape(shape).permute(0, 3, 1, 2))
        return time, time_ids, time_loss, frequency, frequency_ids, frequency_loss, contrastive_loss

    def _decode(self, quantized, channel_ids, domain):
        if (not isinstance(quantized, torch.Tensor) or not quantized.is_floating_point() or quantized.ndim != 4 or
                min(quantized.shape) < 1 or quantized.shape[1] != self.code_dim or quantized.shape[3] > 60):
            raise ValueError('Quantized input must be floating [batch,code_dim,electrodes,1..60patches].')
        ids = self._ids(channel_ids, quantized.shape[2], quantized.device)
        return (self.decoder_t if domain == 'time' else self.decoder_f)(quantized, ids)

    def decode_t(self, quantized, channel_ids=None):
        r'''Return raw reconstruction [batch,electrodes*patches,200].'''
        return self.decode_task_layer_temp(self._decode(quantized, channel_ids, 'time'))

    def decode_f(self, quantized, channel_ids=None):
        r'''Return standardized FFT amplitude/phase predictions [batch,electrodes*patches,200].'''
        features = self._decode(quantized, channel_ids, 'frequency')
        return self.decode_task_layer_xrec(features), self.decode_task_layer_angle(features)

    def get_tokens(self, x, channel_ids=None, contrastive=False):
        r'''Return token_t/token_f [batch,electrodes*patches] and quantized latents.

        Token getters skip the unused contrastive term by default, including
        when gradients are enabled. encode/forward retain explicit loss control.
        '''
        time, time_ids, _, frequency, frequency_ids, _, _ = self.encode(x, channel_ids, contrastive)
        return dict(token_t=time_ids.reshape(x.shape[0], -1), token_f=frequency_ids.reshape(x.shape[0], -1),
                    quantize_t=time.permute(0, 2, 3, 1).flatten(1, 2),
                    quantize_f=frequency.permute(0, 2, 3, 1).flatten(1, 2), input_img=x)

    def get_codebook_indices(self, x, channel_ids=None, contrastive=False):
        r'''Return the two flattened [batch,electrodes*patches] token-ID tensors.'''
        output = self.get_tokens(x, channel_ids, contrastive)
        return output['token_t'], output['token_f']

    @staticmethod
    def _standardize(x):
        return (x-x.mean(dim=(1, 2, 3), keepdim=True)) / x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)

    def forward(self, x, channel_ids=None, contrastive=None):
        r'''Return the complete six-term self-supervised loss and detached log.

        Terms: two quantization losses, raw/amplitude/phase reconstruction,
        and optional temporal-half contrastive loss. FFT target standard
        deviations are clamped at1e-6 to keep constant-signal targets finite.
        Correct InfoNCE excludes the paired positive from negatives. Select
        contrastive_mode='native' to reproduce the source's duplicated positive.
        '''
        _validate_eeg(x)
        transformed = torch.fft.fft(x, dim=-1)
        amplitude, phase = self._standardize(transformed.abs()), self._standardize(transformed.angle())
        time, _, time_loss, frequency, _, frequency_loss, contrastive_loss = self.encode(x, channel_ids, contrastive)
        raw = self.decode_t(time, channel_ids)
        amplitude_prediction, phase_prediction = self.decode_f(frequency, channel_ids)
        raw_loss = F.mse_loss(raw, x.flatten(1, 2))
        amplitude_loss = F.mse_loss(amplitude_prediction, amplitude.flatten(1, 2))
        phase_loss = F.mse_loss(phase_prediction, phase.flatten(1, 2))
        terms = [time_loss, frequency_loss, raw_loss, amplitude_loss, phase_loss, contrastive_loss]
        loss = sum(terms)
        prefix = 'train' if self.training else 'val'
        names = ['quant_t_loss', 'quant_f_loss', 'rec_raw_loss', 'rec_spec_loss', 'rec_angle_loss', 'cl_loss']
        log = {f'{prefix}/{name}': value.detach().mean() for name, value in zip(names, terms)}
        log[f'{prefix}/total_loss'] = loss.detach().mean()
        return loss, log

    @classmethod
    def from_pretrained(cls, checkpoint_path, prefix='', update_codebook=True, track_usage=True,
                        contrastive_mode='infonce'):
        r'''Load every official tokenizer tensor strictly on CPU, including both decoders.

        Safetensors, flat tensor states and model/state_dict-wrapped tensor
        states are supported. The original training file's known Namespace
        and NumPy scalar metadata are discarded by inert safe-global
        replacements; no original metadata constructor or NumPy function runs.
        Unknown globals remain rejected, with no unsafe loading fallback.
        Original metadata support requires torch.serialization.safe_globals;
        older runtimes can load a tensor-only/safetensors export instead.
        '''
        if not isinstance(prefix, str) or not isinstance(update_codebook, bool) or not isinstance(track_usage, bool):
            raise ValueError('Require a string prefix and boolean update/usage switches.')
        state = _read_tensor_state(Path(checkpoint_path))
        if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError('Require a complete tensor state dictionary.')
        selected = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
        model = cls(update_codebook=update_codebook, track_usage=track_usage, contrastive_mode=contrastive_mode)
        expected = model.state_dict()
        if set(selected) != set(expected):
            raise ValueError(f'Incomplete CodeBrain tokenizer: missing={sorted(set(expected)-set(selected))}, '
                             f'extra={sorted(set(selected)-set(expected))}.')
        for name, value in selected.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError(f'Checkpoint tensor shape/type mismatch: {name}.')
            if not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor contains nonfinite values: {name}.')
        if any(selected[f'quantize_{domain}.embedding.initted'].item() != 1 for domain in ['t', 'f']):
            raise ValueError('Pretrained codebooks must already be initialized; use the constructor for new training.')
        model.load_state_dict(selected, strict=True)
        return model
