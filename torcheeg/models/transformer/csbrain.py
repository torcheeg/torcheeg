"""Interfaces for a separately supplied complete CSBrain author backend.

The pinned source and linked weight release have no stated license. Author
implementation code and weights are not bundled or implicitly imported.
"""

import hashlib
import json
from numbers import Integral
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _positive_int


class _CSBrainProjectionView(nn.Sequential):
    """Expose input and original projection without changing any state keys."""

    def forward(self, x):
        return x, super().forward(x)


class CSBrain(nn.Module):
    r"""Complete external CSBrain with explicit electrode and mask conventions.

    Floating raw inputs are [batch,electrodes,time] at 200 Hz, containing
    complete 200-sample patches. All sensors declared by the author backend
    must be present. Neither padding nor electrode interpolation is inferred.
    The adapter adds a parameter-free projection view while retaining every
    original projection parameter and the complete author forward body.

    Args:
        backend (nn.Module): Explicit complete original CSBrain backend.
        channel_names (list or tuple): Ordered physical names of input rows.
        sampling_rate (int): Physical sampling rate, must be 200 Hz.
        backend_channel_names (list or tuple, optional): Ordered physical names
            expected before the backend's internal permutation. Defaults to
            the original 19-channel pretraining montage.
        allow_montage_transfer (bool): Explicitly permit different backend
            montage/topology metadata. (default: :obj:`False`)
    """

    PRETRAINING_CHANNELS = (
        'FP1',
        'FP2',
        'F3',
        'F4',
        'C3',
        'C4',
        'P3',
        'P4',
        'O1',
        'O2',
        'F7',
        'F8',
        'T3',
        'T4',
        'T5',
        'T6',
        'FZ',
        'CZ',
        'PZ',
    )
    PRETRAINING_REGIONS = (0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1)
    PRETRAINING_INDICES = (
        0,
        10,
        2,
        16,
        3,
        11,
        1,
        6,
        18,
        7,
        12,
        14,
        15,
        13,
        8,
        9,
        4,
        17,
        5,
    )

    def __init__(
        self,
        backend,
        channel_names,
        sampling_rate=200,
        backend_channel_names=None,
        allow_montage_transfer=False,
    ):
        super().__init__()
        _positive_int('sampling_rate', sampling_rate)
        if sampling_rate != 200:
            raise ValueError('The released CSBrain protocol requires 200 Hz.')
        if not isinstance(allow_montage_transfer, bool):
            raise ValueError('allow_montage_transfer must be boolean.')
        native_names = (
            self.PRETRAINING_CHANNELS
            if backend_channel_names is None
            else backend_channel_names
        )
        for names in [channel_names, native_names]:
            if (
                not isinstance(names, (list, tuple))
                or not names
                or not all(isinstance(v, str) and v and v.strip() == v for v in names)
                or len({v.casefold() for v in names}) != len(names)
            ):
                raise ValueError(
                    'Require nonempty distinct ordered physical channel names.'
                )
        folded, native_folded = (
            tuple(v.casefold() for v in channel_names),
            tuple(v.casefold() for v in native_names),
        )
        if set(folded) != set(native_folded):
            raise ValueError(
                'All backend electrodes must be present exactly once; no implicit interpolation or missing sensors.'
            )
        if not isinstance(backend, nn.Module):
            raise ValueError(
                'Require an explicitly constructed complete native nn.Module.'
            )
        regions, indices = (
            getattr(backend, 'brain_regions', None),
            getattr(backend, 'sorted_indices', None),
        )
        count = len(native_names)
        if (
            not isinstance(regions, (tuple, list))
            or len(regions) != count
            or not all(
                isinstance(v, Integral) and not isinstance(v, bool) for v in regions
            )
            or not isinstance(indices, (tuple, list))
            or len(indices) != count
            or not all(
                isinstance(v, Integral) and not isinstance(v, bool) for v in indices
            )
            or sorted(indices) != list(range(count))
            or [regions[i] for i in indices] != sorted(regions)
        ):
            raise ValueError(
                'Require a complete coupled channel permutation and sorted region geometry.'
            )
        regions, indices = (
            tuple(int(v) for v in regions),
            tuple(int(v) for v in indices),
        )
        native_geometry = (
            native_folded == tuple(v.casefold() for v in self.PRETRAINING_CHANNELS)
            and regions == self.PRETRAINING_REGIONS
            and indices == self.PRETRAINING_INDICES
        )
        if not native_geometry and not allow_montage_transfer:
            raise ValueError(
                'Use the original pretraining montage/topology or explicitly set allow_montage_transfer=True.'
            )
        expected_area = {}
        start = 0
        for region in sorted(set(regions)):
            size = regions.count(region)
            expected_area['region_' + str(region)] = {
                'channels': size,
                'slice': slice(start, start + size),
            }
            start += size
        if getattr(backend, 'area_config', None) != expected_area:
            raise ValueError(
                'Region slices must match the complete sorted electrode geometry.'
            )
        encoder = getattr(backend, 'encoder', None)
        layers = getattr(encoder, 'layers', [])
        if len(layers) != 12 or getattr(encoder, 'num_layers', None) != 12:
            raise ValueError('Require all twelve complete native encoder layers.')
        for layer in layers:
            if (
                getattr(layer, 'd_model', None) != 200
                or getattr(layer, 'nhead', None) != 8
                or getattr(layer, 'batch_first', None) is not True
                or getattr(layer, 'area_config', None) != expected_area
                or getattr(getattr(layer, 'activation', None), '__name__', None)
                != 'gelu'
            ):
                raise ValueError(
                    'Require original 200-wide/eight-head structured attention and region configuration.'
                )
            for name in ['inter_region_attn', 'inter_window_attn']:
                attention = getattr(layer, name, None)
                if (
                    getattr(attention, 'embed_dim', None) != 200
                    or getattr(attention, 'num_heads', None) != 8
                    or getattr(attention, 'batch_first', None) is not True
                ):
                    raise ValueError(
                        'Keep both complete original structured attention branches.'
                    )
        embedding = getattr(backend, 'patch_embedding', None)
        if getattr(embedding, 'd_model', None) != 200 or tuple(
            getattr(backend, 'TemEmbed_kernel_sizes', [])
        ) != ((1,), (3,), (5,)):
            raise ValueError(
                'Keep the native temporal scales and complete 200-wide tokenizer.'
            )
        state = backend.state_dict()
        expected_shapes = {
            'patch_embedding.mask_encoding': (200,),
            'patch_embedding.positional_encoding.0.weight': (200, 1, 19, 7),
            'patch_embedding.spectral_proj.0.weight': (200, 101),
            'proj_out.0.weight': (200, 200),
            'proj_out.0.bias': (200,),
        }
        if any(
            key not in state or tuple(state[key].shape) != shape
            for key, shape in expected_shapes.items()
        ):
            raise ValueError(
                'Keep all tokenizer, spectral and native reconstruction projection states.'
            )
        for index in range(12):
            for suffix, shape in [
                ('linear1.weight', (800, 200)),
                ('linear2.weight', (200, 800)),
            ]:
                key = 'encoder.layers.' + str(index) + '.' + suffix
                if key not in state or tuple(state[key].shape) != shape:
                    raise ValueError('Keep every complete native feed-forward layer.')
        projection = getattr(backend, 'proj_out', None)
        if (
            not isinstance(projection, nn.Sequential)
            or len(projection) != 1
            or not isinstance(projection[0], nn.Linear)
        ):
            raise ValueError(
                'Require the complete original one-layer reconstruction projection.'
            )
        # The original body returns its final projection directly. This pure
        # container view adds its input to the return value; original Linear
        # objects, computations and state keys remain unchanged. No hooks or
        # temporary forward flags are shared between calls.
        if not isinstance(projection, _CSBrainProjectionView):
            backend.proj_out = _CSBrainProjectionView(*list(projection.children()))
        self.backend = backend
        self.channel_names = tuple(channel_names)
        self.backend_channel_names = tuple(native_names)
        self.num_electrodes, self.sampling_rate, self.patch_size, self.hid_channels = (
            count,
            200,
            200,
            200,
        )
        self.allow_montage_transfer = allow_montage_transfer
        self.canonical_indices = tuple(folded.index(name) for name in native_folded)
        self.output_indices = tuple(native_folded.index(name) for name in folded)
        self.sorted_indices = indices
        self.inverse_sorted_indices = tuple(indices.index(i) for i in range(count))
        self.token_channel_names = tuple(native_names[i] for i in indices)
        geometry = {
            key: {
                'channels': v['channels'],
                'slice': [v['slice'].start, v['slice'].stop, v['slice'].step],
            }
            for key, v in expected_area.items()
        }
        behavior = {
            'names': self.channel_names,
            'backend_names': self.backend_channel_names,
            'regions': regions,
            'indices': indices,
            'area_config': geometry,
            'sampling_rate': 200,
            'patch_size': 200,
            'transfer': allow_montage_transfer,
            'dropouts': [
                (name, module.p)
                for name, module in backend.named_modules()
                if isinstance(module, nn.Dropout)
            ],
            'normalization_eps': [
                (name, module.eps)
                for name, module in backend.named_modules()
                if isinstance(module, (nn.GroupNorm, nn.LayerNorm))
            ],
            'projection_view': 'input_and_original_output',
        }
        self._digest = hashlib.sha256(
            json.dumps(behavior, sort_keys=True).encode()
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
                'CSBrain physical channel order, topology or native configuration differs.'
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
            or x.shape[2] < 200
            or x.shape[2] % 200
            or not torch.isfinite(x).all()
        ):
            raise ValueError(
                'Require finite floating EEG [batch,declared_electrodes,time] with complete 200-sample patches.'
            )
        return x[:, self.canonical_indices].reshape(
            x.shape[0], self.num_electrodes, -1, 200
        )

    def forward(self, x, return_tokens=False):
        r"""Return native pre-projection tokens or their mean [B,200].

        Tokens are [B,C,P,200] at internal sorted positions, named by
        token_channel_names. They are contextual features, not exclusive
        physical sensor reconstructions. The full original reconstruction
        head remains loaded and is still evaluated by the projection view.
        """
        if not isinstance(return_tokens, bool):
            raise ValueError('return_tokens must be boolean.')
        features, _ = self.backend(self._patches(x))
        return features if return_tokens else features.mean((1, 2))

    def forward_reconstruction(self, x):
        r"""Return all native predictions [B,C,P,200] in declared target order.

        Target semantics follow the original training criterion, which pairs
        outputs with the original pre-sort input order. Do not infer output
        target order solely from the internal input permutation.
        """
        _, prediction = self.backend(self._patches(x))
        return prediction[:, self.output_indices]

    def _masked(self, x, mask, native_convention):
        patches = self._patches(x)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.device != x.device
            or mask.shape != patches.shape[:3]
            or not mask.any()
        ):
            raise ValueError(
                'Require a same-device boolean patch mask [B,C,P] selecting at least one target.'
            )
        target_mask = mask if native_convention else mask[:, self.canonical_indices]
        native_mask = mask if native_convention else target_mask[:, self.sorted_indices]
        _, prediction = self.backend(patches, mask=native_mask)
        hidden_mask = native_mask[:, self.inverse_sorted_indices]
        loss = F.mse_loss(prediction[target_mask], patches[target_mask])
        return dict(
            prediction=prediction[:, self.output_indices],
            target_mask=target_mask[:, self.output_indices],
            hidden_mask=hidden_mask[:, self.output_indices],
            native_mask=native_mask,
            loss=loss,
        )

    def forward_masked(self, x, mask):
        r"""Physically mask selected input patches and compute reconstruction MSE.

        mask[B,C,P] follows declared channel_names, True for a hidden input
        and selected target. The adapter maps it through both permutations;
        hidden_mask and target_mask therefore agree. This corrects the
        original trainer's mask/target index mismatch and is a distinct
        training protocol, not an assertion of original-paper reproduction.
        Raw target gradients are retained as in the native criterion.
        """
        return self._masked(x, mask, False)

    def forward_native_masked(self, x, native_mask):
        r"""Reproduce the author's exact original mask and target convention.

        native_mask[B,C,P] is passed unchanged in the backend's native index
        convention. The same indices select original pre-sort targets, as
        in the author trainer. Thus hidden_mask can differ from target_mask;
        a selected target can remain visible. Both returned masks use the
        declared physical input order. Use forward_masked for physically
        matched hiding and targets. This method preserves as-trained math.
        """
        return self._masked(x, native_mask, True)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path,
        backend,
        channel_names,
        sampling_rate=200,
        backend_channel_names=None,
        allow_montage_transfer=False,
        prefix='module.',
        state_key=None,
    ):
        r"""Strictly load every native tensor from an explicit local file.

        Official weights are flat and use prefix='module.'. Complete flat
        exports without that prefix specify prefix=''; nested exports need
        explicit state_key. Uses restricted CPU loading, no unsafe fallback,
        source import, download, partial selection or projection removal.
        """
        model = cls(
            backend,
            channel_names,
            sampling_rate,
            backend_channel_names,
            allow_montage_transfer,
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
        if prefix and any(not k.startswith(prefix) for k in state):
            raise ValueError('Every tensor must use the explicit common prefix.')
        state = {k[len(prefix) :]: v for k, v in state.items()}
        expected = backend.state_dict()
        if set(state) != set(expected):
            raise ValueError(
                'Require every complete native tokenizer, encoder and reconstruction projection tensor.'
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


class CSBrainClassifier(nn.Module):
    r"""A new linear task head over mean native pre-projection features.

    Args:
        backbone (CSBrain): Explicit complete native-backend adapter.
        num_classes (int): New task logit width. (default: :obj:`2`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
    """

    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        _positive_int('num_classes', num_classes)
        if not isinstance(backbone, CSBrain) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require a CSBrain adapter and boolean freeze_backbone.')
        self.backbone, self.freeze_backbone = backbone, freeze_backbone
        self.classifier = nn.Linear(200, num_classes).to(
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
