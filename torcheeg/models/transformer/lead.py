"""TorchEEG interface for an explicitly supplied, locally installed LEAD backend.

No author model code is bundled here. DL4mHealth/LEAD source and weights
are CC-BY-NC-SA-4.0 and must be obtained separately under their own terms.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from ._utils import _positive_int


class LEAD(nn.Module):
    r'''LEAD features through an explicitly supplied complete native backend.

    The caller constructs a trusted local author backend. It must expose
    encode(time_major_eeg, label_metadata), d_model, patch_len, stride and
    the dataset-aware nonpersistent coordinate buffers of the documented
    LEAD implementation. This adapter neither imports nor downloads source.
    Official source and weights retain their CC-BY-NC-SA-4.0 terms.

    Input is floating [batch,electrodes,samples]. The adapter supplies neutral
    disease/subject metadata and explicit physical rate/dataset ID. It saves
    coordinates and an ordered-channel fingerprint, which the native state
    omits, so later MNE geometry changes cannot silently change a restore.

    Args:
        backend (nn.Module): Explicitly constructed local LEAD backend.
        channel_names (list or tuple): Ordered electrode names for that backend.
        sampling_rate (int): Physical input sampling rate in Hz. (default: :obj:`200`)
        dataset_id (int): Native coordinate-map identifier. (default: :obj:`0`)
    '''
    def __init__(self, backend, channel_names, sampling_rate=200, dataset_id=0):
        super().__init__()
        if not isinstance(backend, nn.Module) or not callable(getattr(backend, 'encode', None)):
            raise ValueError('Require an explicit nn.Module backend with encode(eeg, metadata).')
        if not isinstance(channel_names, (list, tuple)) or not channel_names or not all(
            isinstance(name, str) and name.strip() for name in channel_names) or len(set(channel_names)) != len(channel_names):
            raise ValueError('Require nonempty distinct ordered channel names.')
        _positive_int('sampling_rate', sampling_rate)
        if isinstance(dataset_id, bool) or not isinstance(dataset_id, int) or dataset_id < 0:
            raise ValueError('dataset_id must be a nonnegative integer.')
        for name in ['d_model', 'patch_len', 'stride']:
            _positive_int('backend.'+name, getattr(backend, name, None))
        embedding = getattr(backend, 'enc_embedding', None)
        mapping = getattr(embedding, 'coord_buffer_names', {})
        if dataset_id not in mapping:
            raise ValueError('The backend must contain the configured dataset coordinate map.')
        coordinate_name = mapping[dataset_id]
        coords = getattr(embedding, coordinate_name, None)
        if coordinate_name not in embedding._buffers or not isinstance(coords, torch.Tensor) or coords.shape != (len(channel_names), 3) or not coords.is_floating_point() or not torch.isfinite(coords).all():
            raise ValueError('Backend coordinates must be finite [electrodes,3] in the declared order.')
        if getattr(embedding, 'use_sampling_embedding', False):
            rates = getattr(embedding, 'sr_embeddings', None)
            if rates is None or 'hz_'+str(sampling_rate) not in rates:
                raise ValueError('The backend lacks the requested sampling-rate embedding.')
        self.backend, self.channel_names = backend, tuple(channel_names)
        self.sampling_rate, self.dataset_id = sampling_rate, dataset_id
        self.hid_channels = backend.d_model
        self.num_electrodes = len(channel_names)
        self._coordinate_name = coordinate_name
        fingerprint = hashlib.sha256(json.dumps(self.channel_names, ensure_ascii=True).encode()).digest()
        self._channel_digest = fingerprint
        self.register_buffer('channel_fingerprint', torch.tensor(list(fingerprint), dtype=torch.uint8))
        self.register_buffer('label_metadata', torch.tensor([[0, 0, sampling_rate, dataset_id]], dtype=torch.long))
        self.register_buffer('electrode_coords', coords.detach().clone())
        self._bind_geometry()

    def _bind_geometry(self):
        # Share the persistent adapter copy with the native nonpersistent buffer.
        self.backend.enc_embedding._buffers[self._coordinate_name] = self.electrode_coords

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for name in ['channel_fingerprint', 'label_metadata']:
            value = state_dict.get(prefix+name)
            # Python configuration remains defined after Module.to_empty().
            expected = torch.tensor(list(self._channel_digest), dtype=torch.uint8) if name == 'channel_fingerprint' else torch.tensor(
                [[0, 0, self.sampling_rate, self.dataset_id]], dtype=torch.long)
            if isinstance(value, torch.Tensor) and (value.dtype != expected.dtype or not torch.equal(value.cpu(), expected.cpu())):
                error_msgs.append('LEAD '+name+' differs from the configured channel order or physical metadata.')
        coordinates = state_dict.get(prefix+'electrode_coords')
        if isinstance(coordinates, torch.Tensor) and (not coordinates.is_floating_point() or not torch.isfinite(coordinates).all()):
            error_msgs.append('LEAD electrode coordinates must be finite.')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._bind_geometry()

    def forward(self, x):
        r'''Return contextual features [batch,hid_channels] without labels.'''
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or not x.is_floating_point():
            raise ValueError('Require floating EEG[batch,electrodes,samples].')
        if x.shape[0] < 1 or x.shape[1] != self.num_electrodes or x.shape[2] < 1:
            raise ValueError('Require nonempty EEG with the configured ordered electrodes.')
        self._bind_geometry()
        metadata = self.label_metadata.expand(x.shape[0], -1).to(x.device)
        features = self.backend.encode(x.transpose(1, 2), metadata)
        if not isinstance(features, torch.Tensor) or features.shape != (x.shape[0], self.hid_channels):
            raise ValueError('The backend must return one feature vector per sample.')
        return features

    @classmethod
    def from_pretrained(cls, checkpoint_path, backend, channel_names, sampling_rate=200, dataset_id=0, prefix=''):
        r'''Strictly load every local native-backend tensor, then bind metadata.

        Backend construction is an explicit caller action. This method only
        reads a flat tensor checkpoint on CPU with weights_only=True; it does
        not import source, execute checkpoint code, download files, skip a
        head or resize embeddings. The backend must match the complete file.
        '''
        if not isinstance(backend, nn.Module) or not isinstance(prefix, str):
            raise ValueError('Require an explicit backend and string prefix.')
        model = cls(backend, channel_names, sampling_rate, dataset_id)
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
            raise ValueError('Require a complete flat native tensor state dictionary.')
        if prefix and any(not key.startswith(prefix) for key in state):
            raise ValueError('Every key must use the explicit prefix.')
        selected = {name[len(prefix):]: value for name, value in state.items()}
        expected = backend.state_dict()
        if set(selected) != set(expected):
            raise ValueError(f'Incomplete LEAD backend state: missing={sorted(set(expected)-set(selected))}, '
                             f'extra={sorted(set(selected)-set(expected))}.')
        for name, value in selected.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
                raise ValueError(f'Checkpoint tensor shape/type/value mismatch: {name}.')
        backend.load_state_dict(selected, strict=True)
        return model


class LEADClassifier(nn.Module):
    r'''New classification head over complete externally supplied LEAD features.

    Args:
        backbone (LEAD): Backend adapter with explicit geometry and metadata.
        num_classes (int): New task logit width. (default: :obj:`2`)
        freeze_backbone (bool): Train only the new head. (default: :obj:`False`)
    '''
    def __init__(self, backbone, num_classes=2, freeze_backbone=False):
        super().__init__()
        if not isinstance(backbone, LEAD) or not isinstance(freeze_backbone, bool):
            raise ValueError('Require a LEAD adapter and boolean freeze_backbone.')
        _positive_int('num_classes', num_classes)
        self.backbone, self.freeze_backbone = backbone, freeze_backbone
        self.classifier = nn.Linear(backbone.hid_channels, num_classes).to(backbone.electrode_coords)
        if freeze_backbone:
            backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        return self.classifier(self.backbone(x))
