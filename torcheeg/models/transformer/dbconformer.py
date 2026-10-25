"""Interface for the separately supplied complete DBConformer author model."""

import hashlib
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from ._utils import _positive_int


_SOURCE_LOCK = threading.RLock()


class DBConformer(nn.Module):
    r"""Complete external dual-branch DBConformer with a logits interface.

    The original temporal/spatial embeddings, two Transformer stacks, fusion
    and classifier remain registered in :attr:`backend`. Inputs are floating
    tensors of shape :obj:`[batch, electrodes, time]` or
    :obj:`[batch, 1, electrodes, time]`. Channel order, signal units and
    preprocessing are defined by the caller; no filtering is inferred.

    The author repository does not state a license. Its code and weights are
    supplied separately and are never bundled or downloaded by this class.
    Use :meth:`from_source` with the pinned, trusted author file to construct
    a complete backend with TorchEEG parameter names.

    Args:
        backend (nn.Module): Complete original DBConformer with both branches.
        chunk_size (int): Number of input samples. (default: :obj:`1001`)

    .. code-block:: python

        model = DBConformer.from_source(
            './author/DBConformer.py', num_classes=4)
        logits = model(torch.randn(2, 22, 1001))
        features, logits = model(
            torch.randn(2, 1, 22, 1001), return_features=True)
    """

    SOURCE_REVISION = '02cf63206087045c5cd0e8fa882831102e6fd01b'
    SOURCE_SHA256 = 'b891d8ba28b3ef5064106065e4a2c9bd1e346fe29b6d68cc50c91dbf6cfa4bd1'

    def __init__(self, backend: nn.Module, chunk_size: int = 1001):
        super().__init__()
        _positive_int('chunk_size', chunk_size)
        if not isinstance(backend, nn.Module):
            raise TypeError('backend must be a complete DBConformer nn.Module.')
        required = (
            'embedding',
            'channel_embedding',
            'temporal_transformer',
            'spatial_transformer',
            'classifier',
            'C',
            'D',
            'P',
            'branch',
            'gate_flag',
            'posemb_flag',
            'chn_atten_flag',
        )
        if any(not hasattr(backend, name) for name in required):
            raise ValueError('backend must contain the complete author network.')
        if backend.branch != 'all':
            raise ValueError('Both temporal and spatial branches are required.')
        self.num_electrodes = backend.C
        self.hid_channels = backend.D
        self.chunk_size = chunk_size
        self.patch_size = backend.embedding.stem.downSampling.kernel_size
        if isinstance(self.patch_size, tuple) and len(self.patch_size) == 1:
            self.patch_size = self.patch_size[0]
        self.drop_last_sample = backend.embedding.stem.data_name != 'MI1-7'
        self.num_classes = backend.classifier.fc[-1].out_features
        for name in ('num_electrodes', 'hid_channels', 'patch_size', 'num_classes'):
            _positive_int(name, getattr(self, name))
        patches = (chunk_size - int(self.drop_last_sample)) // self.patch_size
        if patches < 1 or (backend.posemb_flag and patches != backend.P):
            raise ValueError('Input length and native temporal positions disagree.')
        if backend.posemb_flag and (
            tuple(backend.pos_embedding_temporal.shape) != (1, patches, backend.D)
            or tuple(backend.pos_embedding_spatial.shape) != (1, backend.C, backend.D)
        ):
            raise ValueError('Native temporal or spatial positional shape disagrees.')
        self.backend = backend

    @classmethod
    def from_source(
        cls,
        source_path,
        num_electrodes: int = 22,
        chunk_size: int = 1001,
        num_classes: int = 4,
        hid_channels: int = 40,
        temporal_depth: int = 5,
        spatial_depth: int = 5,
        spatial_channels: int = 16,
        patch_size: int = 125,
        fusion: str = 'attention',
        positional_embedding: bool = True,
        drop_last_sample: bool = True,
    ):
        r"""Construct the full pinned author network from an explicit local file.

        This executes caller-supplied Python after checking its exact SHA256.
        Supply a trusted copy of :obj:`models/DBConformer.py` at
        :attr:`SOURCE_REVISION` in the author repository. Install :obj:`timm`
        separately (native tests use :obj:`1.0.10`). Importing TorchEEG does
        not import timm or the author code. Caller cuDNN settings are restored
        after source execution, including when an import fails.

        Args:
            source_path (str or Path): Explicit pinned author Python file.
            num_electrodes (int): Ordered EEG rows. (default: :obj:`22`)
            chunk_size (int): Input samples. (default: :obj:`1001`)
            num_classes (int): Output classes. (default: :obj:`4`)
            hid_channels (int): Embedding size, divisible by the native ten
                attention heads. (default: :obj:`40`)
            temporal_depth (int): Temporal Transformer blocks. (default: :obj:`5`)
            spatial_depth (int): Spatial Transformer blocks. (default: :obj:`5`)
            spatial_channels (int): Spatial convolution filters. (default: :obj:`16`)
            patch_size (int): Temporal average-pool width. (default: :obj:`125`)
            fusion (str): Native :obj:`'attention'`, :obj:`'mean'` concatenation,
                or :obj:`'gated'` aggregation. (default: :obj:`'attention'`)
            positional_embedding (bool): Both native position parameters.
                (default: :obj:`True`)
            drop_last_sample (bool): Crop the final temporal sample before
                pooling, as in the author's non-MI1-7 path. The spatial branch
                still receives every sample. (default: :obj:`True`)
        """
        for name, value in (
            ('num_electrodes', num_electrodes),
            ('chunk_size', chunk_size),
            ('num_classes', num_classes),
            ('hid_channels', hid_channels),
            ('temporal_depth', temporal_depth),
            ('spatial_depth', spatial_depth),
            ('spatial_channels', spatial_channels),
            ('patch_size', patch_size),
        ):
            _positive_int(name, value)
        if hid_channels % 10:
            raise ValueError('hid_channels must be divisible by ten native heads.')
        if fusion not in ('attention', 'mean', 'gated'):
            raise ValueError('fusion must be attention, mean or gated.')
        if not isinstance(positional_embedding, bool) or not isinstance(
            drop_last_sample, bool
        ):
            raise TypeError('positional_embedding and drop_last_sample must be bool.')
        patches = (chunk_size - int(drop_last_sample)) // patch_size
        if patches < 1 or (
            positional_embedding and patches != chunk_size // patch_size
        ):
            raise ValueError('Input length and native temporal positions disagree.')
        path = Path(source_path)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != cls.SOURCE_SHA256:
            raise ValueError('Author source SHA256 does not match the pinned release.')
        module_name = '_torcheeg_dbconformer_author_' + cls.SOURCE_SHA256
        with _SOURCE_LOCK:
            module = sys.modules.get(module_name)
            if module is None:
                module = ModuleType(module_name)
                module.__file__ = str(path)
                benchmark = torch.backends.cudnn.benchmark
                deterministic = torch.backends.cudnn.deterministic
                try:
                    exec(compile(source, str(path), 'exec'), module.__dict__)
                finally:
                    torch.backends.cudnn.benchmark = benchmark
                    torch.backends.cudnn.deterministic = deterministic
                sys.modules[module_name] = module
        args = SimpleNamespace(
            data_name='BNCI2014001' if drop_last_sample else 'MI1-7',
            chn=num_electrodes,
            patch_size=patch_size,
            time_sample_num=chunk_size,
            class_num=num_classes,
            spa_dim=spatial_channels,
            gate_flag=fusion == 'gated',
            posemb_flag=positional_embedding,
            branch='all',
            chn_atten_flag=fusion == 'attention',
        )
        backend = module.DBConformer(
            args,
            emb_size=hid_channels,
            tem_depth=temporal_depth,
            chn_depth=spatial_depth,
            chn=num_electrodes,
            n_classes=num_classes,
        )
        return cls(backend, chunk_size=chunk_size)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        r"""Return class logits, or the unchanged native features/logits pair."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('x must be a floating EEG tensor.')
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1:] != (1, self.num_electrodes, self.chunk_size):
            raise ValueError('Expected [batch,1,num_electrodes,chunk_size] input.')
        features, logits = self.backend(x)
        return (features, logits) if return_features else logits

    def native_state_dict(self):
        """Return every native state without the wrapper's backend prefix."""
        return self.backend.state_dict()

    def load_native_state_dict(self, state_dict, strict: bool = True):
        """Load an author state dictionary, preserving complete key coverage."""
        return self.backend.load_state_dict(state_dict, strict=strict)
