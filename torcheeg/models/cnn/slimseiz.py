"""Interface for the complete separately supplied SlimSeiz neural network."""

import hashlib
import json
import math
from pathlib import Path
import sys
import threading
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F


_SOURCE_LOCK = threading.RLock()


class SlimSeiz(nn.Module):
    r"""Complete external SlimSeiz CNN and selective state-space network.

    Retains both multiscale convolutional stages, the complete pure-PyTorch
    Mamba mixer, RMS normalization, residual, temporal mean and binary head.
    Input is floating :obj:`[batch, electrodes, time]` or
    :obj:`[batch, 1, electrodes, time]`; output is two categorical logits.
    Channel selection, physical sampling rate, units, filtering and the
    preictal/interictal protocol are supplied explicitly by the caller.

    The author repository does not state a license. Supply its complete
    trusted notebook separately; no source or weights are bundled/downloaded.
    :meth:`from_source` verifies the notebook before executing only its two
    complete neural-definition cells. Data loading and training cells are
    not executed. This class exposes the neural network, not the separate
    channel-selection or supervised-contrastive training pipeline.

    Args:
        backend (nn.Module): Complete original :obj:`OneDCNN`, including
            its original two-class classifier and 32-dimensional features.
        chunk_size (int): Input time points, at least 32. (default: 1024)
        return_features (bool): Return :obj:`(logits, features)`, as in the
            complete author model. (default: False)

    .. code-block:: python

        model = SlimSeiz.from_source('./author/Loop_ADDNet_kFold.ipynb')
        logits = model(torch.randn(2, 8, 1024))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
    """

    SOURCE_REVISION = '84b168278173a0e14163a6839fb36d7c41bc3c52'
    SOURCE_SHA256 = 'a77807eee44b3a95780c12f656091172467dd7f967be942061d6cf417921d2a9'
    SOURCE_CELLS = (1, 7)

    @staticmethod
    def _positive_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')

    def __init__(
        self,
        backend: nn.Module,
        chunk_size: int = 1024,
        return_features: bool = False,
    ):
        super().__init__()
        self._positive_int('chunk_size', chunk_size)
        if chunk_size < 32:
            raise ValueError(
                'The complete native pooling requires at least 32 samples.'
            )
        if not isinstance(return_features, bool):
            raise TypeError('return_features must be a boolean.')
        if not isinstance(backend, nn.Module):
            raise TypeError('backend must be the complete SlimSeiz nn.Module.')
        required = (
            'conv1',
            'conv2_1',
            'conv2_2',
            'pool3',
            'conv4_1',
            'conv4_2',
            'mixer',
            'norm',
            'adaptive_avg_pool',
            'classifier',
            'input_channels',
        )
        if any(not hasattr(backend, name) for name in required):
            raise ValueError(
                'Provide the complete original OneDCNN including its head.'
            )
        if (
            not isinstance(backend.classifier, nn.Sequential)
            or len(backend.classifier) != 1
            or not isinstance(backend.classifier[0], nn.Linear)
            or backend.classifier[0].in_features != 32
            or backend.classifier[0].out_features != 2
        ):
            raise ValueError(
                'The full native classifier maps 32 features to two classes.'
            )
        self.num_electrodes = backend.input_channels
        self._positive_int('num_electrodes', self.num_electrodes)
        self.num_classes = 2
        self.chunk_size = chunk_size
        self.return_features = return_features
        self.backend = backend

    @classmethod
    def from_source(
        cls,
        source_path,
        num_electrodes: int = 8,
        chunk_size: int = 1024,
        num_classes: int = 2,
        return_features: bool = False,
    ):
        r"""Construct the complete original model from an explicit local notebook.

        Args:
            source_path (str or Path): Trusted raw
                :obj:`Loop_ADDNet_kFold.ipynb` at :attr:`SOURCE_REVISION`.
                Re-saving a notebook changes its hash and is rejected.
            num_electrodes (int): Explicit selected EEG rows. (default: 8)
            chunk_size (int): Samples per window, at least 32. (default: 1024)
            num_classes (int): The complete original head supports exactly
                two classes. (default: 2)
            return_features (bool): Return native logits and features.
                (default: False)

        Only complete cells 1 (MambaBlock/RMSNorm) and 7 (OneDCNN) execute,
        without editing their bytes or replacing any neural computation.
        Their neural dependencies are PyTorch and einops with einsum support.
        The unrelated notebook data/training imports and cells are not run.
        """
        for name, value in (
            ('num_electrodes', num_electrodes),
            ('chunk_size', chunk_size),
            ('num_classes', num_classes),
        ):
            cls._positive_int(name, value)
        if chunk_size < 32:
            raise ValueError(
                'The complete native pooling requires at least 32 samples.'
            )
        if num_classes != 2:
            raise ValueError('The complete native head supports exactly two classes.')
        if not isinstance(return_features, bool):
            raise TypeError('return_features must be a boolean.')
        path = Path(source_path)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != cls.SOURCE_SHA256:
            raise ValueError('Notebook SHA256 does not match the pinned author source.')
        notebook = json.loads(source)
        namespace = '_torcheeg_slimseiz_author_' + cls.SOURCE_SHA256
        with _SOURCE_LOCK:
            module = sys.modules.get(namespace)
            if module is None:
                try:
                    from einops import rearrange, repeat, einsum
                except ImportError as error:
                    raise ImportError(
                        'SlimSeiz source requires einops with einsum support (>=0.6).'
                    ) from error
                module = ModuleType(namespace)
                module.__file__ = str(path)
                module.__dict__.update(
                    torch=torch,
                    nn=nn,
                    F=F,
                    math=math,
                    rearrange=rearrange,
                    repeat=repeat,
                    einsum=einsum,
                )
                for index in cls.SOURCE_CELLS:
                    cell = notebook['cells'][index]
                    if cell['cell_type'] != 'code':
                        raise ValueError(
                            'The pinned complete neural cells are required.'
                        )
                    exec(
                        compile(
                            ''.join(cell['source']),
                            str(path) + ':cell' + str(index),
                            'exec',
                        ),
                        module.__dict__,
                    )
                sys.modules[namespace] = module
        return cls(
            module.OneDCNN(input_channels=num_electrodes),
            chunk_size=chunk_size,
            return_features=return_features,
        )

    def forward(self, x: torch.Tensor):
        r"""Return complete binary logits and optionally the native features."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('x must be a floating EEG tensor.')
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3 or x.shape[1:] != (self.num_electrodes, self.chunk_size):
            raise ValueError('Expected [batch,num_electrodes,chunk_size] EEG input.')
        logits, features = self.backend(x)
        return (logits, features) if self.return_features else logits

    def native_state_dict(self):
        """Return every complete native CNN, mixer, normalization and head state."""
        return self.backend.state_dict()

    def load_native_state_dict(self, state_dict, strict: bool = True):
        """Load complete original tensors with normal strict state-key checks."""
        return self.backend.load_state_dict(state_dict, strict=strict)
