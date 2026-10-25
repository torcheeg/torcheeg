"""Interface for the separately supplied complete EEGWaveNet author model."""

import hashlib
from pathlib import Path
import sys
import threading
from types import ModuleType

import torch
from torch import nn


_SOURCE_LOCK = threading.RLock()


class EEGWaveNet(nn.Module):
    r"""Complete external EEGWaveNet with its native log-probability output.

    Retains all six depthwise temporal reductions, five channel-pooling
    branches, batch normalization and the original three-layer classifier.
    Provide floating :obj:`[batch, electrodes, time]` or
    :obj:`[batch, 1, electrodes, time]` EEG. The caller defines sampling rate,
    channel order, signal units and preprocessing. No filter is inferred.

    The author repository does not state a license. Its implementation and
    weights must be supplied separately; TorchEEG does not bundle or download
    them. :meth:`from_source` constructs the complete pinned author network
    from an explicitly supplied trusted file with normalized parameter names.

    Args:
        backend (nn.Module): Complete original :obj:`architecture.Net` instance.
        chunk_size (int): Input samples; at least 448. (default: :obj:`1000`)

    .. code-block:: python

        model = EEGWaveNet.from_source('./author/architecture.py')
        log_probabilities = model(torch.randn(2, 22, 1000))
        loss = torch.nn.functional.nll_loss(
            log_probabilities, torch.tensor([0, 3]))
        loss.backward()
    """

    SOURCE_REVISION = '3b1909873f6cde2d8b5a916327791cf9a59f975c'
    SOURCE_SHA256 = '6922fecd52bfd240dca4ce7475ea94657fc750b2e05f65e2eda52cfa09caf408'

    def __init__(self, backend: nn.Module, chunk_size: int = 1000):
        super().__init__()
        self._validate_int('chunk_size', chunk_size)
        if chunk_size < 448:
            raise ValueError('chunk_size must be at least 448 for all five scales.')
        if not isinstance(backend, nn.Module):
            raise TypeError('backend must be a complete EEGWaveNet nn.Module.')
        required = ['temp_conv' + str(i) for i in range(1, 7)]
        required += ['chpool' + str(i) for i in range(1, 6)] + ['classifier']
        if any(not hasattr(backend, name) for name in required):
            raise ValueError('backend must contain the complete author network.')
        self.num_electrodes = backend.temp_conv1.in_channels
        self.num_classes = backend.classifier[-1].out_features
        self.chunk_size = chunk_size
        self._validate_int('num_electrodes', self.num_electrodes)
        self._validate_int('num_classes', self.num_classes)
        self.backend = backend

    @staticmethod
    def _validate_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')

    @classmethod
    def from_source(
        cls,
        source_path,
        num_electrodes: int = 22,
        chunk_size: int = 1000,
        num_classes: int = 4,
    ):
        r"""Execute the explicit pinned local source and construct its full Net.

        Args:
            source_path (str or Path): Trusted :obj:`architecture.py` from the
                author repository at :attr:`SOURCE_REVISION`. Its exact SHA256
                is checked before execution. The author file only needs torch.
            num_electrodes (int): Ordered EEG input rows. (default: :obj:`22`)
            chunk_size (int): Input samples, at least 448. (default: :obj:`1000`)
            num_classes (int): Output classes. (default: :obj:`4`)
        """
        for name, value in (
            ('num_electrodes', num_electrodes),
            ('chunk_size', chunk_size),
            ('num_classes', num_classes),
        ):
            cls._validate_int(name, value)
        if chunk_size < 448:
            raise ValueError('chunk_size must be at least 448 for all five scales.')
        path = Path(source_path)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != cls.SOURCE_SHA256:
            raise ValueError('Author source SHA256 does not match the pinned release.')
        module_name = '_torcheeg_eegwavenet_author_' + cls.SOURCE_SHA256
        with _SOURCE_LOCK:
            module = sys.modules.get(module_name)
            if module is None:
                module = ModuleType(module_name)
                module.__file__ = str(path)
                exec(compile(source, str(path), 'exec'), module.__dict__)
                sys.modules[module_name] = module
        return cls(module.Net(num_electrodes, num_classes), chunk_size=chunk_size)

    def forward(self, x: torch.Tensor):
        r"""Return unchanged native log probabilities for use with NLLLoss."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('x must be a floating EEG tensor.')
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3 or x.shape[1:] != (self.num_electrodes, self.chunk_size):
            raise ValueError('Expected [batch,num_electrodes,chunk_size] input.')
        return self.backend(x)

    def native_state_dict(self):
        """Return all native states without the wrapper's backend prefix."""
        return self.backend.state_dict()

    def load_native_state_dict(self, state_dict, strict: bool = True):
        """Load every original author state with normal strict key checks."""
        return self.backend.load_state_dict(state_dict, strict=strict)
