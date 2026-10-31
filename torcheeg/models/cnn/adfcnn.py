"""Interface for the separately supplied complete ADFCNN author network."""

import builtins
import hashlib
from pathlib import Path
import sys
import threading
from types import ModuleType

import torch
from torch import nn


_SOURCE_LOCK = threading.RLock()


class ADFCNN(nn.Module):
    r"""Complete external attention-based dual-scale fusion CNN.

    Retains the author's two spectral/spatial paths, constrained convolutions,
    square/log aggregation, normalized channel attention, original reshapes
    and full convolutional log-softmax classifier. Input is floating
    :obj:`[batch, electrodes, time]` or :obj:`[batch, 1, electrodes, time]`.
    Output is native log probabilities for NLLLoss. Sampling rate, signal
    scale, ordered channels and preprocessing are supplied by the caller.

    The inspected author repository does not state a license. Supply its
    complete trusted source files separately; this class does not bundle or
    download code or weights. :meth:`from_source` checks both exact hashes.

    Args:
        backend (nn.Module): Complete original :obj:`model.ADFCNN.Net`,
            including its original classifier.
        chunk_size (int): Samples consistent with the fixed native readout.
            The default 750 gives 23+28=51 pooled positions.
            (default: :obj:`750`)

    .. code-block:: python

        model = ADFCNN.from_source('./author/ADFCNN/model')
        log_probabilities = model(torch.randn(2, 22, 750))
        loss = torch.nn.functional.nll_loss(
            log_probabilities, torch.tensor([0, 3]))
        loss.backward()
    """

    SOURCE_REVISION = 'b4796ca3a93075e8da9dac5860ea6346f79d5a09'
    SOURCE_HASHES = {
        'ADFCNN.py': '85b9d40dd7d994539456ff94d87f75ba7f08f789503d4006ec128ca76e611c31',
        'layers.py': '7db116eaa6046ba56d73f6aeafbda07b220c71b8afd5ba105469b97310b211a4',
    }

    @staticmethod
    def _positive_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')

    @staticmethod
    def _pooled_positions(chunk_size):
        if chunk_size < 75:
            return 0
        return chunk_size // 32 + (chunk_size - 75) // 25 + 1

    def __init__(self, backend: nn.Module, chunk_size: int = 750):
        super().__init__()
        self._positive_int('chunk_size', chunk_size)
        if not isinstance(backend, nn.Module):
            raise TypeError('backend must be the complete ADFCNN nn.Module.')
        if not hasattr(backend, 'backbone') or not hasattr(backend, 'classifier'):
            raise ValueError('Provide the complete author Net, including classifier.')
        required = (
            'spectral_1',
            'spectral_2',
            'spatial_1',
            'spatial_2',
            'w_q',
            'w_k',
            'w_v',
        )
        if any(not hasattr(backend.backbone, name) for name in required) or not hasattr(
            backend.classifier, 'dense'
        ):
            raise ValueError('Provide both complete native branches and readout.')
        readout = backend.classifier.dense[0]
        if readout.kernel_size != (1, 51) or self._pooled_positions(chunk_size) != 51:
            raise ValueError(
                'The complete native readout requires 51 pooled positions.'
            )
        self.num_electrodes = backend.backbone.spatial_1[0].kernel_size[0]
        self.num_classes = readout.out_channels
        self.chunk_size = chunk_size
        self._positive_int('num_electrodes', self.num_electrodes)
        self._positive_int('num_classes', self.num_classes)
        self.backend = backend

    @classmethod
    def from_source(
        cls,
        source_dir,
        num_electrodes: int = 22,
        chunk_size: int = 750,
        num_classes: int = 4,
    ):
        r"""Construct the complete original Net from an explicit local folder.

        Args:
            source_dir (str or Path): Trusted folder containing both
                :obj:`ADFCNN.py` and :obj:`layers.py` from the author's
                :obj:`ADFCNN/model` directory at :attr:`SOURCE_REVISION`.
                Both hashes are checked before any source execution.
            num_electrodes (int): Ordered EEG rows. (default: :obj:`22`)
            chunk_size (int): Samples producing exactly 51 pooled positions,
                as required by the complete native head. (default: :obj:`750`)
            num_classes (int): Output classes. (default: :obj:`4`)

        Only the original import of :obj:`model.layers` is resolved into a
        private hash-named namespace; neither source file nor any neural
        calculation is edited. Existing top-level model modules are retained.
        The author's unused sampling_rate argument is passed as 250; this
        does not resample input or infer its physical sampling rate.
        """
        for name, value in (
            ('num_electrodes', num_electrodes),
            ('chunk_size', chunk_size),
            ('num_classes', num_classes),
        ):
            cls._positive_int(name, value)
        if cls._pooled_positions(chunk_size) != 51:
            raise ValueError(
                'The complete native readout requires 51 pooled positions.'
            )
        root = Path(source_dir)
        sources = {name: (root / name).read_bytes() for name in cls.SOURCE_HASHES}
        for name, expected in cls.SOURCE_HASHES.items():
            if hashlib.sha256(sources[name]).hexdigest() != expected:
                raise ValueError(
                    f'{name} SHA256 does not match the pinned author source.'
                )
        namespace = (
            '_torcheeg_adfcnn_author_'
            + hashlib.sha256(b''.join(sources.values())).hexdigest()
        )
        with _SOURCE_LOCK:
            module = sys.modules.get(namespace + '.ADFCNN')
            if module is None:
                package = ModuleType(namespace)
                package.__path__ = [str(root)]
                layers = ModuleType(namespace + '.layers')
                layers.__file__ = str(root / 'layers.py')
                exec(
                    compile(sources['layers.py'], layers.__file__, 'exec'),
                    layers.__dict__,
                )
                original_import = builtins.__import__

                def source_import(
                    name, globals=None, locals=None, fromlist=(), level=0
                ):
                    if name == 'model.layers' and level == 0:
                        return layers
                    return original_import(name, globals, locals, fromlist, level)

                module = ModuleType(namespace + '.ADFCNN')
                module.__file__ = str(root / 'ADFCNN.py')
                module.__package__ = namespace
                module.__dict__['__builtins__'] = dict(
                    vars(builtins), __import__=source_import
                )
                exec(
                    compile(sources['ADFCNN.py'], module.__file__, 'exec'),
                    module.__dict__,
                )
                package.layers, package.ADFCNN = layers, module
                sys.modules.update(
                    {
                        namespace: package,
                        namespace + '.layers': layers,
                        namespace + '.ADFCNN': module,
                    }
                )
        return cls(
            module.Net(
                num_classes=num_classes, num_channels=num_electrodes, sampling_rate=250
            ),
            chunk_size=chunk_size,
        )

    def forward(self, x: torch.Tensor):
        r"""Return unchanged native log probabilities, suitable for NLLLoss."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError('x must be a floating EEG tensor.')
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1:] != (1, self.num_electrodes, self.chunk_size):
            raise ValueError('Expected [batch,1,num_electrodes,chunk_size] input.')
        return self.backend(x)

    def native_state_dict(self):
        """Return every native backbone and classifier state without a prefix."""
        return self.backend.state_dict()

    def load_native_state_dict(self, state_dict, strict: bool = True):
        """Load complete original author tensors with normal strict key checks."""
        return self.backend.load_state_dict(state_dict, strict=strict)
