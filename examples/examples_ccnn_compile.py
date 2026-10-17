"""
Compile CCNN training with PyTorch
==================================

This CPU example compares a compiled CCNN training step with eager execution.
Install a PyTorch version with ``torch.compile`` and a working CPU Inductor
compiler toolchain. The first call includes compilation time.

The synthetic input represents four feature maps on a 3-by-5 grid. It has no
recording units, sampling rate, or scientific performance interpretation.
For training with EEG recordings, see the dataset-specific examples.

Run::

    python examples/examples_ccnn_compile.py
"""

from copy import deepcopy

import torch
from torch.nn import functional as F

from torcheeg.models import CCNN


def main():
    if not hasattr(torch, 'compile'):
        raise RuntimeError('This example requires PyTorch with torch.compile.')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = CCNN(in_channels=4, grid_size=(3, 5), num_classes=2, dropout=0).train()
    reference = deepcopy(model)
    compiled = torch.compile(model, fullgraph=True)
    inputs = torch.randn(2, 4, 3, 5)
    targets = torch.tensor([0, 1])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)

    actual = compiled(inputs)
    expected = reference(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    loss = F.cross_entropy(actual, targets)
    loss.backward()
    F.cross_entropy(expected, targets).backward()
    for parameter, other in zip(model.parameters(), reference.parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-3, atol=1e-4)
    optimizer.step()
    reference_optimizer.step()
    for parameter, other in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, other, rtol=1e-3, atol=1e-4)
    print({'torch': torch.__version__, 'device': 'cpu', 'loss': loss.item(),
           'compiled_training_matches_eager': True})


if __name__ == '__main__':
    main()
