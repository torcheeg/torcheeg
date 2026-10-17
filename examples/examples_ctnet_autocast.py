"""
Train CTNet with CPU bfloat16 autocast
====================================

This example uses a standard PyTorch optimizer and CPU bfloat16 autocast.
Keep parameters in float32 and put the forward pass and loss inside autocast.
Run backward outside the autocast context. CPU bfloat16 does not use a CUDA
gradient scaler. Hardware without native bfloat16 can run this example slowly.

Inputs are synthetic (batch, electrodes, samples) tensors with shape (2, 4,
128). They have no recording units, sampling rate, or scientific interpretation.
Use the EEGMat CTNet example for an actual recording workflow.

Run::

    python examples/examples_ctnet_autocast.py
"""

import torch
from torch.nn import functional as F

from torcheeg.models import CTNet


def main():
    if not hasattr(torch, 'amp') or not hasattr(torch.amp, 'autocast'):
        raise RuntimeError('This example requires PyTorch with torch.amp.autocast.')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = CTNet(num_electrodes=4, chunk_size=128, num_classes=2,
                  f1=2, d=2, temporal_kernel=8, spatial_kernel=4,
                  depth=1, heads=2, dropout=0, transformer_dropout=0,
                  forward_dropout=0, pos_dropout=0, cls_dropout=0).train()
    inputs = torch.randn(2, 4, 128)
    targets = torch.tensor([0, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    with torch.amp.autocast('cpu', dtype=torch.bfloat16):
        predictions = model(inputs)
        loss = F.cross_entropy(predictions, targets)
    assert predictions.dtype == torch.bfloat16 and torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    print({'torch': torch.__version__, 'device': 'cpu', 'loss': loss.item(),
           'prediction_dtype': str(predictions.dtype), 'finite_training_step': True})


if __name__ == '__main__':
    main()
