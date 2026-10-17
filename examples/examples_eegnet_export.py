"""
Export EEGNet inference with PyTorch
===================================

Export a frozen evaluation copy, save it, and check the loaded predictions.
The original model remains available for gradient-based training. Use a
PyTorch version with ``torch.export.export``, ``save``, and ``load``.

The synthetic input has shape (2, 1, 32, 128): batch, input map, electrodes,
and samples. It has no physical units, sampling rate, or benchmark meaning.
Export records this static shape. Dynamic-shape and device support require
separate validation for the intended deployment.

Run::

    python examples/examples_eegnet_export.py --output ./eegnet.pt2
"""

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from torcheeg.models import EEGNet


def main():
    parser = argparse.ArgumentParser(description='Save and verify a frozen EEGNet export.')
    parser.add_argument('--output', default='./eegnet.pt2')
    args = parser.parse_args()
    if not hasattr(torch, 'export') or not hasattr(torch.export, 'export'):
        raise RuntimeError('This example requires PyTorch with torch.export.')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = EEGNet(num_electrodes=32, chunk_size=128, num_classes=2)
    inference_model = deepcopy(model).eval().requires_grad_(False)
    inputs = torch.randn(2, 1, 32, 128)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Older exporters also require the no-grad context for constrained weights.
    with torch.no_grad():
        expected = inference_model(inputs)
        exported = torch.export.export(inference_model, (inputs,), strict=True)
        torch.export.save(exported, output)
        loaded = torch.export.load(output)
        torch.testing.assert_close(loaded.module()(inputs), expected)
    assert model.training and all(parameter.requires_grad for parameter in model.parameters())
    print({'torch': torch.__version__, 'device': 'cpu', 'output': str(output),
           'loaded_predictions_match': True, 'training_model_preserved': True})


if __name__ == '__main__':
    main()
