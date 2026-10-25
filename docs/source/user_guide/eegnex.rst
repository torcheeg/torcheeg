EEGNeX with differentiable weight constraints
=============================================

EEGNeX is a supervised five-block convolutional EEG classifier. It retains
two temporal convolution/normalization blocks, depthwise spatial filtering,
two dilated temporal blocks, two padded average-pooling layers and a dense
readout. ELU and dropout occur in blocks three and five. This is a complete
task-specific architecture, not a pretrained foundation model.

Use the model
-------------

.. code-block:: python

   import torch
   from torcheeg.models import EEGNeX

   model = EEGNeX(num_electrodes=22, chunk_size=1000, num_classes=4)
   logits = model(torch.randn(3, 22, 1000))
   loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 3]))
   loss.backward()

Input is ``[batch, electrodes, time]`` or
``[batch, 1, electrodes, time]``; output is class logits. The constructor
electrode and time dimensions must match. Temporal kernel lengths are in
samples. The model does not filter, normalize or resample signals: retain
the sampling rate, montage and preprocessing used in training.

Default filters are eight and 32, with a spatial depth multiplier of two.
The shared first/second-block temporal kernel is 64 samples. Blocks four and
five use 16-sample kernels with dilation two and four. Batch normalization
uses momentum 0.1 and epsilon 1e-5. Pooling factors four and eight each add
one zero-padded sample on each temporal side and include padding in the
average. Flattened size is computed from both padded pooling operations
without a probe forward or changes to running statistics. At 22 electrodes,
1000 samples and four classes, the model has 56,452 parameters and 32 state
tensors.

Weight constraints and checkpoints
----------------------------------

Spatial filters and readout rows have differentiable L2 max-norm constraints
of 1.0 and 0.25 by default. The constraints use PyTorch parametrization;
optimizers update ``parametrizations.weight.original``. Accessing ``weight``
computes the constrained tensor without overwriting the original parameter.
Gradients pass through that computation. This retains the pinned reference's
behavior, rather than replacing it with an in-place clipping operation.

The reference initializes the spatial convolution with Xavier uniform and
retains the other convolution/dense defaults. Native state names, including
the original constrained weights, are preserved. Save the full state dict,
constructor settings, optimizer and random states. Do not replace original
weights in a training checkpoint with already-clamped inference weights.
Strict native loading refers to Braindecode v1.3.0 with matching geometry;
other implementations can use different state names or constraints.

EEGNeX requires ``torch.nn.utils.parametrize``, available in PyTorch 1.9+.
That API is imported when constructing this model so importing other
TorchEEG models remains possible on older supported runtimes. Current
PyTorch uses native SAME convolution for the reference's numerical path;
explicit padding retains the geometry on older versions. A dedicated test
checks that fallback, including dilation and even kernels. Cross-version
optimizer trajectories are not promised bitwise equal.

Independent motor-imagery example
---------------------------------

:doc:`../auto_examples/examples_bciciv2a_eegnex` trains the full constrained
model on original labelled BCI Competition IV 2a MAT files. Obtain at least
two subjects and retain provider attribution.

.. code-block:: console

   python examples/examples_bciciv2a_eegnex.py --root-path ./BCICIV2a \
       --epochs 1 --output-path ./bciciv2a-eegnex.pt

The script uses the provider's 22 EEG channels at 250 Hz in microvolts. It
filters continuous recordings at 0.5--40 Hz, starts four-second windows two
seconds after the one-based trial onset, skips marked artifacts and
normalizes each channel within its window. Labels one through four map to
zero through three. The last subject is held out; fixed epochs and settings
do not use test labels for selection.

The output safely stores all model and AdamW states, Torch and data-order
random states, held-out logits/targets and protocol metadata with source
hashes. Existing output files are refused. This script exports full state
but does not offer a resume command; model-level tests verify the next
update after checkpoint restoration. The CPU teaching protocol is not the
original paper's cross-validation or an EEG-Arena performance reproduction.
No recordings or pretrained weights are distributed.

Source and validation
---------------------

The implementation adapts Braindecode v1.3.0 at
``3a2c4c28e7c7bda470cf3ad06534badd5f5bb119``, including its differentiable
max-norm helpers and complete initialization. Bruno Aristimunha's credit
and the BSD-3-Clause license are in ``LICENSES/EEGNeX.txt``. See the
`original EEGNeX project <https://github.com/chenxiachan/EEGNeX>`_.

The complete separately obtained reference is checked by source hashes.
Tests compare all initial states and random consumption, train/eval logits,
cross entropy, input/all-parameter gradients, batch normalization buffers
and Adam updates. Constraints are forced active, and custom depth, pooling,
channels, kernels and dilation are included. Separate functional tests cover
unmodified original weights, gradients through constraints, optimizer and
random-state restore, eager full-graph compilation and the standalone MAT
workflow. CPU checks do not establish GPU, AMP, optimized compiler,
distributed training or the paper's reported accuracy.
