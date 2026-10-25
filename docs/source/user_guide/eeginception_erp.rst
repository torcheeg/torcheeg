EEGInceptionERP for event-related potentials
============================================

EEGInceptionERP is a supervised multiscale convolutional classifier for
event-related potentials (ERPs). It retains three temporal branches and
three depthwise spatial filters in the first Inception block, three temporal
branches in the second block, two aggregation convolutions, four average
pooling layers and a dense readout. It is not a pretrained foundation model.

Use the model
-------------

.. code-block:: python

   import torch
   from torcheeg.models import EEGInceptionERP

   model = EEGInceptionERP(num_electrodes=8, chunk_size=128,
                           sampling_rate=128, num_classes=2)
   epochs = torch.randn(4, 8, 128)
   labels = torch.tensor([0, 1, 0, 1])
   logits = model(epochs)
   loss = torch.nn.functional.cross_entropy(logits, labels)
   loss.backward()

Input is ``[batch, electrodes, time]`` or
``[batch, 1, electrodes, time]``. Output is ``[batch, num_classes]`` logits.
The electrode and time dimensions must match the constructor. Sampling rate
controls kernel lengths; the model does not filter, resample or normalize the
signal. Match the montage, units and preprocessing used in training.

The default epoch is one second at 128 Hz. Temporal kernels are 64, 32 and 16
samples, followed by branch-specific depthwise convolutions over all eight
electrodes. The second block uses 16, 8 and 4 samples. As in the pinned
reference, this second-stage division stays four even with custom pooling.
Aggregation kernels are eight and four samples. Native SAME padding preserves
the reference's numerical path on current PyTorch. Explicit asymmetric
padding provides the same geometry on older PyTorch, including even kernels
and odd epoch sizes. Older-version optimizer trajectories are not promised
to be bitwise equal to a newer-version reference.

All convolutional blocks include batch normalization, activation and dropout.
Default batch normalization uses momentum 0.01 and epsilon 1e-5. There is no
constructor probe forward, so running statistics start unmodified. With the
default configuration the complete model has 14,926 parameters and 71 state
tensors. Use ordinary PyTorch state dictionaries, optimizers and training
loops, or the :class:`torcheeg.trainers.ClassifierTrainer` interface.

Independent P300 example
------------------------

:doc:`../auto_examples/examples_bnci2014_008_eeginception_erp` is a standalone
workflow for the original local BNCI2014-008 MAT files. Obtain at least two
subjects from the provider and retain the data attribution. The script does
not download or overwrite source recordings.

.. code-block:: console

   python examples/examples_bnci2014_008_eeginception_erp.py \
       --root-path ./BNCI2014-008 --epochs 1 --output-path ./erp.pt

The eight provider channels are Fz, Cz, Pz, Oz, P3, P4, PO7 and PO8. Recordings
are in microvolts at 256 Hz. The workflow applies common average reference
and a continuous 1--20 Hz filter, extracts one-second epochs at rising flash
markers, and resamples each epoch to 128 samples with a polyphase filter.
One sustained marker produces one epoch, not one epoch per marked sample.
NonTarget maps to zero and Target to one. Incomplete end-of-file epochs are
excluded.

The last subject in filename order is held out. Overlapping flash epochs
remain in their original subject, and channel means and standard deviations
are fit on training subjects only. Fixed epochs and settings do not use test
labels for model selection. The example reports accuracy and balanced
accuracy because P300 classes are imbalanced.

The safe output contains all model tensors, Adam state, Torch and data-order
random states, training normalization, held-out logits, targets and onsets,
and JSON protocol metadata with source hashes. Existing outputs are refused.
This script exports training state but does not implement a resume command.
The model-level tests verify the next optimizer update after a complete
checkpoint restoration.

This is a short CPU teaching protocol. Its result does not establish the
paper's cross-validation performance, EEG-Arena performance or clinical
validity. No patient data or pretrained weights are distributed.

Reference and validation
------------------------

The implementation adapts the BSD-3-Clause Braindecode v1.3.0 source at
``3a2c4c28e7c7bda470cf3ad06534badd5f5bb119``. Bruno Aristimunha and Cedric
Rommel's credits and the license are retained in
``LICENSES/EEGInceptionERP.txt``. The full default structure and native state
names are retained. The default epoch length is explicitly 128 samples;
the reference's constructor defaults to 1000 samples. Set ``chunk_size`` to
the actual epoch length when loading a reference checkpoint.

The reference helper named ``glorot_weight_zero_bias`` actually retains the
default convolutional and dense weight initialization, sets normalization
weights to one and zeros biases. This implementation follows that code,
including its random-number consumption, rather than inferring behavior
from the helper's name. The reference also returns logits.

The dedicated reference workflow obtains the complete pinned Braindecode
package separately and verifies source hashes. Tests compare every initial
state tensor and random state, strict checkpoint loading, train/eval logits,
cross entropy, input and all parameter gradients, batch normalization
buffers and Adam updates. They include default geometry and odd kernels,
custom pooling and electrode counts. The reference package is not a runtime
dependency of TorchEEG. Synthetic functional, full-state resume and eager
full-graph compilation checks are separate from real-data example runs.
GPU, AMP, optimized compiler backends and distributed training are not
established by these CPU tests.

See `the original model paper <https://doi.org/10.1109/TNSRE.2020.3048106>`_
and `the BNCI2014-008 study <https://doi.org/10.3389/fnhum.2013.00732>`_.
