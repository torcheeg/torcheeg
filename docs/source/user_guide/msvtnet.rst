MSVTNet
=======

:class:`torcheeg.models.MSVTNet` is a supervised multi-scale convolutional
transformer for motor imagery. It is a task-specific baseline, and does not
add another pretrained foundation-model family to the coverage count.

Model and training contract
---------------------------

Inputs are floating tensors with shape ``[batch, electrodes, time]`` or
``[batch, 1, electrodes, time]``. Electrode count and sample count must match
the constructor. Preserve the data's channel order and document its sampling
rate, units, filtering and normalization. The model does not resample signals.

The default model retains all four temporal/spatial convolution branches,
their auxiliary classifiers, the learnable class token and position embedding,
and the two-layer transformer. With 1000 samples and pooling widths 8 and 7,
each branch has 17 tokens; their concatenated width is 72.

The default output contains main-head logits for standard classifier trainers.
Enable ``return_branch_logits`` to train the complete joint objective:

.. code-block:: python

   import torch
   from torcheeg.models import MSVTNet, MSVTNetLoss

   model = MSVTNet(num_electrodes=22, chunk_size=1000, num_classes=4,
                  return_branch_logits=True)
   optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
   eeg = torch.randn(4, 22, 1000)
   labels = torch.tensor([0, 1, 2, 3])
   optimizer.zero_grad()
   outputs = model(eeg)
   loss = MSVTNetLoss(main_weight=0.6)(outputs, labels)
   loss.backward()
   optimizer.step()
   main_logits, branch_logits = outputs

The loss is ``0.6 * CE(main) + 0.4 * sum(CE(branch))``. It sums the auxiliary
losses, rather than averaging them. Training on main-head cross entropy alone
does not train the auxiliary classifiers and is a different objective.

The author's model returns log probabilities and uses negative log likelihood.
TorchEEG returns logits and uses cross entropy. Parameter names and all branch
weights are retained, including ``last_head``; removing the parameter-free
LogSoftmax does not change checkpoint tensor names. A native state dictionary
with matching dimensions can be loaded strictly. No weights are downloaded.

PyTorch 1.10 and later use the author's native SAME convolution path to
preserve its floating-point behavior, including near-zero attention gradients
that Adam can amplify. PyTorch 1.8 retains explicit SAME padding. Both paths
preserve tensor names and geometry and are checked for outputs and gradients.

Constructor dimensions are calculated without sample forwards. Fresh
BatchNorm statistics start at zero mean, unit variance and zero batches.
Loading an author's complete state restores its stored statistics, including
any statistics produced by that implementation's initialization probes.

Independent example
-------------------

See :doc:`../auto_examples/examples_bciciv2a_msvtnet` for a complete BCI-IV 2a
training and export workflow. It uses the provider's 22 EEG channels at 250 Hz,
four-second windows starting two seconds after trial onset, 0.5--40 Hz filtering
and per-window normalization. The last subject is reserved for final testing.
The fixed-epoch teaching run does not select a checkpoint using test labels.

The export contains the full state dictionary, optimizer, used random states,
configuration, source hashes, held-out logits and targets. It is safe to read
with ``torch.load(..., weights_only=True)``. The script does not implement
resume; complete model/optimizer/random-state continuation is tested separately.

Source and validation
---------------------

The implementation is adapted from the BSD-3-Clause
`Braindecode implementation <https://github.com/braindecode/braindecode/blob/3a2c4c28e7c7bda470cf3ad06534badd5f5bb119/braindecode/models/msvtnet.py>`_
by Tao Yang and Bruno Aristimunha. The notice is retained in
``LICENSES/MSVTNet.txt``.

Tests separately compare the complete
`author implementation <https://github.com/SheepTAO/MSVTNet/blob/c3fa7fd7e0acc13ebf85352fe583dd4e14f85d34/MSVTNet.py>`_
at its pinned file hash. The original source is obtained explicitly for that
test and is not bundled with TorchEEG. The oracle checks main and auxiliary
outputs, the joint loss, input and every parameter gradient, BatchNorm buffers,
and an optimizer update in training and evaluation modes, including even kernels.

Portable tests cover complete checkpoint continuation with dropout and the
used random state, patient-independent evaluation, input validation, and
full-graph compilation with the eager backend. These tests do not establish
GPU, optimized-compiler, mixed-precision, or distributed behavior. Synthetic
MAT fixtures exercise the independent script and measure no motor-imagery
performance. Short real-data runs are functional evidence, not published scores.
