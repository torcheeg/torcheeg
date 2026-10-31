TMSA-Net with complete local and global attention
=================================================

:class:`torcheeg.models.TMSANet` retains the complete supervised network
from Zhao and Zhu: two temporal convolutions, the dense spatial convolution,
both key-attention paths, all normalization/dropout and residual MLP layers,
and the complete embedding-major classifier. Defaults contain 20,303
parameters and 45 state tensors. This is not a pretrained foundation model.

The `author repository <https://github.com/Whit3Zhao/TMSA-Net>`_ is MIT licensed,
with its notice retained under ``LICENSES/TMSANet.txt``. This implementation
follows the complete CPU-capable
`Braindecode adaptation <https://github.com/braindecode/braindecode/blob/082492fb93a3aa4c5784dc44026642bbc82f49c3/braindecode/models/tmsanet.py>`_
at revision ``082492fb93a3aa4c5784dc44026642bbc82f49c3``. The shared
feed-forward implementation also retains its BSD-3-Clause notice.
Braindecode is a validation dependency, not a runtime dependency.

Provide the physical input and labels
-------------------------------------

.. code-block:: python

   import torch
   from torcheeg.models import TMSANet

   model = TMSANet(num_electrodes=22, chunk_size=1000, num_classes=4)
   eeg = torch.randn(2, 22, 1000)
   logits = model(eeg)
   features, logits = model(eeg.unsqueeze(1), return_features=True)
   loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
   loss.backward()

Use floating ``[batch, electrodes, time]`` or
``[batch, 1, electrodes, time]`` in a declared channel order. Sampling rate,
units, filters, window definition, normalization and label mapping are
caller-defined. The network adds none of them implicitly. Outputs are
logits; the complete network works with standard ``ClassifierTrainer`` and
cross entropy. No weights are downloaded.

Preserve both attention paths
-----------------------------

Temporal kernels 31 and 15 retain their SAME padding and are summed before
BN. A dense spatial convolution spans every EEG row, followed by GELU and
BN. Average pooling uses width 50 and stride 15 by default. A 1000-point
window produces 64 tokens; any incomplete trailing stride is discarded.
Constructor sizes are analytic and leave BN buffers pristine.

Every pre-norm block retains local-key convolutions of widths 3 and 5,
their concatenated BN/dropout, shared query/value projections, separate
local/global key projections, both attention softmax/dropout operations,
the output projection and the full residual feed-forward block. The two
attention outputs are **summed**, not averaged. Each score divides by the
square root of the per-head width. Original tensor rearrangement and shared
view paths are retained for consistent gradient/update behavior.

The native default ``hid_channels=19`` and ``heads=4`` intentionally project
into ``4 * (19 // 4) = 16`` attention dimensions. Exact divisibility is not
required, and silently changing the width to 20 changes the network.
``hid_channels`` must be at least ``heads`` so every head has an input.
The final projection returns 19 features per token.

The classifier flattens in embedding-major order: all time tokens of one
embedding row precede those of the next. Default features have width
``19 * 64 = 1216``. Changing the window or pooling can change readout shape;
retain the original configuration with complete checkpoints.

Complete state and independent example
--------------------------------------

Matching complete Braindecode states load strictly, including
``temp_conv1``, ``temp_conv2``, ``spatial_conv``, both BN modules,
``transformer`` and ``final_layer``. There is no automatic conversion from
other checkpoint key layouts. Preserve constructor settings, physical input,
label mapping, split metadata, optimizer and random states with the tensors.

The independent
:doc:`BCI-IV2a example </auto_examples/examples_bciciv2a_tmsanet>` reads
22 EEG rows in provider order at 250 Hz in microvolts. It filters recordings
at 0.5--40 Hz, takes the first four-second cue window per physical trial,
skips marked artifacts and normalizes each extracted window per channel.
The last sorted subject is held out; fixed epochs do not select model
settings on test labels. Training uses AdamW at 0.001 and native cross entropy.
Atomic exports retain every model/optimizer/random state, test output and
source/protocol metadata, refuse overwrite and do not implement CLI resume.

Validation scope
----------------

Native tests import the full pinned reference package and hash its complete
model, feed-forward module file and attention function file. Default and
custom non-divisible dimensions compare initialization/RNG, every state,
train/eval logits and CE, input/all-parameter gradients, BN and AdamW updates.
Portable tests independently compute both attention branches in NumPy,
check embedding-major flattening, sample isolation, input/configuration
validation, strict checkpoint/next-step continuation and eager full-graph
compile. Existing PyTorch operations support an explicit CPU numerical
comparison without replacing any learned layer.

Short real-data training and complete-reference replay establish functional
behavior. They do not reproduce the original paper's full protocol, fixed
EEG-Arena backbone or published accuracy. No GPU, mixed precision,
optimized compilation or distributed execution result is claimed.
