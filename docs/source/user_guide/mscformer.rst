MSCFormer for multi-scale motor imagery decoding
================================================

:class:`torcheeg.models.MSCFormer` implements the complete supervised
multi-scale convolutional Transformer from Zhao et al. The model retains
three temporal/spatial CNN branches, learned positional encodings, the zero
class token, five post-norm Transformer blocks and the complete classifier.
The default network contains 150,724 parameters and 110 state tensors.
It is not a pretrained foundation model and does not download weights.

The `author implementation <https://github.com/snailpt/MSCFormer>`_ is
Apache-2.0. This interface follows the complete CPU-capable
`Braindecode adaptation <https://github.com/braindecode/braindecode/blob/082492fb93a3aa4c5784dc44026642bbc82f49c3/braindecode/models/mscformer.py>`_
at revision ``082492fb93a3aa4c5784dc44026642bbc82f49c3``. Its Apache-2.0
notice and the shared modules' BSD-3-Clause notice are retained separately
under ``LICENSES/``. Braindecode is a validation dependency, not a runtime
dependency of this model.

Input and full architecture
---------------------------

.. code-block:: python

   import torch
   from torcheeg.models import MSCFormer

   model = MSCFormer(num_electrodes=22, chunk_size=1000, num_classes=4)
   eeg = torch.randn(2, 22, 1000)
   logits = model(eeg)
   features, logits = model(eeg.unsqueeze(1), return_features=True)
   loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
   loss.backward()

Provide floating ``[batch, electrodes, time]`` or
``[batch, 1, electrodes, time]`` in the caller's declared channel order.
There is no inferred sampling rate, signal scale, filter, interpolation or
normalization. Use the preprocessing and label mapping of the chosen task.
The model returns class logits; standard ``ClassifierTrainer`` uses the full
network with cross entropy. Optional class-token features have width
``3 * num_filters``, 48 by default.

Each branch retains a biased temporal convolution with a kernel of 85, 65
or 45 samples, a biased depthwise convolution across all EEG rows, BN, ELU,
44-point temporal averaging and dropout. The two consecutive linear
convolutions use the reference's composed convolution calculation while
retaining every original parameter and its gradient. The three branch
outputs concatenate into feature channels, not into extra time samples.

Prepend an unlearned zero token, scale tokens by the square root of their
width and add learned positions drawn from ``N(0, 1)``. The position table has
at least 100 entries and grows when the configured window requires more.
All five blocks retain attention and feed-forward post-norm residuals.
Attention scales scores by ``(3 * num_filters) ** -0.5`` by default, preserving
the original total-embedding scaling rather than silently using head width.
The final class token passes through dropout and the complete linear readout.
There is no added auxiliary training loss.

Configure dimensions and compatible states
------------------------------------------

``heads`` must divide ``3 * num_filters``; all three positive
``temporal_kernels`` are required. ``chunk_size`` must contain at least one
``pool_size`` patch. Incomplete final temporal patches are discarded by the
native average pool. Constructor dimensions are analytic: no sample forward
updates BN buffers or consumes extra random numbers.

The default 44-point pool matches the reference's BCI-IV2a configuration;
choose ``pool_size=52`` for its alternative BCI-IV2b setting. Dataset identity,
sampling rate and preprocessing are not inferred from either value. The bare
author ``Parameters`` helper defaults to 52 even when its network defaults to
the four-class data setting; specify the actual intended configuration.

Complete matching Braindecode state dictionaries load strictly, including
``cnn.branches``, ``position.encoding``, ``trans.layers`` and ``final_layer``.
The bare author CUDA implementation uses different key names; no automatic
conversion or GPU verification is claimed here. Store model settings, ordered
electrodes, sampling rate, units, normalization, labels and split metadata with
checkpoints. Preserve optimizer and random states for exact continuation.

PyTorch 1.10 and later use native SAME convolution; the older path explicitly
pads before the same composed convolution. Modern attention uses PyTorch SDPA
where appropriate and preserves the pinned reference's CPU dropout math and
random draws. An explicit math fallback supports the older attention path.
Tests force both fallbacks and compare outputs and every gradient; this is
separate from claiming a run on every historical PyTorch binary.

Independent training example and validation
-------------------------------------------

The independent
:doc:`BCI-IV2a example </auto_examples/examples_bciciv2a_mscformer>` reads
22 EEG rows in provider order at 250 Hz in microvolts. It filters recordings
at 0.5--40 Hz, retains only the first four-second cue window per trial, skips
marked artifacts and normalizes each extracted window per channel. The last
sorted subject is held out. Fixed epochs use convolution dropout 0.25 and
AdamW at 0.001, with no test-label checkpoint selection. Atomic exports retain
all model/optimizer/random states, test outputs and protocol/source metadata,
refuse overwrite and do not implement command-line resume.

Native tests import the entire fixed Braindecode package and hash the complete
model and four shared implementation files. Default and custom configurations
compare initialization/RNG, all state tensors, train/eval outputs and CE,
input/all-parameter gradients, BN buffers and AdamW updates. Portable tests
also check independent NumPy attention scaling, geometry, input validation,
sample isolation, strict checkpoint continuation and eager full-graph compile.

A short real-data run establishes functional behavior. It does not reproduce
the paper's augmentation, five-fold cross-validation or published scores.
The fixed EEG-Arena backbone is a separate adaptation; this reference check
does not establish its numerical or protocol equivalence. GPU, mixed
precision, optimized compilation and distributed execution are not claimed.
