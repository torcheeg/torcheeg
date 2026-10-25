IFNet and IFNetV2 with explicit frequency bands
===============================================

These models implement both complete networks in the author's IFNet
repository. Both use grouped spatial projection, independent depthwise
temporal convolutions per band and additive GELU interaction across bands.
They are supervised classifiers with no pretrained foundation checkpoint.

.. list-table:: Author variants
   :header-rows: 1
   :widths: 20 40 40

   * - Model
     - Temporal aggregation
     - Classifier
   * - :class:`torcheeg.models.IFNet`
     - Average pooling; incomplete last patches are discarded
     - Linear logits
   * - :class:`torcheeg.models.IFNetV2`
     - Mean squared activation per complete patch, clamped to
       ``[1e-4, 1e4]`` and then logged
     - Linear logits with forward max-norm projection

Prepare sub-band input
----------------------

Provide ``[batch, bands, electrodes, time]`` or band-major
``[batch, bands * electrodes, time]``. A raw ``[batch, electrodes, time]``
tensor with fewer channels is rejected when more than one band is configured.
The networks do not automatically filter or repeat raw EEG channels.

.. code-block:: python

   import numpy as np
   import torch
   from torcheeg import transforms
   from torcheeg.models import IFNetV2

   band_signal = transforms.BandSignal(
       sampling_rate=250, order=5,
       band_dict={'low': (4, 16), 'high': (16, 40)})
   bands = band_signal(eeg=np.random.randn(22, 750))['eeg']
   bands = bands.astype(np.float32) / 5.0
   model = IFNetV2(num_electrodes=22, num_bands=2, chunk_size=750)
   logits = model(torch.from_numpy(bands).unsqueeze(0))

The example uses fifth-order causal Butterworth bands at 250 Hz and the
author's fixed offset of zero and scale factor of five. Sampling rate, band order,
units and preprocessing are caller responsibilities. Match those used in
training. Filtering is outside the neural network; these NumPy transforms
are not an autograd path through raw EEG.

Default geometry and state
--------------------------

Defaults follow the author configuration: 22 electrodes per band, two bands,
750 samples, 64 features per band, kernels 63 and 31, and 125-sample patches.
Both complete default models have 10,884 parameters and 20 state tensors.
Successive kernel halving must produce positive odd lengths for every band.
IFNetV2 requires an epoch divisible by the patch length; IFNet uses floor
pooling. The constructor performs no probe forwards.

All convolutional and readout weights use the author's recursive truncated
normal initialization with standard deviation 0.01. Normalization weights
start at one and biases at zero. Native module/state names are retained,
including ``stem.sconv``, ``stem.tconv`` and ``fc.0``. Batch normalization uses
epsilon 1e-5 and momentum 0.1. Strict loading refers
to the two pinned author files with matching constructor geometry. Both
variants share state names and shapes, so strict loading alone cannot
distinguish them. Save and verify the variant identifier as well as the
constructor settings; the standalone exports include ``model_family``.

IFNetV2 projects readout rows to L2 norm 0.5 before each forward, in training
and evaluation. This changes the stored weight and does not differentiate
through the projection, matching the author code. It is different from a
parametrized constraint whose gradient passes through a norm operation.
Save the complete model state, optimizer and random states for continuation.
Model-level tests restore the next training update exactly for both variants.

Independent BCI-IV2a workflows
------------------------------

Each model has a standalone script:

* :doc:`../auto_examples/examples_bciciv2a_ifnet`
* :doc:`../auto_examples/examples_bciciv2a_ifnet_v2`

.. code-block:: console

   python examples/examples_bciciv2a_ifnet.py --root-path ./BCICIV2a \
       --epochs 1 --output-path ./bciciv2a-ifnet.pt
   python examples/examples_bciciv2a_ifnet_v2.py --root-path ./BCICIV2a \
       --epochs 1 --output-path ./bciciv2a-ifnet-v2.pt

Use original labelled MAT files for at least two subjects, retaining provider
attribution. Signals use the provider's 22 EEG channels at 250 Hz in
microvolts. Each script takes the first three-second cue window two seconds
after the one-based trial onset and skips marked artifacts. Subsequent
windows in the same trial are excluded. Subject/session/run/trial metadata
determine selection; the last sorted subject is held out.

The ordered 4--16 and 16--40 Hz bands are computed per extracted window with
BandSignal, cast to float32 and divided by five. Processed bands are cached
in the private memory dataset. This matches the author filter design and
causal operation, while the filter history and window-selection protocol
differ from filtering the author's longer epochs. The example uses fixed
epochs and AdamW at ``2**-12``, with no test-label selection. It does not
reproduce the paper's cross-validation, repeated-trial augmentation or
EEG-Arena scores.

Safe exports contain all model/optimizer/random states, held-out logits and
targets, constructor settings and hashed protocol metadata. Existing outputs
are refused. The scripts export full state but do not implement a resume
command. No recordings or pretrained weights are distributed.

References and validation
-------------------------

The source is `Jiaheng-Wang/IFNet <https://github.com/Jiaheng-Wang/IFNet>`_
at ``00ca53e29d2318a4cf7259878a4e42a7e00c6211``. Jiaheng Wang's credit and
the MIT copyright/permission notice are retained in ``LICENSES/IFNet.txt``.
The `model paper <https://doi.org/10.1109/TNSRE.2023.3257319>`_ describes the
interactive-frequency approach. The Braindecode and Arena raw-input wrappers
have different front ends and other architectural choices; compatibility
with those wrappers is not inferred from a shared model name.

Dedicated CI obtains and hashes both complete author files separately, then
compares all initial states/random consumption, train/eval logits, cross
entropy, input/all-parameter gradients, batch normalization buffers and
AdamW updates. Readout projection is forced active, and custom geometries
are included. Functional tests cover input/band validation, partial/full
patch rules, two-forward accumulation, full checkpoint restoration, eager
full-graph compilation and both synthetic MAT workflows. The author oracle
uses its Timm helper; the library uses the numerically checked PyTorch
initializer without adding a Timm runtime dependency.

These CPU checks do not establish GPU, AMP, optimized compiler, distributed
training or original-paper accuracy.
