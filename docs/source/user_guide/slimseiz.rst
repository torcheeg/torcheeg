SlimSeiz with the complete original CNN and Mamba network
=========================================================

:class:`torcheeg.models.SlimSeiz` exposes the full binary neural network from
`guoruilu/SlimSeiz <https://github.com/guoruilu/SlimSeiz>`_. It retains both
multiscale convolution stages, the complete selective state-space Mamba
mixer, RMS normalization, residual connection, temporal average and original
two-class linear head. It is a supervised neural model, not an EEG foundation
model. No mixer is replaced by an identity layer or an unrelated sequence
model.

Supply the complete trusted original notebook
---------------------------------------------

The inspected repository does not state a source license. Obtain the raw
``Loop_ADDNet_kFold.ipynb`` at revision
``84b168278173a0e14163a6839fb36d7c41bc3c52`` separately. TorchEEG provides
its own interface and does not bundle or download the notebook, neural code,
data or pretrained weights. The notebook credits the Mamba minimal and
original Mamba implementations; those attributions remain in that original.

The explicit local factory verifies SHA256
``a77807eee44b3a95780c12f656091172467dd7f967be942061d6cf417921d2a9``
before execution. A re-saved notebook has different bytes and is rejected.
Only its two complete neural-definition cells execute, without modification:
cell 1 contains ``MambaBlock`` and ``RMSNorm``, and cell 7 contains ``OneDCNN``.
Unrelated imports, data processing, fitting and training cells do not run.
The neural dependencies are PyTorch and einops with ``einsum`` support
(einops 0.6 or later); CUDA-specific Mamba kernels are not required.
Importing TorchEEG alone does not execute external source.

.. code-block:: python

   import torch
   from torcheeg.models import SlimSeiz

   model = SlimSeiz.from_source(
       './author/Loop_ADDNet_kFold.ipynb', num_electrodes=8,
       chunk_size=1024, num_classes=2)
   eeg = torch.randn(2, 8, 1024)
   logits = model(eeg)
   loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
   loss.backward()

An already constructed complete original ``OneDCNN`` can instead be supplied
as ``SlimSeiz(backend, chunk_size=1024)``. The full native model remains in
``model.backend``. No convolution, normalization, selective scan or head is
replaced. At eight input rows the complete network has 22,946 parameters and
28 state tensors. The notebook constructor defaults to three rows, which
has 21,266 parameters; do not report that smaller count for an eight-row
model.

Physical input and original outputs
-----------------------------------

Input is floating ``[batch, electrodes, time]`` or
``[batch, 1, electrodes, time]``. The caller supplies physical sampling rate,
units, ordered channels, preprocessing and selected electrode rows. Neither
channel selection nor resampling occurs inside the network. Native pooling
first reduces by eight and then by four, requiring at least 32 input samples.
Nonmultiples discard incomplete pooling windows as in the original. The
following stride-two convolution and adaptive temporal mean retain variable
valid input lengths without changing the complete classifier.

Output defaults to two unnormalized categorical logits for cross entropy.
The complete native classifier supports exactly two classes; requests for
other class counts are rejected. Set ``return_features=True`` to return the
original ``(logits, features)`` pair, including the full 32-dimensional feature
vector. This supports explicitly chosen feature objectives without inferring
paper training settings.

The native Mamba implementation projects to width 64, uses state size 16 and
delta rank two, and retains its depthwise causal convolution, input-dependent
state parameters, sequential selective scan and full output projection. Its
transition discretization and skip term remain unchanged. This is the
original notebook's pure PyTorch implementation; no equivalence to a
different fused Mamba implementation or kernel is inferred.

Checkpoint and protocol boundaries
----------------------------------

``state_dict()`` registers every native state under ``backend.``.
``native_state_dict()`` and ``load_native_state_dict(..., strict=True)``
provide the complete original format. Reconstruct from the trusted local
notebook and load tensors strictly. Save the source revision/hash, geometry,
channel selection, physical metadata, class mapping and protocol with the
checkpoint; do not pickle the external Python object.

The published SlimSeiz framework also performs adaptive channel selection
and supervised-contrastive training. These are separate protocol components,
not implicit behavior of this neural interface. Preictal/interictal horizons,
seizure/event isolation, postictal exclusions and channel fitting must be
specified by a seizure-prediction workflow. A successful classification step
alone does not validate seizure prediction or the complete paper pipeline.

Independent motor-imagery transfer workflow
-------------------------------------------

The :doc:`BCI-IV2a example </auto_examples/examples_bciciv2a_slimseiz>`
retains left/right-hand labels 1/2 mapped to 0/1 and all 22 provider-order
EEG rows at 250 Hz in microvolts. It filters each recording at 0.5--40 Hz,
uses only the first cue window per trial (two to six seconds; 1000 points),
skips marked artifacts and standardizes each window per channel. The last
sorted subject is held out; fixed epochs do not select on test labels.

The example explicitly uses cross entropy and all available rows, without
adaptive channel selection or contrastive loss. It is a short motor-imagery
transfer teaching workflow, not a seizure-prediction experiment, clinical
validation, paper score or fixed EEG-Arena protocol reproduction. The atomic
export refuses overwrite and saves full model/optimizer/random states,
test logits/targets and source/physical/channel/label/split metadata. CLI
resume is not implemented.

Validation scope
----------------

Tests execute both complete unmodified neural-definition cells from the
pinned original notebook. They compare initialization/RNG, every state,
train/eval logits and features, CE, input and all parameter gradients and
AdamW updates. Independent NumPy selective-scan recurrence and finite-
difference gradient checks test the state dynamics separately. Other checks
cover source integrity/no execution on a wrong hash, minimum/alternate
geometry, BCT/B1CT, zero inputs, sample isolation, strict checkpoints,
complete next-step optimizer/RNG resume, eager full-graph compilation and
the runnable binary workflow. Dedicated CI supplies the pinned notebook;
source-dependent tests skip explicitly when it is unavailable. GPU, AMP,
optimized compiler backends and distributed execution are not claimed.
