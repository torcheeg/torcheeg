ADFCNN with the complete external author network
================================================

:class:`torcheeg.models.ADFCNN` exposes the complete supervised ``Net`` from
`UM-Tao/ADFCNN-MI <https://github.com/UM-Tao/ADFCNN-MI>`_, including its
original classifier. It is a motor-imagery CNN, rather than a pretrained
foundation model. Both spectral/spatial paths, square/log aggregation,
normalized channel attention and the convolutional log-softmax readout
remain intact.

Supply both original files
--------------------------

The inspected repository does not state a license. Obtain ``ADFCNN.py`` and
``layers.py`` from its ``ADFCNN/model`` directory at revision
``b4796ca3a93075e8da9dac5860ea6346f79d5a09`` separately. TorchEEG does not
bundle or download this implementation or its weights. The explicit local
factory checks both hashes in ``ADFCNN.SOURCE_HASHES`` before execution.
Only the original ``model.layers`` import is resolved to a private namespace;
the source bytes and neural computations are unchanged. Existing global
``model`` modules and the interpreter's import function remain untouched.
Importing TorchEEG alone does not execute external source.

.. code-block:: python

   import torch
   from torcheeg.models import ADFCNN

   model = ADFCNN.from_source(
       './author/ADFCNN/model', num_electrodes=22,
       chunk_size=750, num_classes=4)
   log_probabilities = model(torch.randn(2, 22, 750))
   loss = torch.nn.functional.nll_loss(
       log_probabilities, torch.tensor([0, 3]))
   loss.backward()

Alternatively provide an already constructed complete author ``Net`` as
``ADFCNN(backend, chunk_size=750)``. The backbone-only author class named
``ADFCNN`` is incomplete and is rejected. The retained native convolutions
require a PyTorch version supporting string SAME padding; native comparisons
use PyTorch 2.6. No old-framework fallback is inserted into the external code.

Input, native attention and output
----------------------------------

Input is floating ``[batch, electrodes, time]`` or
``[batch, 1, electrodes, time]``. The caller specifies ordered electrodes,
physical sampling rate, units and preprocessing. The author's constructor
accepts an unused ``sampling_rate`` argument; passing 250 does not resample
or establish the physical sampling rate of your input.

The first branch uses 125-point spectral convolution, depthwise electrode
convolution, a pointwise projection and 32-point stride-32 pooling. The
second uses a 30-point spectral convolution, dense electrode convolution,
square activation and 75-point stride-25 pooling followed by log clamped at
1e-6. Both retain normalization, ELU where present and dropout.

At 750 samples the branches produce 23 and 28 positions. The full original
head requires exactly 51 positions, so supported lengths are 750 through
767. Shorter or longer lengths are rejected; the implementation does not
adaptively replace the classifier or add padding to fit it.

Queries and keys are normalized across time, followed by channel attention
scaled by the square root of the number of positions. The author's literal
reshape after attention is preserved, including its ordering. Constrained
convolutions retain the author's forward-time weight projection with maximum
norm 2. This can update weights during evaluation, as in the source.

Output is native **log probabilities**, suitable for ``NLLLoss``. Use
``log_probabilities.exp()`` for reporting probabilities. Retain the complete
model in checkpoints: ``state_dict()`` uses the ``backend.`` prefix;
``native_state_dict()`` and ``load_native_state_dict(..., strict=True)``
expose the complete original backbone and classifier format. Reconstruct
from the separately supplied source and load tensors strictly.

Independent real-data workflow
------------------------------

The :doc:`BCI-IV2a example </auto_examples/examples_bciciv2a_adfcnn>` uses
22 EEG rows in provider order, 250 Hz and microvolts. It filters each full
recording at 0.5--40 Hz, selects the first 750-sample cue window per trial
(two to five seconds), skips marked artifacts and normalizes each window
per channel. It holds out the last sorted subject and trains for fixed
epochs without selecting a checkpoint on test labels.

Supply ``--author-source`` explicitly. The atomic export refuses overwrite
and includes complete model, optimizer and random states, source hashes,
physical metadata, partition IDs, test scores and targets. CLI training
resume is not implemented. This short functional teaching run does not
reproduce a paper score or establish EEG-Arena task-protocol equivalence.

Validation scope
----------------

Tests compare the complete unmodified author ``Net``: default states and
initialization RNG, train/eval outputs, NLL, input and all parameter gradients,
BN buffers, activated max-norm projections, AdamW updates, alternate valid
geometries, strict state loading and next-step optimizer/RNG restoration.
They also check input errors, sample isolation, a fresh-process import with
foreign ``model`` modules and eager full-graph compilation. Dedicated CI
obtains both pinned files and checks hashes; source-dependent tests skip
explicitly when the separate source is unavailable. GPU, AMP, distributed
execution and optimized compiler backends have not been validated here.
