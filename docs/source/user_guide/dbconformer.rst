DBConformer with a complete external author backend
===================================================

:class:`torcheeg.models.DBConformer` preserves the original dual-branch
supervised network: temporal grouped/depthwise convolutions, spatial
convolutional embeddings, two independent five-block Transformers, learned
temporal/spatial positions, channel-attention fusion and the complete MLP
classifier. This is not a pretrained foundation model.

Supply the author implementation
-------------------------------

The `author repository <https://github.com/wzwvv/DBConformer>`_ does not
state a source license at the inspected revision. TorchEEG distributes an
interface, not the author's source or weights. Obtain a trusted copy of
``models/DBConformer.py`` at revision
``02cf63206087045c5cd0e8fa882831102e6fd01b`` separately. The factory verifies
SHA256 ``b891d8ba28b3ef5064106065e4a2c9bd1e346fe29b6d68cc50c91dbf6cfa4bd1``
before executing that explicit local file. It never downloads Python code.
Install the optional author dependencies; native validation uses timm 1.0.10.

.. code-block:: bash

   pip install -e ".[model,dataset]" timm==1.0.10

.. code-block:: python

   import torch
   from torcheeg.models import DBConformer

   model = DBConformer.from_source(
       './author/DBConformer.py', num_electrodes=22,
       chunk_size=1001, num_classes=4)
   eeg = torch.randn(2, 22, 1001)
   logits = model(eeg)
   features, logits = model(eeg, return_features=True)
   loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
   loss.backward()

Importing TorchEEG does not load the author file or timm. The loader restores
the caller's cuDNN benchmark/deterministic settings after executing the source,
including failure. It caches the complete verified module and registers it
for PyTorch module resolution. The full model remains in ``model.backend``;
no layer, trainable parameter or author forward calculation is replaced.
An already constructed complete original backend can also be supplied as
``DBConformer(backend, chunk_size=1001)``. Both branches are required.

Input, cropping and fusion
--------------------------

Input is floating ``[batch, electrodes, time]`` or
``[batch, 1, electrodes, time]`` in a caller-defined channel order, scale and
sampling rate. There is no implicit filter, normalization or resampling.
The factory uses TorchEEG's ``num_electrodes``, ``chunk_size`` and
``num_classes`` names. Its class-count default is four; the author's bare
constructor defaults to two. Every attention layer retains ten native heads,
so ``hid_channels`` must be divisible by ten.

By default the temporal branch removes the last sample before its 125-point
average pooling. For 1001 input points, it pools 1000 samples into eight
tokens. The spatial branch still uses all 1001 points. Native temporal
positions are allocated using ``chunk_size // patch_size``; with positions
enabled, a 1000-point cropped input would produce only seven tokens and is
rejected before constructing the network. Use ``drop_last_sample=False``
to select the author's uncropped MI1-7 path for an exactly divisible input,
or choose an input length consistent with the native positions. This flag
selects the native crop operation; it does not identify a dataset or infer
its preprocessing. An incomplete final patch is discarded by native pooling.

``fusion='attention'`` uses a temporal mean and learned channel-attention
spatial pooling, concatenated into 80 default features. ``'mean'`` preserves
native mean pooling and concatenation. ``'gated'`` preserves the native
sigmoid weighted mixture, producing 40 default features. Each uses the
original classifier with cross entropy; there is no added auxiliary loss.
``positional_embedding=False`` disables both original positional parameters.
No single-branch ablation is presented as a complete DBConformer.

Complete checkpoints
--------------------

``model.state_dict()`` contains every native state under ``backend.``.
Use ``model.native_state_dict()`` and
``model.load_native_state_dict(state, strict=True)`` for complete original
author checkpoints without that prefix. Standard wrapper checkpoints also
load strictly. Preserve constructor settings, source revision/hash, channel
order, sampling rate, units, normalization, label mapping and split metadata.
Reconstruct from the separately supplied source before loading tensors;
do not pickle the external Python object into a portable checkpoint.

.. code-block:: python

   state = model.state_dict()
   restored = DBConformer.from_source('./author/DBConformer.py')
   restored.load_state_dict(state, strict=True)

The independent
:doc:`BCI-IV2a example </auto_examples/examples_bciciv2a_dbconformer>` uses
22 EEG channels in the provider order at 250 Hz in microvolts. It filters
recordings at 0.5--40 Hz and normalizes each extracted window per channel.
Only the first inclusive 2--6 s cue window per trial is retained, marked
artifacts are skipped and the last sorted subject is held out. Fixed epochs
use no test-label selection. The atomic export includes all model/optimizer
states, Torch/data-order RNG states, test outputs and protocol metadata.
It refuses to overwrite an existing output and does not implement CLI resume.

Validation scope
-----------------

The optional native suite uses the entire pinned author file. It checks
initialization and RNG, all state keys, both branch outputs, train/eval loss,
input/all-parameter gradients, BN buffers, AdamW updates, native fusion/crop
and position options, strict checkpoint/next-step resume, source hash checks,
loader cleanup and eager full-graph compilation. Tests of a short real-data
workflow establish functional behavior, not published or EEG-Arena accuracy.
There is no claim about GPU, mixed precision, optimized compilation or
distributed execution. Source-dependent tests explicitly skip when the
separate author file is unavailable; the dedicated CI supplies it by hash.
