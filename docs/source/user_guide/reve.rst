REVE pretrained models
======================

REVE encodes EEG with overlapping temporal patches, four-dimensional spatial
and temporal Fourier embeddings, and a transformer. The native pretrained
query pools the encoded electrodes and patches. The complete released states
are retained; no pretrained parameters are silently discarded.

Sources and variants
--------------------

The `author repository <https://github.com/elouayas/reve_eeg>`_ at
``06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7`` uses MIT-licensed source code.
See ``LICENSES/REVE.txt``. Cite El Ouahidi et al.,
`REVE: A Foundation Model for EEG -- Adapting to Any Setup with Large-Scale
Pretraining on 25,000 Subjects <https://arxiv.org/abs/2510.21585>`_, NeurIPS2025.

The official checkpoints separately use the **REVE Responsible Use License
v1.0**, dated14September2026. Read the checkpoint's LICENSE for use,
redistribution, adapted-model provenance and attribution requirements.
The source-code MIT license does not replace the model terms. TorchEEG does
not distribute either official or adapted weights.

.. list-table:: Complete official variants
   :header-rows: 1
   :widths: 15 15 15 15 40

   * - Variant
     - Width
     - Layers
     - Heads
     - Official source
   * - base
     - 512
     - 22
     - 8
     - `brain-bzh/reve-base <https://huggingface.co/brain-bzh/reve-base>`_
   * - large
     - 1216
     - 22
     - 19
     - `brain-bzh/reve-large <https://huggingface.co/brain-bzh/reve-large>`_

Base revision ``dc2a075c287bb2f6c04ee5875bd79535a0f7dba6``:
SHA256 ``8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe``.
Large revision ``317531c7b8e6d9777191598de8a8b816764dca07``:
SHA256 ``f519428404aab27f7db816722c7fb8c73d08daf10986c9c21b0066be9e93bb31``.
Each contains140tensors, including ``cls_query_token``. Both use200-sample
patches,20-sample overlap,64-wide heads, GEGLU and a2.66MLP ratio. Base has
four frequencies per coordinate axis; Large has five. Some third-party
benchmark defaults specify other dimensions; use the official configuration.

Load and use features
---------------------

Install ``torcheeg[foundation]`` for safetensors. Download the chosen official
file separately, check its hash, and pass its local path. The loader uses CPU,
checks all keys, shapes, dtypes and finite values, and executes no remote code.

.. code-block:: python

   import torch
   from torcheeg.models import REVE

   model = REVE.from_pretrained('./reve-base.safetensors', variant='base').eval()
   eeg = torch.randn(2, 19, 800)  # synthetic shape check, not EEG performance
   positions = torch.load('./positions.pt', weights_only=True)  # [19,3]
   with torch.no_grad():
       features = model(eeg, positions)  # [2,19,4,512]
       pooled = model.attention_pooling(features)  # [2,512]

Supply EEG as floating ``[batch,electrodes,samples]`` at200Hz. The released
forward converts EEG tofloat32; use float32 model weights. Coordinates must
be floating ``[electrodes,3]`` or ``[batch,electrodes,3]`` in the same order
as EEG channels. Match the author's coordinate frame and metre scale.
The author ``hf/reve-positions/positions.json`` at the source revision above
provides explicit named coordinates. Preserve exact names. Do not silently
drop unknown channels or substitute a different coordinate frame. No filter,
signal scaling, channel selection or normalization occurs inside the model.

Native unfold uses stride180. An800-sample window gives four complete patches
covering740samples, leaving60unused by the backbone. At least200samples are
required. No padding is added. ``forward_features`` returns flattened
``[batch,electrodes*patches,width]`` tokens; ``return_layers=True`` returns the
embedded input and every transformer layer. ``forward`` returns native
``[batch,electrodes,patches,width]`` features.

The released Hugging Face forward is deterministic in train and eval and
does not apply coordinate noise, despite retaining ``noise_ratio`` in its
configuration. TorchEEG follows that behavior and never changes input
coordinates in place. This differs from the training repository's NumPy
coordinate augmentation. PyTorch2 uses SDPA when available; the native
classical attention equation is the fallback for older versions. Fused
attention can have small floating-point differences across backends.

Train a classification head
---------------------------

``REVEClassifier`` pools with the pretrained query, then applies a new
RMSNorm/dropout/linear head. The head is newly initialized, so pretrained
feature support does not imply an already trained downstream classifier.

.. code-block:: python

   from torcheeg.models import REVEClassifier
   from torcheeg.trainers import ClassifierTrainer

   classifier = REVEClassifier(model, num_classes=2, freeze_backbone=True,
                               positions=positions)
   trainer = ClassifierTrainer(classifier, num_classes=2, accelerator='cpu')
   # trainer.fit(train_loader, val_loader, max_epochs=10)

Fixed coordinates are saved as a buffer for ordinary ``classifier(eeg)``
calls. ``classifier(eeg, batch_positions)`` overrides them. A frozen backbone,
including its query token, stays in eval mode; only the new head trains.
For strict restore, recreate the same backbone and fixed coordinate buffer
before loading the complete classifier state.

Independent real-data examples
------------------------------

* :doc:`../auto_examples/examples_eegmat_reve_base`
* :doc:`../auto_examples/examples_eegmat_reve_large`

Each script needs a complete local checkpoint, the author position JSON and
EEGMat subject pairs. It resamples whole recordings500→200Hz, then applies
FIR0.5–75Hz/notch50Hz; each800-sample window receives a channel-wise population
z-score. Normalization includes the trailing60samples, even though the model
does not embed that tail. This is an explicit teaching protocol, distinct
from author pretraining whole-record normalization/clipping.

The last complete subject is held out; fixed epochs use all other subjects.
Each script saves the full model, coordinates, source/model/position hashes,
JSON configuration and predictions, then requires exact strict-restored
predictions. These examples validate the workflow and are not published
benchmark scores. Sphinx Gallery renders them without executing training;
run the scripts separately on real data.
