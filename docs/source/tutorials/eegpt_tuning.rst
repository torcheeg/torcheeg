EEGPT parameter-efficient tuning
================================

Choose an explicit electrode order and preprocessing pipeline before tuning.
The encoder accepts EEG shaped ``[batch, electrodes, samples]``. These helpers
modify an :class:`torcheeg.models.EEGPT` encoder or an EEGPT task model in place.
They apply no resampling, channel mapping or automatic weight download.

Available methods
-----------------

.. list-table:: Registered tuning modules
   :header-rows: 1
   :widths: 20 50 30

   * - Method
     - Added parameters
     - Initial behavior
   * - :func:`torcheeg.models.add_eegpt_adapters`
     - A residual bottleneck after each selected block's attention and MLP residual
     - Zero up-projections preserve evaluation outputs
   * - :func:`torcheeg.models.add_eegpt_prefix`
     - Learned key and value vectors in each selected attention layer
     - Attention changes, including its normalization
   * - :func:`torcheeg.models.add_eegpt_lora`
     - Low-rank branches on exact selected linear projections
     - Zero B factors preserve evaluation outputs

Adapter placement follows the bottleneck approach of
`Houlsby et al. (2019) <https://proceedings.mlr.press/v97/houlsby19a.html>`_.
The prefix method uses direct per-layer key/value parameters, following the
idea of `Li and Liang (2021) <https://aclanthology.org/2021.acl-long.353/>`_.
It has no additional prefix reparameterization network. LoRA applies input
dropout and low-rank updates, as in
`Hu et al. (2022) <https://github.com/microsoft/LoRA>`_. These helpers adapt the methods to EEGPT.

Install modules before the optimizer
-------------------------------------

.. code-block:: python

   import torch
   from torcheeg.models import EEGPT, EEGPTClassifier, add_eegpt_lora

   encoder = EEGPT(num_electrodes=3, channel_names=['C3', 'CZ', 'C4'])
   model = EEGPTClassifier(encoder, num_classes=2)
   add_eegpt_lora(model, rank=4, alpha=8.0, dropout=0.1)
   optimizer = torch.optim.Adam(
       [parameter for parameter in model.parameters() if parameter.requires_grad],
       lr=1e-3)

This encoder starts with random weights. To tune a pretrained encoder, load a
complete local encoder state with ``EEGPT.from_pretrained`` before installing
modules. The encoder loader uses safe deserialization and strict key/shape
checks. A pretrained encoder does not supply trained classification-head
weights.

The default ``freeze_base=True`` freezes all base encoder parameters and
retains all installed adapter, prefix and LoRA parameters as trainable. Task
head flags are preserved. ``freeze_base=False`` preserves existing flags; it
does not unfreeze previously frozen weights. Helpers can target unique,
zero-based ``blocks`` indices. LoRA targets are exact block-relative names:
``attn.qkv``, ``attn.proj``, ``mlp.fc1`` and ``mlp.fc2``. It does not match
adapter projections or task heads by substring.

Use ``freeze_encoder=False`` on task models when tuning encoder modules. A
task model with ``freeze_encoder=True`` intentionally freezes the whole
encoder and keeps it in evaluation mode for head-only training. Tuning
helpers reject that conflicting setting. To return to full training on a
task model created with ``freeze_encoder=False``, use
``model.requires_grad_(True)`` and ``model.train()``.

Adapters and LoRA start with zero up/B projections. Their down/A factors can
have zero gradient on the first step; the up/B factors receive the initial
update. Check gradients over subsequent optimization steps. These registered modules participate in
``train()``, ``eval()``, dtype/device changes, copying and state dictionaries.
LoRA does not merge weights when entering evaluation mode.

Restore a saved task model
--------------------------

The examples save the full tensor state and the encoder, head and tuning
settings. Build the matching structure before loading the state. This
example restores the dedicated LoRA script's output:

.. code-block:: python

   import torch
   from torcheeg.models import EEGPT, EEGPTClassifier, add_eegpt_lora

   checkpoint = torch.load('bciciv2b-eegpt-lora.pt', map_location='cpu',
                           weights_only=True)
   if checkpoint['tuning_method'] != 'lora':
       raise ValueError('Use the matching tuning helper for this checkpoint.')
   model = EEGPTClassifier(EEGPT(**checkpoint['encoder_config']),
                           **checkpoint['head_config'])
   add_eegpt_lora(model, **checkpoint['tuning_config'])
   model.load_state_dict(checkpoint['state_dict'], strict=True)
   model.eval()

Base tensor names and parameter objects are preserved when modules are
installed. Augmented states have extra tuning keys, so they cannot be
loaded into a plain encoder without recreating the tuning structure.

Separate real-data examples
----------------------------

* :doc:`../auto_examples/examples_bciciv2b_eegpt_adapters`
* :doc:`../auto_examples/examples_bciciv2b_eegpt_prefix`
* :doc:`../auto_examples/examples_bciciv2b_eegpt_lora`

Each script uses C3/Cz/C4, 256 Hz, four-second windows, per-window channel
standardization, and a held-out subject. Each states whether an encoder
checkpoint was loaded. Small random-encoder defaults and one-epoch checks
verify training and state restoration. They do not establish pretrained
EEGPT performance or a published benchmark score. Keep the held-out subject
out of epoch and parameter selection.
