EEGMamba pretrained model
=========================

``EEGMamba`` contains the complete pretrained patch encoder, twelve Mamba2
blocks and waveform reconstruction head. ``forward_features`` returns
contextual patch features; ``forward`` reconstructs each 200-sample patch.
``EEGMambaClassifier`` adds the author's flatten/ELU downstream head and
retains the complete backbone state.

Source and official weights
---------------------------

Cite Wang et al., `An EEG foundation model with Mamba
<https://www.sciencedirect.com/science/article/pii/S0893608025006963>`_,
Neural Networks, 2025. The `author implementation
<https://github.com/wjq-learning/EEGMamba>`_ at
``dbc83fa072744201e8897aeb9f65007b952ad323`` uses MIT.
The portable Mamba2/SSD implementation adapts Dao and Gu's
`state-spaces/mamba <https://github.com/state-spaces/mamba>`_ at
``8ffd905c91d207f5c0cc84fc2a2fb748655094f0`` (v2.2.2), under Apache-2.0.
Both notices are retained in ``LICENSES/EEGMamba.txt`` and
``LICENSES/Mamba.txt``. This implementation changes the execution backend
to differentiable PyTorch operations; it needs no CUDA extensions.

Download ``pretrained_EEGMamba.pth`` from the
`official Hub <https://huggingface.co/weighting666/EEGMamba>`_, revision
``0b060d87acd6f23bf1d0b852bf1726064f335f97``. The Hub card specifies MIT.
Its SHA256 is
``b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe``.
The complete checkpoint has 117 tensors and 3,317,443 state elements,
including the frozen mask encoding. Each Mamba2 block has width 200,
inner width 400, eight 50-wide heads, state size 64, depthwise causal
convolution width 4 and a gated RMSNorm.

.. code-block:: python

   import torch
   from torcheeg.models import EEGMamba

   model = EEGMamba.from_pretrained('./pretrained_EEGMamba.pth').eval()
   eeg = torch.randn(2, 19, 4, 200)  # synthetic API check
   with torch.no_grad():
       features = model.forward_features(eeg)  # [2,19,4,200]
       reconstruction = model(eeg)            # [2,19,4,200]

Loading is local and uses ``weights_only=True`` on CPU. No download,
remote code, unrestricted pickle fallback or partial matching occurs.
All keys, shapes, dtypes and finite values are required, including the
reconstruction head. The official checkpoint is an unprefixed flat tensor
state. Complete safetensors files are supported with the ``foundation``
extra. An explicit common key prefix can be removed with ``prefix``;
unprefixed extras are rejected.

Inputs and ordering
-------------------

The floating input shape is ``[batch,electrodes,patches,200]`` at 200 Hz.
The model does not infer filtering, signal units, reference, montage,
channel order or population normalization. Keep those settings explicit.
The patch embedding sums temporal convolution and magnitude-rFFT features,
then adds a depthwise spatial/temporal convolution. It flattens patches in
electrode-major order. Both the mixer output and residual sequence reverse
**after every Mamba2 layer**, matching EEGMamba's alternating traversal.
Changing channel order changes this computation.

``block_size`` sets the SSD computation block length, without adding or
removing model parameters. The default is 64. Partial last blocks are
zero-padded internally and removed from the output. Intra-block computation
uses stable segment sums; inter-block states carry the recurrence forward.
It retains the complete Mamba2 state and is not a shorter substitute model.
Floating-point accumulation order can differ between block sizes and from
the original fused kernels. CPU numerical and gradient checks use the
published SSD listing and original model bodies. They do not establish
CUDA/Triton execution, GPU throughput or bitwise equality to fused kernels.

Masked reconstruction and downstream training
--------------------------------------------

``mask`` is boolean ``[batch,electrodes,patches]`` on the input device.
True patches are replaced with the checkpoint's frozen mask vector before
both temporal and frequency embedding. Their raw values and gradients do
not leak into the prediction. Outputs keep batch and singleton axes.
The pretraining target remains the clean, unmasked input:

.. code-block:: python

   model.train()
   mask = torch.zeros(eeg.shape[:-1], dtype=torch.bool, device=eeg.device)
   mask[:, :, 0] = True  # this example always selects at least one patch
   prediction = model(eeg, mask)
   loss = torch.nn.functional.mse_loss(prediction[mask], eeg[mask])
   loss.backward()

The model returns predictions, not a hidden training loss. A caller can
choose the author's masked-only MSE or an explicit alternative objective.
For classification, use a separate head:

.. code-block:: python

   from torcheeg.models import EEGMambaClassifier

   classifier = EEGMambaClassifier(
       model, num_electrodes=19, num_patches=4, num_classes=2,
       freeze_backbone=True
   )
   logits = classifier(eeg)  # [2,2], pass directly to cross-entropy

A frozen backbone stays in eval mode when the classifier trains. Its full
state, including the unused reconstruction head, is kept for export.
With ``freeze_backbone=False``, feature-encoder parameters receive gradients;
the reconstruction head is unused by the classifier and has no gradient.
For custom pretraining, constructors permit another layer count or state
size. Those models are distinct from the complete official configuration.

Modern execution and independent example
----------------------------------------

The complete official model was checked on Torch 2.6 CPU with
``torch.compile(backend='aot_eager', fullgraph=True)``. Outputs, EEG
gradients and all parameter gradients agree with eager execution. That
fixed-shape eval check is a compatibility check, not an optimized-backend
performance measurement.

See :doc:`../auto_examples/examples_eegmat_eegmamba` for an independent
real EEGMat workflow. It holds out the last complete subject and trains
only a new classifier on all remaining subjects. Preprocessing is
whole-record resampling to 200 Hz, then FIR 0.3–75 Hz/notch 50 Hz,
microvolts divided by 100, and four one-second patches per window.
The exact EEGMat source channel order and reference are preserved.
This teaching protocol does not claim to reproduce the original
pretraining montage, record trimming or published benchmark scores.

The example exports all backbone/classifier state, model configuration,
source/checkpoint hashes and held-out predictions. It strictly restores
and requires identical predictions. Gallery rendering does not run this
training workflow; execute the script separately with real EDF files.
