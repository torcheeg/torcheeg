CBraMod fine-tuning recipes
===========================

Choose which parameters to train before you create the optimizer.
Use :class:`torcheeg.models.CBraModClassifier` with an explicit backbone
configuration and an optional complete local tensor checkpoint. These
recipes also explain the distinction between freezing parameters and
changing module training modes. They do not add new model architectures.

.. list-table:: Parameter choices
   :header-rows: 1
   :widths: 18 42 40

   * - Recipe
     - Backbone parameters
     - Backbone mode during training
   * - Head only
     - Freeze all parameters.
     - Keep evaluation mode with ``freeze_encoder=True``.
   * - Full fine-tuning
     - Keep the native trainable parameters. Keep the fixed mask unchanged.
     - Use training mode with ``freeze_encoder=False``.
   * - Bias tuning
     - Train ordinary biases and packed attention projection biases.
     - Use training mode with ``freeze_encoder=False``.
   * - Last blocks
     - Train the last N complete criss-cross blocks. Freeze the stem and other blocks.
     - Use training mode with ``freeze_encoder=False``.

All four recipes train the complete task head. Use a new model, load the
backbone checkpoint first, and then select parameters. The examples use the
native flatten, 4H, H, classes head with ELU and dropout. The backbone uses
GroupNorm and LayerNorm; it has no running BatchNorm statistics. In the
partial recipes, frozen early layers still use training dropout. This is
a deliberate mode choice, not a promise of identical stochastic outputs.

Head only and full fine-tuning
------------------------------

For a frozen feature extractor, pass ``freeze_encoder=True`` to the task
constructor. The wrapper keeps the complete backbone in evaluation mode
even after ``model.train()``. Input gradients are still available.
See :doc:`/auto_examples/examples_eegmat_cbramod_classifier`.

For full fine-tuning, construct a fresh backbone and pass
``freeze_encoder=False``. Do not call ``requires_grad_(True)`` on the entire
backbone: that would also enable its fixed ``patch_embedding.mask_encoding``.
The native reconstruction projection is retained for complete checkpoint
compatibility. Classification does not use it, so it receives no gradient.
See :doc:`/auto_examples/examples_eegmat_cbramod_finetune`.

Bias tuning
------------

The bias recipe applies the parameter selection idea of
`BitFit, Ben Zaken et al. (ACL 2022) <https://aclanthology.org/2022.acl-short.1/>`_
to CBraMod. It is not a reproduction of that paper's language-model results.
It enables backbone names ending in ``.bias`` or ``.in_proj_bias`` and
freezes all other backbone tensors, including normalization scales and the
fixed mask. Packed spatial and temporal attention biases must be included.
The separate task head remains fully trainable.

The example's ``configure_trainable_parameters`` function shows this short
standard PyTorch recipe directly. It rejects a task with
``freeze_encoder=True`` before changing parameter flags. See
:doc:`/auto_examples/examples_eegmat_cbramod_bitfit`.

Last encoder blocks
--------------------

The block recipe freezes the complete backbone, then enables the last N
registered ``backbone.encoder.layers`` modules. The patch embedding,
positional encoding, earlier blocks, fixed mask and unused reconstruction
projection remain frozen. N must be an integer from one to the encoder
depth. Boolean, negative, zero and excessive values raise an error before
changing parameter flags. This replaces the archived helper's partial
mutation and ambiguous negative indexing on invalid input.

The standalone example defaults to the last two blocks. Its small random
backbone has two blocks; the canonical author backbone has twelve. Set
``--num-unfrozen-blocks 1`` to train only the final block in either model.
See :doc:`/auto_examples/examples_eegmat_cbramod_last_blocks`.

Complete restoration
---------------------

Each example saves the complete tensor state, backbone/task/tuning
configurations, selected parameter names, preprocessing, source and target
channel orders, subject partition and all held-out logits. To restore:

1. Safely read the local artifact with ``torch.load(..., weights_only=True)``.
2. Rebuild ``CBraMod`` and ``CBraModClassifier`` from their saved configurations.
3. Reapply the same bias or block recipe when needed. Use the saved block count.
4. Load the complete state with ``strict=True`` and compare outputs in evaluation mode.

Tensor state does not store ``requires_grad`` flags. Loading only the tensor
state restores predictions but does not restore a partial training policy.
Create the optimizer after selecting parameters. These example artifacts
do not contain optimizer state and do not resume an interrupted optimizer
trajectory.

You can inspect parameter counts with standard PyTorch::

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    print(dict(total=total, trainable=trainable))

Data and result limits
-----------------------

All examples use real EEGMat recordings when supplied by the user. The
default backbone is small and random; a local canonical author checkpoint
is optional. No source data or weights are downloaded by the scripts.
Training and test subjects remain separate. Epochs are fixed, and held-out
labels are not used to select settings.

The source is filtered as complete 256 Hz recordings, then divided into
four-second windows in 100-microvolt units. Each one-second patch is
resampled separately from 256 to 200 samples, with an explicit 19-channel
order. This differs from the author's full-recording 200 Hz preprocessing
and does not reproduce an EEG-Arena protocol. A one-epoch teaching run is
functional evidence, not a benchmark score. See :doc:`cbramod_heads` for the
physical-time contract.
